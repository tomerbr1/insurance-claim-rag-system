"""
Indexing Module - Build Summary Index and Hierarchical Vector Index.

This module implements:
1. Summary Index with MapReduce (stores intermediate summaries)
2. Hierarchical Vector Index with ChromaDB
3. Persistence for all indexes (ChromaDB + JSON)

Summary Index Architecture (from instructor feedback):
- Chunk-level summaries (10-20 per claim) - stored as nodes
- Section-level summaries (3-5 per claim) - stored as nodes  
- Document-level summaries (1 per claim) - stored as nodes
All levels are vectorized and retrievable for multi-granularity queries.

Hierarchical Vector Index:
- Stores only leaf nodes (small chunks)
- ChromaDB for vector storage
- Used with auto-merging retriever

Persistence:
- Summary nodes: Stored in ChromaDB collection "insurance_claims_summaries"
- Docstore: Persisted to JSON file for auto-merging support
"""

import logging
import json
from pathlib import Path
from typing import List, Tuple, Dict, Optional

from llama_index.core import VectorStoreIndex, StorageContext, Settings
from llama_index.core.schema import TextNode, BaseNode
from llama_index.core.storage.docstore import SimpleDocumentStore
from llama_index.vector_stores.chroma import ChromaVectorStore
from llama_index.llms.openai import OpenAI
from llama_index.embeddings.openai import OpenAIEmbedding
import chromadb

from src.config import (
    CHROMA_DIR, CHROMA_COLLECTION_NAME, 
    OPENAI_API_KEY, OPENAI_MODEL, OPENAI_EMBEDDING_MODEL
)

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Persistence constants
SUMMARY_COLLECTION_NAME = "insurance_claims_summaries"
DOCSTORE_FILE = "docstore.json"
SUMMARY_NODES_FILE = "summary_nodes.json"


# Prompts for summary generation
CHUNK_SUMMARY_PROMPT = """
Summarize this insurance claim excerpt in 2-3 sentences.
Focus on: dates, amounts, parties involved, and key events.
Be precise with numbers and names.

Text:
{text}

Summary:
"""

SECTION_SUMMARY_PROMPT = """
Combine these chunk summaries into a cohesive section summary (3-5 sentences).
Maintain chronological order and key details.

Chunk summaries:
{summaries}

Section Summary:
"""

DOCUMENT_SUMMARY_PROMPT = """
Create a comprehensive document summary (1 paragraph) from these section summaries.
Include: claim ID, claim type, timeline overview, outcome, and total amount.

Section summaries:
{summaries}

Document Summary:
"""


def build_summary_index(
    nodes: List[BaseNode],
    llm: Optional[OpenAI] = None,
    save_intermediate: bool = True
) -> Tuple[VectorStoreIndex, List[TextNode]]:
    """
    Build summary index with MapReduce, storing intermediate summaries.
    
    Architecture (instructor approved):
    1. MAP: Summarize each chunk
    2. Store chunk summaries as retrievable nodes
    3. REDUCE: Combine into section summaries
    4. Store section summaries as retrievable nodes
    5. Final document-level summary
    
    Args:
        nodes: All hierarchical nodes (from chunking)
        llm: LLM for summarization (default: GPT-4)
        save_intermediate: Whether to store intermediate summaries (default: True)
    
    Returns:
        Tuple of (summary_index, summary_nodes)
    """
    llm = llm or OpenAI(model=OPENAI_MODEL, api_key=OPENAI_API_KEY, temperature=0)
    
    logger.info("=" * 80)
    logger.info("BUILDING SUMMARY INDEX - MapReduce Architecture")
    logger.info("=" * 80)
    logger.info("This process makes multiple LLM calls to create multi-level summaries:")
    logger.info("  1. CHUNK LEVEL: Each small chunk → summary (most API calls here)")
    logger.info("  2. SECTION LEVEL: Group of chunk summaries → section summary")
    logger.info("  3. DOCUMENT LEVEL: Section summaries → final document summary")
    logger.info("")
    logger.info("Why? Enables multi-granularity retrieval for different query types.")
    logger.info("=" * 80)
    
    # Get leaf nodes only (small chunks)
    leaf_nodes = [n for n in nodes if n.metadata.get('chunk_level') == 'small']
    logger.info(f"Processing {len(leaf_nodes)} leaf nodes")
    
    summary_nodes = []
    
    # Group nodes by claim_id
    claims = {}
    for node in leaf_nodes:
        claim_id = node.metadata.get('claim_id', 'unknown')
        if claim_id not in claims:
            claims[claim_id] = []
        claims[claim_id].append(node)
    
    logger.info(f"Found {len(claims)} unique claims")
    logger.info("")
    
    # Process each claim
    claim_num = 0
    total_claims = len(claims)
    total_chunks_processed = 0
    
    for claim_id, claim_nodes in claims.items():
        claim_num += 1
        logger.info("-" * 80)
        logger.info(f"📋 CLAIM {claim_num}/{total_claims}: {claim_id}")
        logger.info(f"   {len(claim_nodes)} chunks to summarize")
        logger.info("-" * 80)
        
        # Phase 1: MAP - Summarize each chunk
        logger.info(f"PHASE 1 (MAP): Creating chunk-level summaries...")
        logger.info(f"Next {len(claim_nodes)} HTTP POST requests = chunk summarization calls")
        logger.info("")
        
        chunk_summaries = []
        for i, node in enumerate(claim_nodes):
            total_chunks_processed += 1
            
            prompt = CHUNK_SUMMARY_PROMPT.format(text=node.text[:1500])
            
            try:
                summary = llm.complete(prompt).text.strip()
                
                if save_intermediate:
                    # Create summary node (retrievable!)
                    summary_node = TextNode(
                        text=summary,
                        metadata={
                            'claim_id': claim_id,
                            'source_file': node.metadata.get('source_file'),
                            'node_type': 'chunk_summary',
                            'summary_level': 'chunk',
                            'original_node_id': node.node_id,
                            'chunk_index': i
                        }
                    )
                    summary_nodes.append(summary_node)
                
                chunk_summaries.append(summary)
                
            except Exception as e:
                logger.error(f"Error summarizing chunk {i}: {e}")
                chunk_summaries.append(f"[Chunk {i}] {node.text[:200]}...")
        
        # Phase 2: REDUCE - Create section summary
        # For simplicity, treating all chunks of a claim as one "section"
        # In production, you'd detect actual document sections
        logger.info("")
        logger.info(f"PHASE 2 (REDUCE): Creating section-level summary for {claim_id}")
        logger.info(f"Next 1 HTTP POST request = section summary (combines {len(chunk_summaries)} chunk summaries)")
        logger.info("")
        
        combined_chunk_summaries = "\n\n".join([
            f"- {s}" for s in chunk_summaries
        ])
        
        section_prompt = SECTION_SUMMARY_PROMPT.format(summaries=combined_chunk_summaries)
        
        try:
            section_summary = llm.complete(section_prompt).text.strip()
            
            if save_intermediate:
                section_node = TextNode(
                    text=section_summary,
                    metadata={
                        'claim_id': claim_id,
                        'node_type': 'section_summary',
                        'summary_level': 'section'
                    }
                )
                summary_nodes.append(section_node)
                
        except Exception as e:
            logger.error(f"Error creating section summary: {e}")
            section_summary = combined_chunk_summaries[:500]
        
        # Phase 3: Final document summary
        logger.info("")
        logger.info(f"PHASE 3 (FINAL): Creating document-level summary for {claim_id}")
        logger.info(f"Next 1 HTTP POST request = document summary (final aggregation)")
        logger.info("")
        
        doc_prompt = DOCUMENT_SUMMARY_PROMPT.format(summaries=section_summary)
        
        try:
            doc_summary = llm.complete(doc_prompt).text.strip()
            
            doc_node = TextNode(
                text=doc_summary,
                metadata={
                    'claim_id': claim_id,
                    'node_type': 'document_summary',
                    'summary_level': 'document'
                }
            )
            summary_nodes.append(doc_node)
            
        except Exception as e:
            logger.error(f"Error creating document summary: {e}")
        
        logger.info(f"✅ Completed summaries for {claim_id}")
        logger.info("")
    
    logger.info("=" * 80)
    logger.info(f"SUMMARY INDEX BUILD COMPLETE")
    logger.info(f"Created {len(summary_nodes)} summary nodes total")
    
    # Count by level
    levels = {}
    for node in summary_nodes:
        level = node.metadata.get('summary_level', 'unknown')
        levels[level] = levels.get(level, 0) + 1
    logger.info(f"Summary distribution: {levels}")
    
    # Build vector index from summary nodes
    logger.info("Building vector index for summaries...")
    
    embedding_model = OpenAIEmbedding(
        model=OPENAI_EMBEDDING_MODEL,
        api_key=OPENAI_API_KEY
    )
    
    Settings.embed_model = embedding_model
    
    summary_index = VectorStoreIndex(summary_nodes)
    
    # Persist summary nodes for later reuse
    logger.info("Persisting summary index for future reuse...")
    persist_summary_index(summary_nodes)
    
    logger.info("✅ Summary index built and persisted successfully")
    return summary_index, summary_nodes


def build_hierarchical_index(
    leaf_nodes: List[BaseNode],
    docstore: SimpleDocumentStore,
    persist_dir: Optional[Path] = None,
    collection_name: Optional[str] = None
) -> VectorStoreIndex:
    """
    Build hierarchical vector index using ChromaDB.
    
    This index stores only leaf nodes (small chunks) but the docstore
    maintains parent-child relationships for auto-merging.
    
    Args:
        leaf_nodes: Leaf nodes from hierarchical chunking
        docstore: Document store with all nodes (for auto-merging)
        persist_dir: Directory for ChromaDB persistence
        collection_name: ChromaDB collection name
    
    Returns:
        VectorStoreIndex configured for auto-merging retrieval
    """
    persist_dir = persist_dir or CHROMA_DIR
    collection_name = collection_name or CHROMA_COLLECTION_NAME
    
    logger.info(f"Building hierarchical index with ChromaDB: {collection_name}")
    logger.info(f"Persist directory: {persist_dir}")
    
    # Setup ChromaDB
    persist_dir.mkdir(parents=True, exist_ok=True)
    
    chroma_client = chromadb.PersistentClient(path=str(persist_dir))
    
    # Delete existing collection if it exists (for fresh builds)
    try:
        chroma_client.delete_collection(collection_name)
        logger.info(f"Deleted existing collection: {collection_name}")
    except Exception:
        pass
    
    chroma_collection = chroma_client.create_collection(
        name=collection_name,
        metadata={"description": "Insurance claims hierarchical index"}
    )
    
    # Create vector store
    vector_store = ChromaVectorStore(chroma_collection=chroma_collection)
    
    # Create storage context with docstore (important for auto-merging!)
    storage_context = StorageContext.from_defaults(
        vector_store=vector_store,
        docstore=docstore
    )
    
    # Setup embedding model
    embedding_model = OpenAIEmbedding(
        model=OPENAI_EMBEDDING_MODEL,
        api_key=OPENAI_API_KEY
    )
    
    Settings.embed_model = embedding_model
    
    # Build index from leaf nodes only
    logger.info(f"Indexing {len(leaf_nodes)} leaf nodes...")
    
    index = VectorStoreIndex(
        nodes=leaf_nodes,
        storage_context=storage_context
    )
    
    # Persist docstore for later reuse (needed for auto-merging)
    logger.info("Persisting docstore for future reuse...")
    persist_docstore(docstore, persist_dir)
    
    logger.info(f"✅ Hierarchical index built and persisted with {len(leaf_nodes)} leaf nodes")
    return index


def get_index_statistics(
    summary_index: VectorStoreIndex,
    hierarchical_index: VectorStoreIndex
) -> Dict:
    """
    Get statistics about built indexes.
    
    Returns:
        Dictionary with index statistics
    """
    stats = {
        'summary_index': {
            'type': 'VectorStoreIndex',
            'description': 'Multi-level summaries (chunk/section/document)'
        },
        'hierarchical_index': {
            'type': 'VectorStoreIndex with ChromaDB',
            'description': 'Leaf nodes for auto-merging retrieval'
        }
    }
    
    return stats


# =============================================================================
# PERSISTENCE FUNCTIONS
# =============================================================================

def persist_summary_index(
    summary_nodes: List[TextNode],
    persist_dir: Optional[Path] = None
) -> bool:
    """
    Persist summary nodes to ChromaDB for later reloading.
    
    This stores the summary nodes (chunk/section/document summaries) in a 
    separate ChromaDB collection so they can be reused without re-running
    the expensive MapReduce summarization.
    
    Args:
        summary_nodes: List of summary TextNodes to persist
        persist_dir: Directory for ChromaDB persistence
    
    Returns:
        True if successful
    """
    persist_dir = persist_dir or CHROMA_DIR
    persist_dir.mkdir(parents=True, exist_ok=True)
    
    logger.info(f"Persisting {len(summary_nodes)} summary nodes to ChromaDB...")
    
    try:
        # Setup ChromaDB
        chroma_client = chromadb.PersistentClient(path=str(persist_dir))
        
        # Delete existing summary collection if it exists
        try:
            chroma_client.delete_collection(SUMMARY_COLLECTION_NAME)
            logger.info(f"Deleted existing summary collection")
        except Exception:
            pass
        
        # Create new collection
        summary_collection = chroma_client.create_collection(
            name=SUMMARY_COLLECTION_NAME,
            metadata={"description": "Insurance claims summary nodes"}
        )
        
        # Create vector store and index
        vector_store = ChromaVectorStore(chroma_collection=summary_collection)
        storage_context = StorageContext.from_defaults(vector_store=vector_store)
        
        # Setup embedding model
        embedding_model = OpenAIEmbedding(
            model=OPENAI_EMBEDDING_MODEL,
            api_key=OPENAI_API_KEY
        )
        Settings.embed_model = embedding_model
        
        # Build and persist index
        VectorStoreIndex(nodes=summary_nodes, storage_context=storage_context)
        
        # Also save nodes to JSON for metadata (ChromaDB doesn't preserve all metadata)
        nodes_json_path = persist_dir / SUMMARY_NODES_FILE
        nodes_data = []
        for node in summary_nodes:
            nodes_data.append({
                'node_id': node.node_id,
                'text': node.text,
                'metadata': node.metadata
            })
        
        with open(nodes_json_path, 'w') as f:
            json.dump(nodes_data, f, indent=2)
        
        logger.info(f"✅ Persisted {len(summary_nodes)} summary nodes")
        logger.info(f"   ChromaDB collection: {SUMMARY_COLLECTION_NAME}")
        logger.info(f"   JSON backup: {nodes_json_path}")
        
        return True
        
    except Exception as e:
        logger.error(f"Error persisting summary nodes: {e}")
        return False


def persist_docstore(
    docstore: SimpleDocumentStore,
    persist_dir: Optional[Path] = None
) -> bool:
    """
    Persist the document store to JSON for auto-merging support.

    The docstore contains all hierarchical nodes (small + large) and their
    parent-child relationships, which is needed for the auto-merging retriever.

    Args:
        docstore: SimpleDocumentStore with all nodes
        persist_dir: Directory for persistence

    Returns:
        True if successful
    """
    persist_dir = persist_dir or CHROMA_DIR
    persist_dir.mkdir(parents=True, exist_ok=True)

    docstore_path = persist_dir / DOCSTORE_FILE

    logger.info(f"Persisting docstore to {docstore_path}...")

    def serialize_relationship(v):
        """Serialize a relationship value (single or list of RelatedNodeInfo)."""
        if hasattr(v, 'node_id'):
            # Single RelatedNodeInfo
            return v.node_id
        elif isinstance(v, list):
            # List of RelatedNodeInfo (for CHILD relationships)
            return [item.node_id if hasattr(item, 'node_id') else str(item) for item in v]
        else:
            return str(v)

    try:
        # Get all documents from docstore
        all_docs = docstore.docs

        # Serialize to JSON-friendly format
        docstore_data = {
            'nodes': {}
        }

        for doc_id, doc in all_docs.items():
            # Serialize node data
            node_data = {
                'node_id': doc.node_id,
                'text': doc.text,
                'metadata': doc.metadata,
                'class_name': doc.class_name()
            }

            # Handle relationships (parent/child)
            if hasattr(doc, 'relationships') and doc.relationships:
                node_data['relationships'] = {
                    str(k): serialize_relationship(v)
                    for k, v in doc.relationships.items()
                }

            docstore_data['nodes'][doc_id] = node_data

        with open(docstore_path, 'w') as f:
            json.dump(docstore_data, f, indent=2)

        logger.info(f"✅ Persisted docstore with {len(docstore_data['nodes'])} nodes")
        return True

    except Exception as e:
        logger.error(f"Error persisting docstore: {e}")
        return False


def load_summary_index(
    persist_dir: Optional[Path] = None
) -> Tuple[Optional[VectorStoreIndex], Optional[List[TextNode]]]:
    """
    Load persisted summary index from ChromaDB.
    
    Args:
        persist_dir: Directory where ChromaDB is persisted
    
    Returns:
        Tuple of (summary_index, summary_nodes) or (None, None) if not found
    """
    persist_dir = persist_dir or CHROMA_DIR
    
    if not persist_dir.exists():
        logger.warning(f"Persist directory does not exist: {persist_dir}")
        return None, None
    
    try:
        # Check for ChromaDB collection
        chroma_client = chromadb.PersistentClient(path=str(persist_dir))
        collections = [c.name for c in chroma_client.list_collections()]
        
        if SUMMARY_COLLECTION_NAME not in collections:
            logger.warning(f"Summary collection not found: {SUMMARY_COLLECTION_NAME}")
            return None, None
        
        logger.info(f"Loading summary index from ChromaDB...")
        
        # Setup embedding model
        embedding_model = OpenAIEmbedding(
            model=OPENAI_EMBEDDING_MODEL,
            api_key=OPENAI_API_KEY
        )
        Settings.embed_model = embedding_model
        
        # Get collection and create vector store
        summary_collection = chroma_client.get_collection(SUMMARY_COLLECTION_NAME)
        vector_store = ChromaVectorStore(chroma_collection=summary_collection)
        
        # Create index from existing vector store
        summary_index = VectorStoreIndex.from_vector_store(vector_store)
        
        # Load nodes from JSON backup (for full metadata)
        summary_nodes = []
        nodes_json_path = persist_dir / SUMMARY_NODES_FILE
        
        if nodes_json_path.exists():
            with open(nodes_json_path, 'r') as f:
                nodes_data = json.load(f)
            
            for node_data in nodes_data:
                node = TextNode(
                    text=node_data['text'],
                    metadata=node_data['metadata'],
                    id_=node_data['node_id']
                )
                summary_nodes.append(node)
        
        logger.info(f"✅ Loaded summary index with {len(summary_nodes)} nodes")
        return summary_index, summary_nodes
        
    except Exception as e:
        logger.error(f"Error loading summary index: {e}")
        return None, None


def load_docstore(
    persist_dir: Optional[Path] = None
) -> Optional[SimpleDocumentStore]:
    """
    Load persisted docstore from JSON.

    Restores nodes WITH their parent-child relationships, which is
    critical for the auto-merging retriever to work.

    Args:
        persist_dir: Directory where docstore is persisted

    Returns:
        SimpleDocumentStore or None if not found
    """
    from llama_index.core.schema import NodeRelationship, RelatedNodeInfo

    persist_dir = persist_dir or CHROMA_DIR
    docstore_path = persist_dir / DOCSTORE_FILE

    if not docstore_path.exists():
        logger.warning(f"Docstore file not found: {docstore_path}")
        return None

    try:
        logger.info(f"Loading docstore from {docstore_path}...")

        with open(docstore_path, 'r') as f:
            docstore_data = json.load(f)

        # First pass: Create all nodes without relationships
        nodes_dict = {}
        for node_id, node_data in docstore_data['nodes'].items():
            node = TextNode(
                text=node_data['text'],
                metadata=node_data['metadata'],
                id_=node_data['node_id']
            )
            nodes_dict[node_data['node_id']] = node

        # Second pass: Restore relationships
        for node_id, node_data in docstore_data['nodes'].items():
            if 'relationships' in node_data and node_data['relationships']:
                node = nodes_dict[node_data['node_id']]
                relationships = {}

                for rel_type_str, related_value in node_data['relationships'].items():
                    # Convert string back to NodeRelationship enum
                    try:
                        rel_type = NodeRelationship(rel_type_str)
                    except ValueError:
                        # Try numeric conversion for older formats
                        try:
                            rel_type = NodeRelationship(int(rel_type_str))
                        except (ValueError, TypeError):
                            logger.warning(f"Unknown relationship type: {rel_type_str}")
                            continue

                    # Handle both single ID and list of IDs
                    if isinstance(related_value, list):
                        # List of child node IDs
                        relationships[rel_type] = [
                            RelatedNodeInfo(node_id=rid) for rid in related_value
                        ]
                    elif isinstance(related_value, str):
                        # Check if it's an old-format string representation of a list
                        if related_value.startswith('[') and 'RelatedNodeInfo' in related_value:
                            # Old format - skip and log warning
                            logger.warning(f"Found old-format relationship data for {node_data['node_id']}, skipping")
                            continue
                        # Single node ID
                        relationships[rel_type] = RelatedNodeInfo(node_id=related_value)
                    else:
                        logger.warning(f"Unexpected relationship value type: {type(related_value)}")
                        continue

                node.relationships = relationships

        # Add all nodes to docstore
        docstore = SimpleDocumentStore()
        docstore.add_documents(list(nodes_dict.values()))

        logger.info(f"✅ Loaded docstore with {len(nodes_dict)} nodes (relationships restored)")
        return docstore

    except Exception as e:
        logger.error(f"Error loading docstore: {e}")
        import traceback
        traceback.print_exc()
        return None


def load_hierarchical_index(
    docstore: SimpleDocumentStore,
    persist_dir: Optional[Path] = None,
    collection_name: Optional[str] = None
) -> Optional[VectorStoreIndex]:
    """
    Load persisted hierarchical index from ChromaDB.
    
    Args:
        docstore: The loaded docstore (needed for auto-merging)
        persist_dir: Directory where ChromaDB is persisted
        collection_name: ChromaDB collection name
    
    Returns:
        VectorStoreIndex or None if not found
    """
    persist_dir = persist_dir or CHROMA_DIR
    collection_name = collection_name or CHROMA_COLLECTION_NAME
    
    if not persist_dir.exists():
        logger.warning(f"Persist directory does not exist: {persist_dir}")
        return None
    
    try:
        # Check for ChromaDB collection
        chroma_client = chromadb.PersistentClient(path=str(persist_dir))
        collections = [c.name for c in chroma_client.list_collections()]
        
        if collection_name not in collections:
            logger.warning(f"Collection not found: {collection_name}")
            return None
        
        logger.info(f"Loading hierarchical index from ChromaDB...")
        
        # Setup embedding model
        embedding_model = OpenAIEmbedding(
            model=OPENAI_EMBEDDING_MODEL,
            api_key=OPENAI_API_KEY
        )
        Settings.embed_model = embedding_model
        
        # Get collection and create vector store
        collection = chroma_client.get_collection(collection_name)
        vector_store = ChromaVectorStore(chroma_collection=collection)
        
        # Create storage context with docstore
        storage_context = StorageContext.from_defaults(
            vector_store=vector_store,
            docstore=docstore
        )
        
        # Create index from existing vector store
        index = VectorStoreIndex.from_vector_store(
            vector_store,
            storage_context=storage_context
        )
        
        logger.info(f"✅ Loaded hierarchical index from {collection_name}")
        return index
        
    except Exception as e:
        logger.error(f"Error loading hierarchical index: {e}")
        return None


def load_all_indexes(
    persist_dir: Optional[Path] = None
) -> Tuple[Optional[VectorStoreIndex], Optional[VectorStoreIndex], 
           Optional[SimpleDocumentStore], Optional[List[TextNode]]]:
    """
    Load all persisted indexes at once.
    
    Args:
        persist_dir: Directory where data is persisted
    
    Returns:
        Tuple of (summary_index, hierarchical_index, docstore, summary_nodes)
        Any component may be None if not found/loadable
    """
    persist_dir = persist_dir or CHROMA_DIR
    
    # Load docstore first (needed for hierarchical index)
    docstore = load_docstore(persist_dir)
    
    # Load summary index
    summary_index, summary_nodes = load_summary_index(persist_dir)
    
    # Load hierarchical index (needs docstore)
    hierarchical_index = None
    if docstore:
        hierarchical_index = load_hierarchical_index(docstore, persist_dir)
    
    return summary_index, hierarchical_index, docstore, summary_nodes


# Quick test
if __name__ == "__main__":
    from src.config import validate_config
    from src.data_loader import load_claim_documents
    from src.chunking import create_hierarchical_nodes
    
    print("Testing indexing module...")
    
    try:
        validate_config()
        
        # Load just 2 documents for testing
        print("\n📄 Loading documents...")
        documents = load_claim_documents()[:2]
        
        # Create hierarchical chunks
        print("\n✂️  Creating chunks...")
        all_nodes, leaf_nodes, docstore = create_hierarchical_nodes(documents)
        
        # Build summary index
        print("\n📚 Building summary index...")
        summary_index, summary_nodes = build_summary_index(all_nodes)
        print(f"   Created {len(summary_nodes)} summary nodes")
        
        # Build hierarchical index
        print("\n🔍 Building hierarchical index...")
        hierarchical_index = build_hierarchical_index(leaf_nodes, docstore)
        
        # Test query on summary index
        print("\n🧪 Testing summary index query...")
        response = summary_index.as_query_engine().query(
            "What are the main claims in the system?"
        )
        print(f"   Response: {str(response)[:200]}...")
        
        print("\n✅ Indexing test complete!")
        
    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback
        traceback.print_exc()

