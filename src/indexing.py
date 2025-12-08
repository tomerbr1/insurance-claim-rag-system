"""
Indexing Module - Build Summary Index and Hierarchical Vector Index.

This module implements:
1. Summary Index with MapReduce (stores intermediate summaries)
2. Hierarchical Vector Index with ChromaDB

Summary Index Architecture (from instructor feedback):
- Chunk-level summaries (10-20 per claim) - stored as nodes
- Section-level summaries (3-5 per claim) - stored as nodes  
- Document-level summaries (1 per claim) - stored as nodes
All levels are vectorized and retrievable for multi-granularity queries.

Hierarchical Vector Index:
- Stores only leaf nodes (small chunks)
- ChromaDB for vector storage
- Used with auto-merging retriever
"""

import logging
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
    
    logger.info("✅ Summary index built successfully")
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
    
    logger.info(f"✅ Hierarchical index built with {len(leaf_nodes)} leaf nodes")
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

