"""
Chunking Module - Hierarchical document chunking for auto-merging retrieval.

This module implements:
1. Two-level hierarchical chunking (small/large)
2. Parent-child relationships for auto-merging
3. Metadata preservation across chunk levels
4. **Sentence-aware chunking** to prevent mid-sentence splits

Chunk Size Strategy:
| Level  | Size (tokens) | Overlap | Purpose                      |
|--------|---------------|---------|------------------------------|
| Large  | 1024          | 100     | Full sections, broad context |
| Small  | 256           | 100     | Individual facts, precision  |

Why hierarchical chunking?
- Small chunks: Capture precise facts (amounts, dates, IDs)
- Large chunks: Full sections for comprehensive understanding
- Auto-merging: Start with small, merge up when more context needed

Why sentence-aware chunking?
- Token-based chunking can split mid-sentence (e.g., "Gold Rolex (ref." | "116618LB)")
- Sentence-aware ensures semantic units stay intact
- Critical for needle queries that need precise facts

Note: Two levels are sufficient for this insurance claims project.
"""

import logging
import uuid
from typing import List, Tuple, Optional

from llama_index.core import Document
from llama_index.core.node_parser import HierarchicalNodeParser, get_leaf_nodes, SentenceSplitter
from llama_index.core.schema import TextNode, BaseNode, NodeRelationship, RelatedNodeInfo
from llama_index.core.storage.docstore import SimpleDocumentStore

from src.config import CHUNK_SIZES, CHUNK_OVERLAP

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def create_hierarchical_nodes(
    documents: List[Document],
    chunk_sizes: List[int] = None,
    chunk_overlap: int = None,
    sentence_aware: bool = True
) -> Tuple[List[BaseNode], List[BaseNode], SimpleDocumentStore]:
    """
    Parse documents into hierarchical nodes for auto-merging retrieval.

    Args:
        documents: List of LlamaIndex Document objects
        chunk_sizes: List of chunk sizes [large, small] (default: [1024, 256])
        chunk_overlap: Overlap between chunks (default: 100)
        sentence_aware: If True, use sentence-aware chunking (default: True)

    Returns:
        Tuple of (all_nodes, leaf_nodes, docstore)
        - all_nodes: All nodes at all levels
        - leaf_nodes: Only the smallest chunks (for indexing)
        - docstore: Document store with all nodes (for auto-merging)

    Why this approach?
    - Leaf nodes (small) are indexed for precise retrieval
    - Parent nodes (large) stored in docstore
    - Auto-merging retriever can "merge up" when needed

    Why sentence_aware=True (default)?
    - Prevents splitting mid-sentence (e.g., "Rolex (ref." | "116618LB)")
    - Critical for needle queries needing precise facts
    """
    chunk_sizes = chunk_sizes or CHUNK_SIZES
    chunk_overlap = chunk_overlap or CHUNK_OVERLAP

    if sentence_aware:
        return _create_sentence_aware_hierarchical_nodes(documents, chunk_sizes, chunk_overlap)
    else:
        return _create_token_based_hierarchical_nodes(documents, chunk_sizes, chunk_overlap)


def _create_token_based_hierarchical_nodes(
    documents: List[Document],
    chunk_sizes: List[int],
    chunk_overlap: int
) -> Tuple[List[BaseNode], List[BaseNode], SimpleDocumentStore]:
    """Original token-based hierarchical chunking (may split mid-sentence)."""
    logger.info(f"Creating token-based hierarchical chunks: sizes={chunk_sizes}, overlap={chunk_overlap}")

    # Create hierarchical node parser
    node_parser = HierarchicalNodeParser.from_defaults(
        chunk_sizes=chunk_sizes,
        chunk_overlap=chunk_overlap
    )

    # Parse documents into hierarchical nodes
    all_nodes = node_parser.get_nodes_from_documents(documents)
    logger.info(f"Created {len(all_nodes)} total nodes")

    # Get only leaf nodes (smallest chunks) for indexing
    leaf_nodes = get_leaf_nodes(all_nodes)
    logger.info(f"Extracted {len(leaf_nodes)} leaf nodes")

    # Add level metadata to nodes for debugging
    _annotate_node_levels(all_nodes, chunk_sizes)

    # Create document store with ALL nodes (needed for auto-merging)
    docstore = SimpleDocumentStore()
    docstore.add_documents(all_nodes)
    logger.info(f"Added {len(all_nodes)} nodes to docstore")

    # Log statistics
    _log_node_statistics(all_nodes, chunk_sizes)

    return all_nodes, leaf_nodes, docstore


def _create_sentence_aware_hierarchical_nodes(
    documents: List[Document],
    chunk_sizes: List[int],
    chunk_overlap: int
) -> Tuple[List[BaseNode], List[BaseNode], SimpleDocumentStore]:
    """
    Create hierarchical nodes using sentence-aware chunking.

    This ensures parent chunks never split mid-sentence, which prevents
    critical information from being separated across chunks.

    Approach:
    1. Use SentenceSplitter for parent (large) chunks - respects sentence boundaries
    2. Create child (small) chunks from parents - can use token-based since
       they're within sentence boundaries of the parent
    3. Manually establish parent-child relationships for auto-merging
    """
    logger.info(f"Creating SENTENCE-AWARE hierarchical chunks: sizes={chunk_sizes}, overlap={chunk_overlap}")

    large_chunk_size = chunk_sizes[0]  # e.g., 1024
    small_chunk_size = chunk_sizes[1] if len(chunk_sizes) > 1 else 256

    # Step 1: Create parent chunks using SentenceSplitter (respects sentence boundaries)
    parent_splitter = SentenceSplitter(
        chunk_size=large_chunk_size,
        chunk_overlap=chunk_overlap,
        paragraph_separator="\n\n",
        secondary_chunking_regex="[^,.;。？！]+[,.;。？！]?",  # Fallback: split on punctuation
    )

    # Step 2: Create child chunk splitter (smaller chunks from parents)
    child_splitter = SentenceSplitter(
        chunk_size=small_chunk_size,
        chunk_overlap=chunk_overlap // 2,  # Less overlap for children
        paragraph_separator="\n",
    )

    all_nodes = []
    leaf_nodes = []

    for doc in documents:
        # Create parent nodes from document
        parent_nodes = parent_splitter.get_nodes_from_documents([doc])

        # Track previous parent for NEXT/PREVIOUS relationships
        prev_parent = None

        for parent_node in parent_nodes:
            # Ensure parent has unique ID
            parent_node.id_ = str(uuid.uuid4())

            # Copy document metadata to parent
            parent_node.metadata = {**doc.metadata, **parent_node.metadata}
            parent_node.metadata['chunk_level'] = 'large'
            parent_node.metadata['approx_tokens'] = len(parent_node.text) // 4

            # Set source relationship
            parent_node.relationships[NodeRelationship.SOURCE] = RelatedNodeInfo(
                node_id=doc.doc_id or doc.id_,
                metadata=doc.metadata
            )

            # Set PREVIOUS/NEXT relationships between parents
            if prev_parent:
                parent_node.relationships[NodeRelationship.PREVIOUS] = RelatedNodeInfo(
                    node_id=prev_parent.id_
                )
                prev_parent.relationships[NodeRelationship.NEXT] = RelatedNodeInfo(
                    node_id=parent_node.id_
                )

            # Create child nodes from this parent
            parent_doc = Document(text=parent_node.text, metadata=parent_node.metadata)
            child_nodes = child_splitter.get_nodes_from_documents([parent_doc])

            # Track previous child for relationships
            prev_child = None
            child_ids = []

            for child_node in child_nodes:
                # Ensure child has unique ID
                child_node.id_ = str(uuid.uuid4())
                child_ids.append(child_node.id_)

                # Copy metadata and set level
                child_node.metadata = {**parent_node.metadata}
                child_node.metadata['chunk_level'] = 'small'
                child_node.metadata['approx_tokens'] = len(child_node.text) // 4

                # Set PARENT relationship (critical for auto-merging!)
                child_node.relationships[NodeRelationship.PARENT] = RelatedNodeInfo(
                    node_id=parent_node.id_
                )

                # Set SOURCE relationship
                child_node.relationships[NodeRelationship.SOURCE] = RelatedNodeInfo(
                    node_id=doc.doc_id or doc.id_,
                    metadata=doc.metadata
                )

                # Set PREVIOUS/NEXT relationships between siblings
                if prev_child:
                    child_node.relationships[NodeRelationship.PREVIOUS] = RelatedNodeInfo(
                        node_id=prev_child.id_
                    )
                    prev_child.relationships[NodeRelationship.NEXT] = RelatedNodeInfo(
                        node_id=child_node.id_
                    )

                prev_child = child_node
                all_nodes.append(child_node)
                leaf_nodes.append(child_node)

            # Set CHILD relationship on parent (list of child IDs)
            if child_ids:
                parent_node.relationships[NodeRelationship.CHILD] = [
                    RelatedNodeInfo(node_id=cid) for cid in child_ids
                ]

            all_nodes.append(parent_node)
            prev_parent = parent_node

    logger.info(f"Created {len(all_nodes)} total nodes (sentence-aware)")
    logger.info(f"  - Parent nodes: {len([n for n in all_nodes if n.metadata.get('chunk_level') == 'large'])}")
    logger.info(f"  - Child nodes: {len(leaf_nodes)}")

    # Create document store with ALL nodes (needed for auto-merging)
    docstore = SimpleDocumentStore()
    docstore.add_documents(all_nodes)
    logger.info(f"Added {len(all_nodes)} nodes to docstore")

    # Log statistics
    _log_node_statistics(all_nodes, chunk_sizes)

    return all_nodes, leaf_nodes, docstore


def _annotate_node_levels(nodes: List[BaseNode], chunk_sizes: List[int]):
    """
    Add 'level' metadata to nodes based on their approximate size.
    
    Levels (two-level hierarchy):
    - 'large': ~1024 tokens (parent chunks)
    - 'small': ~256 tokens (leaf nodes)
    """
    for node in nodes:
        # Estimate level based on text length (rough approximation)
        text_len = len(node.text)
        
        # Approximate tokens to characters ratio is ~4 chars per token
        approx_tokens = text_len / 4
        
        # Assign level based on size (two levels)
        # If larger than the smallest chunk size, it's a large chunk
        if len(chunk_sizes) >= 2 and approx_tokens > chunk_sizes[1]:
            level = 'large'
        else:
            level = 'small'
        
        node.metadata['chunk_level'] = level
        node.metadata['approx_tokens'] = int(approx_tokens)


def _log_node_statistics(nodes: List[BaseNode], chunk_sizes: List[int]):
    """Log statistics about created nodes."""
    levels = {'large': 0, 'small': 0, 'unknown': 0}
    
    for node in nodes:
        level = node.metadata.get('chunk_level', 'unknown')
        levels[level] = levels.get(level, 0) + 1
    
    logger.info("Node distribution by level:")
    for level, count in levels.items():
        if count > 0:
            logger.info(f"  {level}: {count} nodes")


def get_node_hierarchy_info(node: BaseNode, docstore: SimpleDocumentStore) -> dict:
    """
    Get hierarchy information for a node.
    
    Args:
        node: The node to get info for
        docstore: The document store containing all nodes
    
    Returns:
        Dictionary with hierarchy info
    """
    info = {
        'node_id': node.node_id,
        'level': node.metadata.get('chunk_level', 'unknown'),
        'text_preview': node.text[:100] + '...' if len(node.text) > 100 else node.text,
        'has_parent': False,
        'parent_id': None,
        'children_count': 0
    }
    
    # Check for parent
    if hasattr(node, 'parent_node') and node.parent_node:
        info['has_parent'] = True
        info['parent_id'] = node.parent_node.node_id
    
    # Check for children
    if hasattr(node, 'child_nodes') and node.child_nodes:
        info['children_count'] = len(node.child_nodes)
    
    return info


def preview_chunks(documents: List[Document], num_chunks: int = 5) -> str:
    """
    Create a preview of how documents will be chunked.
    
    Args:
        documents: Documents to preview chunking for
        num_chunks: Number of chunks to show per level
    
    Returns:
        Formatted string preview
    """
    all_nodes, leaf_nodes, docstore = create_hierarchical_nodes(documents)
    
    lines = [
        "=" * 60,
        "📄 CHUNKING PREVIEW",
        "=" * 60,
        f"Total nodes: {len(all_nodes)}",
        f"Leaf nodes (indexed): {len(leaf_nodes)}",
        ""
    ]
    
    # Group by level
    by_level = {'large': [], 'small': []}
    for node in all_nodes:
        level = node.metadata.get('chunk_level', 'small')
        if level in by_level:
            by_level[level].append(node)
    
    # Show samples from each level
    for level in ['large', 'small']:
        nodes = by_level[level]
        lines.append(f"\n--- {level.upper()} CHUNKS ({len(nodes)} total) ---")
        
        for i, node in enumerate(nodes[:num_chunks]):
            claim_id = node.metadata.get('claim_id', 'Unknown')
            text_preview = node.text[:150].replace('\n', ' ')
            lines.append(f"\n  [{i+1}] Claim: {claim_id}")
            lines.append(f"      Length: ~{node.metadata.get('approx_tokens', '?')} tokens")
            lines.append(f"      Text: {text_preview}...")
    
    lines.append("\n" + "=" * 60)
    return "\n".join(lines)


# Quick test
if __name__ == "__main__":
    from src.config import validate_config
    from src.data_loader import load_claim_documents
    
    print("Testing chunking module...")
    
    try:
        validate_config()
        
        # Load just 2 documents for testing
        documents = load_claim_documents()[:2]
        
        # Create hierarchical chunks
        all_nodes, leaf_nodes, docstore = create_hierarchical_nodes(documents)
        
        print(f"\n✅ Created {len(all_nodes)} total nodes")
        print(f"✅ Created {len(leaf_nodes)} leaf nodes (for indexing)")
        
        # Show preview
        if documents:
            print("\n" + "=" * 40)
            print("Sample leaf node:")
            sample = leaf_nodes[0]
            print(f"  Claim: {sample.metadata.get('claim_id', 'Unknown')}")
            print(f"  Level: {sample.metadata.get('chunk_level', 'Unknown')}")
            print(f"  Text preview: {sample.text[:200]}...")
        
    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback
        traceback.print_exc()

