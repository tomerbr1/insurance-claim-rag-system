"""
Chunking Module - Hierarchical document chunking for auto-merging retrieval.

This module implements:
1. Two-level hierarchical chunking (small/large)
2. Parent-child relationships for auto-merging
3. Metadata preservation across chunk levels

Chunk Size Strategy:
| Level  | Size (tokens) | Overlap | Purpose                      |
|--------|---------------|---------|------------------------------|
| Large  | 1024          | 20      | Full sections, broad context |
| Small  | 256           | 20      | Individual facts, precision  |

Why hierarchical chunking?
- Small chunks: Capture precise facts (amounts, dates, IDs)
- Large chunks: Full sections for comprehensive understanding
- Auto-merging: Start with small, merge up when more context needed

Note: Two levels are sufficient for this insurance claims project.
"""

import logging
from typing import List, Tuple

from llama_index.core import Document
from llama_index.core.node_parser import HierarchicalNodeParser, get_leaf_nodes
from llama_index.core.schema import TextNode, BaseNode
from llama_index.core.storage.docstore import SimpleDocumentStore

from src.config import CHUNK_SIZES, CHUNK_OVERLAP

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def create_hierarchical_nodes(
    documents: List[Document],
    chunk_sizes: List[int] = None,
    chunk_overlap: int = None
) -> Tuple[List[BaseNode], List[BaseNode], SimpleDocumentStore]:
    """
    Parse documents into hierarchical nodes for auto-merging retrieval.
    
    Args:
        documents: List of LlamaIndex Document objects
        chunk_sizes: List of chunk sizes [large, small] (default: [1024, 256])
        chunk_overlap: Overlap between chunks (default: 20)
    
    Returns:
        Tuple of (all_nodes, leaf_nodes, docstore)
        - all_nodes: All nodes at all levels
        - leaf_nodes: Only the smallest chunks (for indexing)
        - docstore: Document store with all nodes (for auto-merging)
    
    Why this approach?
    - Leaf nodes (small) are indexed for precise retrieval
    - Parent nodes (large) stored in docstore
    - Auto-merging retriever can "merge up" when needed
    """
    chunk_sizes = chunk_sizes or CHUNK_SIZES
    chunk_overlap = chunk_overlap or CHUNK_OVERLAP
    
    logger.info(f"Creating hierarchical chunks: sizes={chunk_sizes}, overlap={chunk_overlap}")
    
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

