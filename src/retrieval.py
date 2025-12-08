"""
Retrieval Module - Auto-merging retriever setup.

This module implements:
1. Auto-merging retriever for hierarchical chunks
2. Query engine configuration for both indexes

Auto-Merging Retrieval:
1. Query searches among leaf nodes (small chunks)
2. If >50% of a parent's children are retrieved
3. Automatically merge to parent chunk for more context
4. Continues up the hierarchy as needed
"""

import logging
from typing import Optional

from llama_index.core import VectorStoreIndex
from llama_index.core.retrievers import AutoMergingRetriever
from llama_index.core.query_engine import RetrieverQueryEngine
from llama_index.core.storage.docstore import SimpleDocumentStore
from llama_index.core.postprocessor import SimilarityPostprocessor
from llama_index.llms.openai import OpenAI

from src.config import (
    SIMILARITY_TOP_K, AUTO_MERGE_THRESHOLD,
    OPENAI_API_KEY, OPENAI_MODEL
)

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def create_auto_merging_retriever(
    index: VectorStoreIndex,
    docstore: SimpleDocumentStore,
    similarity_top_k: int = None,
    merge_threshold: float = None
):
    """
    Create an auto-merging retriever for hierarchical chunks.
    
    How it works:
    1. Starts by retrieving small (leaf) chunks based on similarity
    2. If many siblings of a parent are retrieved (>threshold)
    3. Automatically fetches the parent chunk instead
    4. Provides more context without manual tuning
    
    Args:
        index: VectorStoreIndex built from leaf nodes
        docstore: SimpleDocumentStore containing all hierarchical nodes
        similarity_top_k: Number of chunks to retrieve (default: 6)
        merge_threshold: Threshold for merging (default: 0.5)
    
    Returns:
        AutoMergingRetriever instance
    """
    similarity_top_k = similarity_top_k or SIMILARITY_TOP_K
    merge_threshold = merge_threshold or AUTO_MERGE_THRESHOLD
    
    logger.info(f"Creating auto-merging retriever: top_k={similarity_top_k}, threshold={merge_threshold}")
    
    # Create base retriever from index
    base_retriever = index.as_retriever(similarity_top_k=similarity_top_k)
    
    # Wrap with auto-merging capability
    retriever = AutoMergingRetriever(
        base_retriever,
        storage_context=index.storage_context,
        simple_ratio_thresh=merge_threshold  # Merge if >threshold of children match
    )
    
    logger.info("✅ Auto-merging retriever created")
    return retriever


def create_needle_query_engine(
    index: VectorStoreIndex,
    docstore: SimpleDocumentStore,
    llm: Optional[OpenAI] = None
) -> RetrieverQueryEngine:
    """
    Create query engine for precise fact retrieval (needle-in-haystack).
    
    Uses auto-merging retriever for optimal context window.
    
    Args:
        index: Hierarchical vector index
        docstore: Document store with all nodes
        llm: LLM for query synthesis (default: GPT-4)
    
    Returns:
        RetrieverQueryEngine configured for precision retrieval
    """
    llm = llm or OpenAI(model=OPENAI_MODEL, api_key=OPENAI_API_KEY, temperature=0)
    
    # Create auto-merging retriever
    retriever = create_auto_merging_retriever(index, docstore)
    
    # Create query engine with custom prompt
    query_engine = RetrieverQueryEngine.from_args(
        retriever=retriever,
        llm=llm,
        node_postprocessors=[
            SimilarityPostprocessor(similarity_cutoff=0.7)  # Filter low-confidence
        ]
    )
    
    logger.info("✅ Needle query engine created")
    return query_engine


def create_summary_query_engine(
    summary_index: VectorStoreIndex,
    llm: Optional[OpenAI] = None
) -> RetrieverQueryEngine:
    """
    Create query engine for high-level summary queries.
    
    Uses the multi-level summary index.
    
    Args:
        summary_index: Index containing chunk/section/document summaries
        llm: LLM for query synthesis
    
    Returns:
        RetrieverQueryEngine configured for summary retrieval
    """
    llm = llm or OpenAI(model=OPENAI_MODEL, api_key=OPENAI_API_KEY, temperature=0)
    
    # Create standard retriever (no auto-merging needed for summaries)
    query_engine = summary_index.as_query_engine(
        llm=llm,
        similarity_top_k=5,
        response_mode="tree_summarize"  # Good for combining multiple summaries
    )
    
    logger.info("✅ Summary query engine created")
    return query_engine


# Quick test
if __name__ == "__main__":
    print("Retrieval module loaded successfully")
    print(f"Default settings:")
    print(f"  SIMILARITY_TOP_K: {SIMILARITY_TOP_K}")
    print(f"  AUTO_MERGE_THRESHOLD: {AUTO_MERGE_THRESHOLD}")

