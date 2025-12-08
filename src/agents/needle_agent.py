"""
Needle-in-Haystack Agent - Precise fact retrieval using hierarchical index.

This agent handles:
- "What was the exact towing cost?"
- "What is the wire transfer reference number?"
- "How long was the spill on the floor?"
- Specific numbers, IDs, names buried in documents

Uses auto-merging retriever for optimal context.
"""

import logging
from typing import Optional

from llama_index.core import VectorStoreIndex
from llama_index.core.retrievers import AutoMergingRetriever
from llama_index.core.query_engine import RetrieverQueryEngine
from llama_index.core.prompts import PromptTemplate
from llama_index.core.storage.docstore import SimpleDocumentStore
from llama_index.core.postprocessor import SimilarityPostprocessor
from llama_index.llms.openai import OpenAI

from src.config import (
    OPENAI_API_KEY, OPENAI_MODEL, 
    SIMILARITY_TOP_K, AUTO_MERGE_THRESHOLD
)

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


# Custom prompt for needle agent
NEEDLE_AGENT_PROMPT = PromptTemplate(
    """You are a precision retrieval specialist for insurance claims.
Your role is to find and extract SPECIFIC facts from claim documents.

CRITICAL INSTRUCTIONS:
- Be PRECISE - cite exact values, dates, reference numbers
- If the information is NOT in the context, say "I could not find this information in the available documents"
- Include the claim ID when referencing specific documents
- Double-check numbers and IDs for accuracy
- Do NOT make up or guess information

Context from claims:
{context_str}

Question: {query_str}

Precise answer (cite the exact value if found):
"""
)


def create_needle_agent(
    hierarchical_index: VectorStoreIndex,
    docstore: SimpleDocumentStore,
    llm: Optional[OpenAI] = None,
    similarity_top_k: int = None,
    merge_threshold: float = None
) -> RetrieverQueryEngine:
    """
    Create a needle-in-haystack agent for precise fact retrieval.
    
    Uses auto-merging retriever to:
    1. Start with small chunks (precise)
    2. Automatically merge to parent when more context needed
    
    Args:
        hierarchical_index: VectorStoreIndex built from leaf nodes
        docstore: Document store with all hierarchical nodes
        llm: LLM for response generation (default: GPT-4)
        similarity_top_k: Number of chunks to retrieve (default: 6)
        merge_threshold: Auto-merge threshold (default: 0.5)
    
    Returns:
        Query engine configured for precise retrieval
    """
    llm = llm or OpenAI(model=OPENAI_MODEL, api_key=OPENAI_API_KEY, temperature=0)
    similarity_top_k = similarity_top_k or SIMILARITY_TOP_K
    merge_threshold = merge_threshold or AUTO_MERGE_THRESHOLD
    
    logger.info(f"Creating needle agent: top_k={similarity_top_k}, threshold={merge_threshold}")
    
    # Create base retriever
    base_retriever = hierarchical_index.as_retriever(
        similarity_top_k=similarity_top_k
    )
    
    # Wrap with auto-merging capability
    retriever = AutoMergingRetriever(
        base_retriever,
        storage_context=hierarchical_index.storage_context,
        simple_ratio_thresh=merge_threshold
    )
    
    # Create query engine with custom prompt
    query_engine = RetrieverQueryEngine.from_args(
        retriever=retriever,
        llm=llm,
        text_qa_template=NEEDLE_AGENT_PROMPT,
        node_postprocessors=[
            SimilarityPostprocessor(similarity_cutoff=0.5)  # Filter low-confidence
        ]
    )
    
    logger.info("✅ Needle agent created")
    return query_engine


def create_high_precision_needle_agent(
    hierarchical_index: VectorStoreIndex,
    docstore: SimpleDocumentStore,
    llm: Optional[OpenAI] = None
) -> RetrieverQueryEngine:
    """
    Create a high-precision needle agent.
    
    More conservative settings for very specific queries.
    
    Args:
        hierarchical_index: VectorStoreIndex built from leaf nodes
        docstore: Document store with all hierarchical nodes
        llm: LLM for response generation
    
    Returns:
        High-precision query engine
    """
    return create_needle_agent(
        hierarchical_index=hierarchical_index,
        docstore=docstore,
        llm=llm,
        similarity_top_k=10,  # More candidates
        merge_threshold=0.3   # Less aggressive merging
    )


# Quick test
if __name__ == "__main__":
    print("Needle agent module loaded successfully")
    print("Use create_needle_agent(hierarchical_index, docstore) to create an agent")

