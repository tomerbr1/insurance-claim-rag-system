"""
Needle-in-Haystack Agent - Precise fact retrieval using hierarchical index.

This agent handles:
- "What was the exact towing cost?"
- "What is the wire transfer reference number?"
- "How long was the spill on the floor?"
- Specific numbers, IDs, names buried in documents

Uses auto-merging retriever for optimal context.
Supports metadata filtering when a specific claim ID is mentioned.
"""

import logging
import re
from typing import Optional

from llama_index.core import VectorStoreIndex
from llama_index.core.retrievers import AutoMergingRetriever
from llama_index.core.query_engine import RetrieverQueryEngine
from llama_index.core.prompts import PromptTemplate
from llama_index.core.storage.docstore import SimpleDocumentStore
from llama_index.core.vector_stores import MetadataFilters, MetadataFilter, FilterOperator
from llama_index.llms.openai import OpenAI

from src.config import (
    OPENAI_API_KEY, OPENAI_MODEL,
    SIMILARITY_TOP_K, AUTO_MERGE_THRESHOLD
)

# Regex pattern to extract claim IDs (e.g., CLM-2024-001847)
CLAIM_ID_PATTERN = re.compile(r'CLM[-_]?\d{4}[-_]?\d{6}', re.IGNORECASE)


def extract_claim_id(query: str) -> Optional[str]:
    """Extract and normalize a claim ID from the query if present."""
    match = CLAIM_ID_PATTERN.search(query)
    if match:
        raw_id = match.group(0).upper()
        normalized = raw_id.replace('_', '-')
        if not normalized.startswith('CLM-'):
            normalized = 'CLM-' + normalized[3:]
        return normalized
    return None

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


class SmartNeedleQueryEngine:
    """
    A smart wrapper around the needle index that applies metadata filtering
    when a claim ID is detected in the query.

    Uses auto-merging retriever for optimal context and supports
    filtering by claim ID when specified in the query.
    """

    def __init__(
        self,
        hierarchical_index: VectorStoreIndex,
        docstore: SimpleDocumentStore,
        llm: Optional[OpenAI] = None,
        similarity_top_k: int = None,
        merge_threshold: float = None
    ):
        self._hierarchical_index = hierarchical_index
        self._docstore = docstore
        self._llm = llm or OpenAI(model=OPENAI_MODEL, api_key=OPENAI_API_KEY, temperature=0)
        self._similarity_top_k = similarity_top_k or SIMILARITY_TOP_K
        self._merge_threshold = merge_threshold or AUTO_MERGE_THRESHOLD

    def _create_query_engine(self, filters: Optional[MetadataFilters] = None):
        """Create a query engine with optional metadata filters."""
        # Create base retriever with optional filters
        base_retriever = self._hierarchical_index.as_retriever(
            similarity_top_k=self._similarity_top_k,
            filters=filters
        )

        # Try to use auto-merging if docstore has proper relationships
        try:
            # Use the explicitly passed docstore, not the index's storage_context.docstore
            # (which may be empty when loading from persistence)
            if not self._docstore.docs:
                raise ValueError("Docstore is empty")

            # Inject our loaded docstore into the index's storage_context
            # This preserves all other components (vector_store, index_store, etc.)
            # while providing the docstore needed for auto-merging
            storage_context = self._hierarchical_index.storage_context
            storage_context.docstore = self._docstore

            retriever = AutoMergingRetriever(
                base_retriever,
                storage_context=storage_context,
                simple_ratio_thresh=self._merge_threshold
            )
        except Exception as e:
            # Fall back to basic retriever if auto-merging fails
            logger.warning(f"Auto-merging unavailable ({e}), using basic retriever")
            retriever = base_retriever

        # Create query engine with custom prompt
        return RetrieverQueryEngine.from_args(
            retriever=retriever,
            llm=self._llm,
            text_qa_template=NEEDLE_AGENT_PROMPT
        )

    def _create_basic_query_engine(self, filters: Optional[MetadataFilters] = None):
        """Create a basic query engine without auto-merging (fallback)."""
        base_retriever = self._hierarchical_index.as_retriever(
            similarity_top_k=self._similarity_top_k,
            filters=filters
        )
        return RetrieverQueryEngine.from_args(
            retriever=base_retriever,
            llm=self._llm,
            text_qa_template=NEEDLE_AGENT_PROMPT
        )

    def query(self, query_str: str):
        """
        Query the needle index with smart claim ID detection and filtering.

        If a claim ID is found in the query, filters results to that claim
        before doing semantic search.
        """
        claim_id = extract_claim_id(query_str)
        filters = None

        if claim_id:
            logger.info(f"📋 Detected claim ID in query: {claim_id}")
            filters = MetadataFilters(
                filters=[
                    MetadataFilter(
                        key="claim_id",
                        value=claim_id,
                        operator=FilterOperator.EQ
                    )
                ]
            )

        query_engine = self._create_query_engine(filters=filters)

        try:
            return query_engine.query(query_str)
        except Exception as e:
            # Auto-merging failed (likely docstore relationship issue)
            # Fall back to basic retriever without auto-merging
            if "doc_id" in str(e) and "not found" in str(e):
                logger.warning(f"Auto-merging failed: {e}. Falling back to basic retriever.")
                logger.warning("Consider rebuilding indexes with: python main.py --build")
                basic_engine = self._create_basic_query_engine(filters=filters)
                return basic_engine.query(query_str)
            else:
                raise


def create_needle_agent(
    hierarchical_index: VectorStoreIndex,
    docstore: SimpleDocumentStore,
    llm: Optional[OpenAI] = None,
    similarity_top_k: int = None,
    merge_threshold: float = None
) -> SmartNeedleQueryEngine:
    """
    Create a needle-in-haystack agent for precise fact retrieval.

    Uses auto-merging retriever to:
    1. Start with small chunks (precise)
    2. Automatically merge to parent when more context needed

    Now includes smart claim ID detection - when a query mentions a specific
    claim ID, the agent filters to that claim first.

    Args:
        hierarchical_index: VectorStoreIndex built from leaf nodes
        docstore: Document store with all hierarchical nodes
        llm: LLM for response generation (default: GPT-4)
        similarity_top_k: Number of chunks to retrieve (default: 6)
        merge_threshold: Auto-merge threshold (default: 0.5)

    Returns:
        SmartNeedleQueryEngine configured for precise retrieval
    """
    llm = llm or OpenAI(model=OPENAI_MODEL, api_key=OPENAI_API_KEY, temperature=0)
    similarity_top_k = similarity_top_k or SIMILARITY_TOP_K
    merge_threshold = merge_threshold or AUTO_MERGE_THRESHOLD

    logger.info(f"Creating smart needle agent: top_k={similarity_top_k}, threshold={merge_threshold}")

    agent = SmartNeedleQueryEngine(
        hierarchical_index=hierarchical_index,
        docstore=docstore,
        llm=llm,
        similarity_top_k=similarity_top_k,
        merge_threshold=merge_threshold
    )

    logger.info("✅ Needle agent created")
    return agent


def create_high_precision_needle_agent(
    hierarchical_index: VectorStoreIndex,
    docstore: SimpleDocumentStore,
    llm: Optional[OpenAI] = None
) -> SmartNeedleQueryEngine:
    """
    Create a high-precision needle agent.

    More conservative settings for very specific queries.

    Args:
        hierarchical_index: VectorStoreIndex built from leaf nodes
        docstore: Document store with all hierarchical nodes
        llm: LLM for response generation

    Returns:
        High-precision SmartNeedleQueryEngine
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

