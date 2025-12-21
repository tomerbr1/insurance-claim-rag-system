"""
Summary Agent - High-level RAG queries using Summary Index.

This agent handles:
- "What happened in claim X?"
- "Give me an overview of..."
- "Summarize the timeline..."
- General questions needing broad context

Uses the multi-level summary index (chunk/section/document summaries).
Supports metadata filtering when a specific claim ID is mentioned.
"""

import logging
import re
from typing import Optional

from llama_index.core import VectorStoreIndex
from llama_index.core.query_engine import RetrieverQueryEngine
from llama_index.core.prompts import PromptTemplate
from llama_index.core.vector_stores import MetadataFilters, MetadataFilter, FilterOperator
from llama_index.llms.openai import OpenAI

from src.config import OPENAI_API_KEY, OPENAI_MODEL

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Regex pattern to extract claim IDs (e.g., CLM-2024-001847)
CLAIM_ID_PATTERN = re.compile(r'CLM[-_]?\d{4}[-_]?\d{6}', re.IGNORECASE)


def extract_claim_id(query: str) -> Optional[str]:
    """
    Extract a claim ID from the query if present.

    Normalizes claim IDs to the format CLM-YYYY-NNNNNN.

    Args:
        query: User's query string

    Returns:
        Normalized claim ID or None if not found
    """
    match = CLAIM_ID_PATTERN.search(query)
    if match:
        # Normalize to CLM-YYYY-NNNNNN format
        raw_id = match.group(0).upper()
        # Remove underscores and ensure hyphens
        normalized = raw_id.replace('_', '-')
        # Ensure proper format: CLM-YYYY-NNNNNN
        if not normalized.startswith('CLM-'):
            normalized = 'CLM-' + normalized[3:]
        return normalized
    return None


# Custom prompt for summary agent
SUMMARY_AGENT_PROMPT = PromptTemplate(
    """You are an insurance claims summarization expert.
Your role is to provide clear, comprehensive overviews of insurance claims.

When answering:
- Provide a chronological overview when relevant
- Highlight key events, parties involved, and outcomes
- Mention claim status and total values
- Keep responses organized and easy to understand
- If multiple claims are involved, address each one

Context from claims:
{context_str}

Question: {query_str}

Provide a helpful summary:
"""
)

# Alternative prompt for timeline questions
TIMELINE_PROMPT = PromptTemplate(
    """You are an insurance claims timeline expert.
Your role is to reconstruct and explain the sequence of events in insurance claims.

When answering:
- Present events in chronological order
- Include specific dates when available
- Explain the significance of key events
- Note any delays or unusual patterns

Context from claims:
{context_str}

Question: {query_str}

Timeline response:
"""
)


class SmartSummaryQueryEngine:
    """
    A smart wrapper around the summary index that applies metadata filtering
    when a claim ID is detected in the query.

    This solves the problem where semantic search alone fails to find
    the right claim because claim IDs aren't semantically meaningful.
    """

    def __init__(
        self,
        summary_index: VectorStoreIndex,
        llm: Optional[OpenAI] = None,
        similarity_top_k: int = 5
    ):
        self._summary_index = summary_index
        self._llm = llm or OpenAI(model=OPENAI_MODEL, api_key=OPENAI_API_KEY, temperature=0)
        self._similarity_top_k = similarity_top_k

    def query(self, query_str: str):
        """
        Query the summary index with smart claim ID detection and filtering.

        If a claim ID is found in the query, filters results to that claim
        before doing semantic search. This ensures queries like
        "What happened in claim CLM-2024-003012?" return relevant results.
        """
        claim_id = extract_claim_id(query_str)

        if claim_id:
            logger.info(f"📋 Detected claim ID in query: {claim_id}")
            # Apply metadata filter for this claim
            filters = MetadataFilters(
                filters=[
                    MetadataFilter(
                        key="claim_id",
                        value=claim_id,
                        operator=FilterOperator.EQ
                    )
                ]
            )
            query_engine = self._summary_index.as_query_engine(
                llm=self._llm,
                similarity_top_k=self._similarity_top_k,
                response_mode="tree_summarize",
                text_qa_template=SUMMARY_AGENT_PROMPT,
                filters=filters
            )
        else:
            # No claim ID - use standard semantic search
            query_engine = self._summary_index.as_query_engine(
                llm=self._llm,
                similarity_top_k=self._similarity_top_k,
                response_mode="tree_summarize",
                text_qa_template=SUMMARY_AGENT_PROMPT
            )

        return query_engine.query(query_str)


def create_summary_agent(
    summary_index: VectorStoreIndex,
    llm: Optional[OpenAI] = None
) -> SmartSummaryQueryEngine:
    """
    Create a summary agent using the multi-level summary index.

    This agent is designed for:
    - High-level overview questions
    - Timeline summaries
    - General claim information

    Now includes smart claim ID detection - when a query mentions a specific
    claim ID (e.g., CLM-2024-003012), the agent filters to that claim first.

    Args:
        summary_index: VectorStoreIndex containing multi-level summaries
        llm: LLM for response generation (default: GPT-4)

    Returns:
        SmartSummaryQueryEngine configured for summary queries
    """
    llm = llm or OpenAI(model=OPENAI_MODEL, api_key=OPENAI_API_KEY, temperature=0)

    logger.info("Creating smart summary agent with claim ID detection...")

    agent = SmartSummaryQueryEngine(
        summary_index=summary_index,
        llm=llm,
        similarity_top_k=5
    )

    logger.info("✅ Summary agent created")
    return agent


def create_timeline_agent(
    summary_index: VectorStoreIndex,
    llm: Optional[OpenAI] = None
) -> RetrieverQueryEngine:
    """
    Create a specialized timeline agent.
    
    Optimized for questions about sequence of events.
    
    Args:
        summary_index: VectorStoreIndex containing multi-level summaries
        llm: LLM for response generation
    
    Returns:
        Query engine configured for timeline queries
    """
    llm = llm or OpenAI(model=OPENAI_MODEL, api_key=OPENAI_API_KEY, temperature=0)
    
    query_engine = summary_index.as_query_engine(
        llm=llm,
        similarity_top_k=7,  # More context for timelines
        response_mode="tree_summarize",
        text_qa_template=TIMELINE_PROMPT
    )
    
    return query_engine


# Quick test
if __name__ == "__main__":
    print("Summary agent module loaded successfully")
    print("Use create_summary_agent(summary_index) to create an agent")

