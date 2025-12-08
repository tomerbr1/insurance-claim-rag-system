"""
Summary Agent - High-level RAG queries using Summary Index.

This agent handles:
- "What happened in claim X?"
- "Give me an overview of..."
- "Summarize the timeline..."
- General questions needing broad context

Uses the multi-level summary index (chunk/section/document summaries).
"""

import logging
from typing import Optional

from llama_index.core import VectorStoreIndex
from llama_index.core.query_engine import RetrieverQueryEngine
from llama_index.core.prompts import PromptTemplate
from llama_index.llms.openai import OpenAI

from src.config import OPENAI_API_KEY, OPENAI_MODEL

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


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


def create_summary_agent(
    summary_index: VectorStoreIndex,
    llm: Optional[OpenAI] = None
) -> RetrieverQueryEngine:
    """
    Create a summary agent using the multi-level summary index.
    
    This agent is designed for:
    - High-level overview questions
    - Timeline summaries
    - General claim information
    
    Args:
        summary_index: VectorStoreIndex containing multi-level summaries
        llm: LLM for response generation (default: GPT-4)
    
    Returns:
        Query engine configured for summary queries
    """
    llm = llm or OpenAI(model=OPENAI_MODEL, api_key=OPENAI_API_KEY, temperature=0)
    
    logger.info("Creating summary agent...")
    
    # Create query engine with custom prompt
    query_engine = summary_index.as_query_engine(
        llm=llm,
        similarity_top_k=5,  # Retrieve multiple summaries
        response_mode="tree_summarize",  # Combine multiple summaries
        text_qa_template=SUMMARY_AGENT_PROMPT
    )
    
    logger.info("✅ Summary agent created")
    return query_engine


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

