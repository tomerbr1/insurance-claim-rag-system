"""
Router Agent - 3-way intelligent query routing with MCP tool integration.

This is the Manager agent that:
1. Receives user queries
2. Uses MCP tools to gather system context (collection stats)
3. Classifies query type
4. Routes to appropriate specialist agent

Routing options:
- STRUCTURED: Exact lookups, filters, aggregations → SQL
- SUMMARY: High-level overviews, timelines → RAG Summary
- NEEDLE: Precise fact extraction → RAG Needle

MCP Integration:
- Router calls MCP tools (collection_stats) before routing
- Tool calls are logged visibly for demonstration
- Demonstrates agent-tool interaction pattern

Hybrid architecture benefits:
- Structured queries are 100x faster (SQL vs vector search)
- No hallucination on structured data
- Semantic understanding for narratives
"""

import logging
from typing import Optional, Tuple, List, Any
from enum import Enum

from llama_index.core.query_engine import RouterQueryEngine, CustomQueryEngine
from llama_index.core.selectors import LLMSingleSelector
from llama_index.core.tools import QueryEngineTool, FunctionTool
from llama_index.llms.openai import OpenAI
from llama_index.core.bridge.pydantic import PrivateAttr

from src.config import OPENAI_API_KEY, OPENAI_MODEL, OPENAI_MINI_MODEL, MCP_LOG_TOOL_CALLS

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class QueryType(Enum):
    """Types of queries the system can handle."""
    STRUCTURED = "structured"
    SUMMARY = "summary"
    NEEDLE = "needle"


# Detailed routing prompt
ROUTER_PROMPT = """
You are a query router for a hybrid insurance claims system.
Your job is to classify queries to route them to the correct specialist.

Available specialists:

1. STRUCTURED AGENT (SQL Database)
   Best for:
   - Exact claim ID lookups: "Get claim CLM-2024-001847"
   - Filtering by metadata: "All auto claims", "Claims over $50k"
   - Date range queries: "Claims filed in October 2024"
   - Aggregations: "Average claim value", "Count of settled claims"
   - Status checks: "Which claims are still open?"
   Keywords: get, list, filter, count, average, sum, show all, which

2. SUMMARY AGENT (RAG on Summaries)
   Best for:
   - High-level overviews: "What happened in claim X?"
   - Timeline narratives: "Summarize the events"
   - General explanations: "Why was the claim denied?"
   - Questions spanning multiple sections
   Keywords: what happened, summarize, overview, explain, timeline, describe

3. NEEDLE AGENT (RAG on Full Documents)
   Best for:
   - Specific facts buried in text: "What was the exact towing cost?"
   - Reference IDs from documents: "Wire transfer number?"
   - Precise details: "How long was the spill on the floor?"
   - Quote extraction: "What did the witness say?"
   Keywords: exact, specific, how much, what number, reference, invoice

Analyze the query and decide which specialist should handle it.

Query: {query}

Think step by step:
1. Does this need exact metadata (dates, IDs, amounts as filters)? → STRUCTURED
2. Does this need narrative understanding or broad context? → SUMMARY  
3. Does this need precise fact extraction from document text? → NEEDLE

Return ONLY one word: STRUCTURED, SUMMARY, or NEEDLE
"""


class RouterAgent(CustomQueryEngine):
    """
    Intelligent 3-way router for hybrid architecture with MCP tool support.

    Routes queries to:
    - Structured agent (SQL) for metadata queries
    - Summary agent (RAG) for high-level questions
    - Needle agent (RAG) for precise fact retrieval

    MCP Integration:
    - Calls MCP tools (collection_stats) before routing decisions
    - Visible tool call output for demonstration
    - Uses system context to inform routing
    """

    _structured_engine: CustomQueryEngine = PrivateAttr()
    _summary_engine: CustomQueryEngine = PrivateAttr()
    _needle_engine: CustomQueryEngine = PrivateAttr()
    _llm: OpenAI = PrivateAttr()
    _verbose: bool = PrivateAttr()
    _mcp_tools: List[FunctionTool] = PrivateAttr()
    _log_tool_calls: bool = PrivateAttr()

    def __init__(
        self,
        structured_engine,
        summary_engine,
        needle_engine,
        llm: Optional[OpenAI] = None,
        verbose: bool = True,
        use_mini_for_routing: bool = True,
        mcp_tools: Optional[List[FunctionTool]] = None,
        log_tool_calls: bool = True,
        **kwargs
    ):
        """
        Initialize the router agent.

        Args:
            structured_engine: Query engine for SQL-based queries
            summary_engine: Query engine for high-level RAG queries
            needle_engine: Query engine for precise RAG queries
            llm: LLM for routing decisions (default: GPT-4o-mini for cost efficiency)
            verbose: Whether to log routing decisions
            use_mini_for_routing: Use cheaper model for routing (default True)
            mcp_tools: List of MCP FunctionTools for system introspection
            log_tool_calls: Whether to print MCP tool calls (for demo visibility)
        """
        super().__init__(**kwargs)
        self._structured_engine = structured_engine
        self._summary_engine = summary_engine
        self._needle_engine = needle_engine
        self._mcp_tools = mcp_tools or []
        self._log_tool_calls = log_tool_calls and MCP_LOG_TOOL_CALLS

        # Use cheaper model for routing by default - classification doesn't need GPT-4
        if llm:
            self._llm = llm
        else:
            routing_model = OPENAI_MINI_MODEL if use_mini_for_routing else OPENAI_MODEL
            self._llm = OpenAI(model=routing_model, api_key=OPENAI_API_KEY, temperature=0)
            if verbose:
                logger.info(f"Router using model: {routing_model}")

        self._verbose = verbose

        if self._mcp_tools and verbose:
            tool_names = [t.metadata.name for t in self._mcp_tools]
            logger.info(f"Router initialized with MCP tools: {tool_names}")

    def _find_mcp_tool(self, tool_name: str) -> Optional[FunctionTool]:
        """Find an MCP tool by name."""
        for tool in self._mcp_tools:
            if tool.metadata.name == tool_name:
                return tool
        return None

    def _call_mcp_tool(self, tool_name: str, *args, **kwargs) -> Optional[str]:
        """
        Call an MCP tool by name with visible logging.

        This method demonstrates agent-tool interaction by:
        1. Finding the tool by name
        2. Printing the tool call (visible to user)
        3. Executing the tool
        4. Printing and returning the result

        Args:
            tool_name: Name of the MCP tool to call
            *args, **kwargs: Arguments to pass to the tool

        Returns:
            Tool result as string, or None if tool not found
        """
        tool = self._find_mcp_tool(tool_name)
        if not tool:
            if self._verbose:
                logger.warning(f"MCP tool '{tool_name}' not found")
            return None

        # Format the call for display
        args_str = ", ".join([repr(a) for a in args])
        kwargs_str = ", ".join([f"{k}={repr(v)}" for k, v in kwargs.items()])
        all_args = ", ".join(filter(None, [args_str, kwargs_str]))

        if self._log_tool_calls:
            print(f"🔧 [MCP TOOL CALL] {tool_name}({all_args})")

        try:
            # Execute the tool
            result = tool.fn(*args, **kwargs)

            if self._log_tool_calls:
                # Truncate long results for display
                display_result = result if len(str(result)) < 100 else str(result)[:97] + "..."
                print(f"   └─ Result: {display_result}")

            return result

        except Exception as e:
            error_msg = f"Error: {str(e)}"
            if self._log_tool_calls:
                print(f"   └─ {error_msg}")
            return error_msg

    def _get_system_context(self) -> str:
        """
        Gather system context using MCP tools before routing.

        Calls collection_stats to understand the available data,
        which can inform routing decisions.

        Returns:
            System context string describing available collections
        """
        if not self._mcp_tools:
            return ""

        context_parts = []

        # Call collection_stats for the main collection
        stats_result = self._call_mcp_tool("collection_stats")
        if stats_result:
            context_parts.append(stats_result)

        # Optionally get summary collection stats too
        # This shows we can call multiple MCP tools
        summary_stats = self._call_mcp_tool("collection_stats", "insurance_claims_summaries")
        if summary_stats and "Error" not in summary_stats:
            context_parts.append(summary_stats)

        return " | ".join(context_parts) if context_parts else ""

    def _classify_query(self, query: str) -> QueryType:
        """
        Classify a query to determine which agent should handle it.
        
        Args:
            query: User's natural language query
        
        Returns:
            QueryType enum indicating which agent to use
        """
        prompt = ROUTER_PROMPT.format(query=query)
        
        try:
            response = self._llm.complete(prompt)
            classification = response.text.strip().upper()
            
            if "STRUCTURED" in classification:
                return QueryType.STRUCTURED
            elif "SUMMARY" in classification:
                return QueryType.SUMMARY
            elif "NEEDLE" in classification:
                return QueryType.NEEDLE
            else:
                # Default to needle for unknown
                logger.warning(f"Unknown classification: {classification}, defaulting to NEEDLE")
                return QueryType.NEEDLE
                
        except Exception as e:
            logger.error(f"Error classifying query: {e}")
            return QueryType.NEEDLE  # Default fallback
    
    def custom_query(self, query_str: str) -> str:
        """
        Route and execute a query with MCP tool integration.

        Steps:
        1. Call MCP tools to gather system context (visible tool calls)
        2. Classify the query type
        3. Route to appropriate specialist
        4. Return the response

        Args:
            query_str: User's natural language query

        Returns:
            Response from the appropriate specialist agent
        """
        # Step 0: Gather system context using MCP tools (visible to user)
        if self._mcp_tools:
            system_context = self._get_system_context()
            # Context can be used to inform routing (future enhancement)
            # For now, we're demonstrating that the router USES MCP tools

        # Step 1: Classify
        query_type = self._classify_query(query_str)

        if self._verbose:
            logger.info(f"🔀 Routing query to: {query_type.value.upper()}")
            print(f"🔀 Routed to: {query_type.value.upper()} agent")
        
        # Step 2: Route and execute
        try:
            if query_type == QueryType.STRUCTURED:
                response = self._structured_engine.custom_query(query_str)
            elif query_type == QueryType.SUMMARY:
                response = self._summary_engine.query(query_str)
                response = str(response)
            else:  # NEEDLE
                response = self._needle_engine.query(query_str)
                response = str(response)
            
            return response
            
        except Exception as e:
            logger.error(f"Error executing query: {e}")
            return f"I encountered an error processing your query: {str(e)}"
    
    def query_with_metadata(self, query_str: str) -> Tuple[str, dict]:
        """
        Execute query and return response with routing metadata.
        
        Args:
            query_str: User's query
        
        Returns:
            Tuple of (response, metadata dict)
        """
        query_type = self._classify_query(query_str)
        response = self.custom_query(query_str)
        
        metadata = {
            'query': query_str,
            'routed_to': query_type.value,
            'agent_type': query_type.name
        }
        
        return response, metadata


def create_router_agent(
    structured_engine,
    summary_engine,
    needle_engine,
    llm: Optional[OpenAI] = None,
    verbose: bool = True,
    mcp_tools: Optional[List[FunctionTool]] = None
) -> RouterAgent:
    """
    Factory function to create a router agent with MCP tool support.

    Args:
        structured_engine: Engine for SQL queries
        summary_engine: Engine for summary RAG
        needle_engine: Engine for needle RAG
        llm: Optional LLM override
        verbose: Whether to log routing decisions
        mcp_tools: Optional list of MCP tools for system introspection

    Returns:
        RouterAgent instance configured with MCP tools
    """
    return RouterAgent(
        structured_engine=structured_engine,
        summary_engine=summary_engine,
        needle_engine=needle_engine,
        llm=llm,
        verbose=verbose,
        mcp_tools=mcp_tools
    )


def create_llamaindex_router(
    structured_engine,
    summary_engine,
    needle_engine
) -> RouterQueryEngine:
    """
    Alternative: Create router using LlamaIndex's built-in RouterQueryEngine.
    
    This uses LlamaIndex's selector for routing decisions.
    
    Args:
        structured_engine: Engine for SQL queries
        summary_engine: Engine for summary RAG
        needle_engine: Engine for needle RAG
    
    Returns:
        RouterQueryEngine instance
    """
    structured_tool = QueryEngineTool.from_defaults(
        query_engine=structured_engine,
        description=(
            "Use for exact lookups by claim ID, filtering by metadata "
            "(dates, amounts, status), and aggregations (count, average). "
            "Best for: 'Get claim X', 'All claims over $50k', 'Count of settled claims'"
        )
    )
    
    summary_tool = QueryEngineTool.from_defaults(
        query_engine=summary_engine,
        description=(
            "Use for high-level summaries, overviews, timelines, and narrative questions. "
            "Best for: 'What happened in claim X?', 'Summarize the events', 'Give an overview'"
        )
    )
    
    needle_tool = QueryEngineTool.from_defaults(
        query_engine=needle_engine,
        description=(
            "Use for precise facts, specific numbers, reference IDs, and details "
            "buried in document text. Best for: 'What was the exact towing cost?', "
            "'What is the wire transfer number?', 'How long was the spill?'"
        )
    )
    
    router = RouterQueryEngine(
        selector=LLMSingleSelector.from_defaults(),
        query_engine_tools=[structured_tool, summary_tool, needle_tool]
    )
    
    return router


# Quick test
if __name__ == "__main__":
    print("Router agent module loaded successfully")
    print("Use create_router_agent(structured, summary, needle) to create")

