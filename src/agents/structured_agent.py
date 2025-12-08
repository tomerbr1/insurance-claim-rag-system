"""
Structured Query Agent - SQL-based queries on claim metadata.

This agent handles:
- Exact lookups by claim ID
- Filtering by type, status, amount ranges
- Date range queries
- Aggregations (count, sum, average)

Part of hybrid architecture:
- Structured queries → SQL (fast, deterministic)
- Narrative queries → RAG (semantic understanding)
"""

import logging
from typing import Optional

from llama_index.llms.openai import OpenAI
from llama_index.core.query_engine import CustomQueryEngine
from llama_index.core.bridge.pydantic import PrivateAttr

from src.metadata_store import MetadataStore
from src.config import OPENAI_API_KEY, OPENAI_MODEL

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


# Prompt for natural language to SQL conversion
NL_TO_SQL_PROMPT = """
You are a SQL query specialist for an insurance claims database.

Database schema:
Table: claims
  - claim_id (TEXT, PRIMARY KEY) - e.g., "CLM-2024-001847"
  - claim_type (TEXT) - e.g., "Auto Accident", "Slip and Fall", "Water Damage"
  - claimant (TEXT) - claimant's full name
  - policy_number (TEXT)
  - claim_status (TEXT) - "OPEN", "CLOSED", or "SETTLED"
  - total_value (REAL) - dollar amount
  - incident_date (DATE) - YYYY-MM-DD format
  - filing_date (DATE) - YYYY-MM-DD format
  - settlement_date (DATE) - YYYY-MM-DD format or NULL

Convert the user's natural language query into a valid SQLite SELECT query.
Return ONLY the SQL query, nothing else. No explanation.

User Query: {query}

SQL Query:
"""

# Prompt for formatting SQL results into natural language
RESULT_FORMATTING_PROMPT = """
User asked: {query}

SQL query executed: {sql}

Query results:
{results}

Provide a clear, natural language response to the user's question based on these results.
Be concise but include relevant details. Format numbers as currency where appropriate.

Response:
"""


class StructuredQueryAgent(CustomQueryEngine):
    """
    Query engine for structured metadata queries.
    
    Converts natural language to SQL, executes, and formats results.
    """
    
    _metadata_store: MetadataStore = PrivateAttr()
    _llm: OpenAI = PrivateAttr()
    
    def __init__(
        self,
        metadata_store: MetadataStore,
        llm: Optional[OpenAI] = None,
        **kwargs
    ):
        """
        Initialize the structured query agent.
        
        Args:
            metadata_store: MetadataStore instance for SQL queries
            llm: LLM for NL-to-SQL conversion (default: GPT-4)
        """
        super().__init__(**kwargs)
        self._metadata_store = metadata_store
        self._llm = llm or OpenAI(model=OPENAI_MODEL, api_key=OPENAI_API_KEY, temperature=0)
    
    def custom_query(self, query_str: str) -> str:
        """
        Process a natural language query against the metadata store.
        
        Steps:
        1. Convert NL to SQL using LLM
        2. Validate SQL (safety check)
        3. Execute query
        4. Format results as natural language
        
        Args:
            query_str: Natural language query
        
        Returns:
            Natural language response
        """
        logger.info(f"Structured agent processing: {query_str}")
        
        # Step 1: Convert to SQL
        sql_prompt = NL_TO_SQL_PROMPT.format(query=query_str)
        
        try:
            sql_response = self._llm.complete(sql_prompt)
            sql_query = sql_response.text.strip()
            
            # Clean up SQL (remove markdown if present)
            if sql_query.startswith("```"):
                lines = sql_query.split("\n")
                sql_query = "\n".join(lines[1:-1])
            sql_query = sql_query.strip().rstrip(";")
            
            logger.info(f"Generated SQL: {sql_query}")
            
        except Exception as e:
            logger.error(f"Error generating SQL: {e}")
            return f"I encountered an error understanding your query: {str(e)}"
        
        # Step 2: Validate SQL (basic safety)
        if not self._validate_sql(sql_query):
            return "I can only perform read queries on the claims database."
        
        # Step 3: Execute query
        try:
            results = self._metadata_store.query(sql_query)
            
            if results.empty:
                return "No matching claims found for your query."
            
            results_str = results.to_string(index=False, max_rows=20)
            logger.info(f"Query returned {len(results)} rows")
            
        except Exception as e:
            logger.error(f"SQL execution error: {e}")
            return f"I encountered an error executing the query: {str(e)}"
        
        # Step 4: Format results
        try:
            format_prompt = RESULT_FORMATTING_PROMPT.format(
                query=query_str,
                sql=sql_query,
                results=results_str
            )
            
            response = self._llm.complete(format_prompt)
            return response.text.strip()
            
        except Exception as e:
            logger.error(f"Error formatting results: {e}")
            # Fall back to raw results
            return f"Query results:\n{results_str}"
    
    def _validate_sql(self, sql: str) -> bool:
        """
        Basic SQL validation for safety.
        
        Only allows SELECT queries (no INSERT, UPDATE, DELETE, DROP, etc.)
        
        Args:
            sql: SQL query string
        
        Returns:
            True if query is safe, False otherwise
        """
        sql_upper = sql.upper().strip()
        
        # Must start with SELECT
        if not sql_upper.startswith("SELECT"):
            logger.warning(f"Blocked non-SELECT query: {sql[:50]}")
            return False
        
        # Block dangerous keywords
        dangerous = ["INSERT", "UPDATE", "DELETE", "DROP", "CREATE", "ALTER", "TRUNCATE"]
        for keyword in dangerous:
            if keyword in sql_upper:
                logger.warning(f"Blocked query with {keyword}: {sql[:50]}")
                return False
        
        return True


def create_structured_agent(
    metadata_store: MetadataStore,
    llm: Optional[OpenAI] = None
) -> StructuredQueryAgent:
    """
    Factory function to create a structured query agent.
    
    Args:
        metadata_store: MetadataStore instance
        llm: Optional LLM override
    
    Returns:
        StructuredQueryAgent instance
    """
    return StructuredQueryAgent(metadata_store=metadata_store, llm=llm)


# Quick test
if __name__ == "__main__":
    from src.config import validate_config
    from pathlib import Path
    
    print("Testing structured agent...")
    
    try:
        validate_config()
        
        # Create in-memory store with test data
        store = MetadataStore(db_path=Path(":memory:"))
        
        # Insert test data
        test_claims = [
            {
                "claim_id": "CLM-2024-001847",
                "claim_type": "Auto Accident",
                "claimant": "Robert J. Mitchell",
                "claim_status": "SETTLED",
                "total_value": 14050.33,
                "incident_date": "2024-10-15"
            },
            {
                "claim_id": "CLM-2024-003012",
                "claim_type": "Slip and Fall",
                "claimant": "Patricia Vaughn",
                "claim_status": "SETTLED",
                "total_value": 142500.00,
                "incident_date": "2024-09-20"
            },
            {
                "claim_id": "CLM-2024-005234",
                "claim_type": "Travel Insurance",
                "claimant": "Amanda Foster",
                "claim_status": "OPEN",
                "total_value": 16650.00,
                "incident_date": "2024-10-25"
            }
        ]
        
        for claim in test_claims:
            store.insert_claim(claim)
        
        # Create agent
        agent = create_structured_agent(store)
        
        # Test queries
        test_queries = [
            "Get claim CLM-2024-001847",
            "Show me all claims over $50,000",
            "Which claims are still open?",
            "What is the average claim value?"
        ]
        
        for query in test_queries:
            print(f"\n📊 Query: {query}")
            print("-" * 40)
            response = agent.custom_query(query)
            print(response)
        
        print("\n✅ Structured agent test complete!")
        store.close()
        
    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback
        traceback.print_exc()

