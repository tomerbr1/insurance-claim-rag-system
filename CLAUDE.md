# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## System Overview

This is a multi-agent RAG (Retrieval-Augmented Generation) system for insurance claim document analysis using a **hybrid architecture** that combines SQL for structured queries with semantic search for unstructured narratives.

**Framework**: LlamaIndex (NOT LangChain)
**Python Version**: 3.11.14 (required)

## Core Architecture

### Three-Way Agent Routing

The system routes queries to specialized agents:

1. **STRUCTURED Agent** (SQL-based)
   - Exact lookups, filters, aggregations
   - Queries SQLite metadata store directly
   - Examples: "Claims over $50k", "Get claim CLM-2024-001847"

2. **SUMMARY Agent** (RAG on summary index)
   - High-level overviews, narrative questions
   - Uses MapReduce-generated summaries at 3 levels (chunk/section/document)
   - Examples: "What happened in claim X?", "Timeline of events"

3. **NEEDLE Agent** (RAG with auto-merging)
   - Precise fact extraction from document text
   - Uses hierarchical chunking (1024/256 tokens) with auto-merging retrieval
   - Examples: "Exact towing cost?", "Wire transfer number?"

### Data Flow

```
PDFs → Table Extraction (pdfplumber) → Text + Tables
  ↓
  LLM Metadata Extraction → Split to:
  ├─ SQLite (structured data) → Structured Agent
  ├─ MapReduce Summaries → Summary Agent
  └─ Hierarchical Chunks → Vector Index (ChromaDB) → Needle Agent
                           ↑
                           └─ MCP Tools (system introspection)
```

## Key Design Decisions

### 1. LLM-Based Metadata Extraction (No Regex)

**ALWAYS** use LLM for metadata extraction, NEVER regex patterns.

```python
# Correct approach
metadata = extract_metadata_with_llm(document_text, llm)

# Wrong - DO NOT USE
# metadata = re.search(r"Claim ID: (\w+)", text)
```

**Why**: Handles format variations (dates, currency), semantic understanding, no brittle regex maintenance.

### 2. Hierarchical Chunking Strategy

Two-level hierarchy (sufficient for insurance documents):

| Level | Tokens | Purpose | Indexed? |
|-------|--------|---------|----------|
| Large | 1024 | Full sections, broad context | No (in docstore only) |
| Small | 256 | Precise facts (amounts, dates) | Yes (in ChromaDB) |

**Overlap**: 20 tokens
**Auto-merge threshold**: 50% (merge to parent when >50% of children retrieved)

### 3. Multi-Level Summary Index (MapReduce)

The system stores intermediate summaries at ALL levels:
- **Chunk-level** summaries (~100-200 nodes)
- **Section-level** summaries (~10 nodes)
- **Document-level** summaries (~10 nodes)

All summary levels are vectorized and retrievable. This enables multi-granularity retrieval.

### 4. Table Extraction from PDFs

The system uses `pdfplumber` to extract structured tables from PDF documents:
- Tables are detected and converted to markdown format
- Appended to document text for better context
- Minimum validation (2 rows, 2 columns by default)
- Can be disabled via `EXTRACT_TABLES=false` in `.env`

**Why**: Insurance claims often contain tabular data (itemized expenses, damage assessments) that should be preserved in structured form for better RAG retrieval.

### 5. Unbiased Evaluation

**CRITICAL**: Use Gemini (Google) to evaluate OpenAI outputs to avoid self-evaluation bias.

Both API keys required:
- `OPENAI_API_KEY`: Main agents, routing, processing
- `GOOGLE_API_KEY`: Evaluation judge (different provider)

## Common Commands

### Running the System

```bash
# Interactive mode (prompts to clean or reuse existing data)
python main.py

# Instant startup (reuse existing data, no cleanup prompt)
python main.py --no-cleanup

# Evaluation mode (all test cases)
python main.py --eval

# Evaluation with rate limit handling
python main.py --eval --eval-subset 3 --eval-delay 5

# Build indexes only
python main.py --build

# Verbose output
python main.py -v
```

### Rebuilding Indexes

If you need to rebuild indexes after changes:

```python
from src.cleanup import cleanup_all
cleanup_all()  # Then run: python main.py --build
```

## LlamaIndex Patterns

### Imports

```python
# Use LlamaIndex, NOT LangChain
from llama_index.core import VectorStoreIndex, SummaryIndex
from llama_index.llms.openai import OpenAI
from llama_index.embeddings.openai import OpenAIEmbedding
from llama_index.core.node_parser import HierarchicalNodeParser
from llama_index.core.retrievers import AutoMergingRetriever
```

### LLM Initialization

```python
# Main LLM for agents
llm = OpenAI(model="gpt-4", temperature=0)

# Cost-effective LLM for metadata extraction
extraction_llm = OpenAI(model="gpt-4o-mini", temperature=0)

# Embeddings
embed_model = OpenAIEmbedding(model="text-embedding-3-small")
```

## File Structure Reference

### Core Modules

- `main.py` - Entry point & orchestrator
- `src/config.py` - Configuration, model settings, API validation, MCP settings
- `src/data_loader.py` - PDF loading + LLM metadata extraction + table extraction
- `src/metadata_store.py` - SQLite for structured queries
- `src/chunking.py` - Hierarchical node parser (2 levels)
- `src/indexing.py` - Summary (MapReduce) & vector index builders
- `src/retrieval.py` - Auto-merging retriever
- `src/evaluation.py` - LLM-as-judge evaluation (Gemini)
- `src/cleanup.py` - Index cleanup utilities, data reuse validation

### Agent Modules

- `src/agents/router_agent.py` - 3-way query router
- `src/agents/structured_agent.py` - SQL-based queries
- `src/agents/summary_agent.py` - High-level RAG
- `src/agents/needle_agent.py` - Precise fact retrieval

### MCP Integration

- `src/mcp/chromadb_client.py` - MCP ChromaDB client and tool creation

**MCP (Model Context Protocol)** integration enables:
- System introspection via standardized tools
- ChromaDB collection statistics and queries
- Visible tool calls for demonstration purposes
- Router agent uses MCP tools to gather system context before routing

Currently operates in "direct" mode (using ChromaDB library directly), designed to support future MCP server mode.

### Utility Scripts

- `scripts/generate_table_pdfs.py` - Generate synthetic insurance claim PDFs with embedded tables for testing table extraction

## Performance Expectations

### Query Latency

| Query Type | Expected Latency |
|------------|------------------|
| Structured (SQL) | ~5-10ms |
| Summary (RAG) | ~500ms |
| Needle (RAG) | ~600ms |

### Build Time

**Full Build**: ~2-3 minutes for 13 documents (with table extraction)

**Phase Breakdown**:
- PDF loading + table extraction: ~5-10s
- Metadata extraction (LLM): ~25s (13 calls)
- Chunking: ~10s
- Summary index (MapReduce): ~90-120s (most expensive, ~150-200 LLM calls)
- Vector index: ~30-45s (~150 embedding calls)

**Cost per rebuild**: ~$0.10-0.20

**Instant Startup**: When reusing existing data (--no-cleanup or user chooses to keep data), startup takes <5 seconds

## Development Guidelines

### When Making Changes to Agents

1. Agent prompts should follow this structure:

```python
AGENT_PROMPT = """
You are a [role description].
Your role is to [specific task].

When answering:
- [Guideline 1]
- [Guideline 2]
- [Guideline 3]

Context from claims:
{context}

Question: {query}
"""
```

2. Always include error handling:

```python
try:
    response = query_engine.query(query)
except Exception as e:
    logger.error(f"Query failed: {str(e)}")
    return f"Error processing query: {str(e)}"
```

### Testing Changes

Add new test cases to `tests/test_queries.py`:

```python
TEST_CASES.append({
    "query": "Your new query",
    "expected_agent": "needle",  # structured | summary | needle
    "ground_truth": "Expected answer",
    "expected_chunks": ["CLM-2024-XXXXXX"]
})
```

Run evaluation:
```bash
python main.py --eval
```

### Debugging Tips

**Check routing decision**:
```python
if hasattr(response, 'metadata') and 'selector_result' in response.metadata:
    print(f"Routed to: {response.metadata['selector_result']}")
```

**Inspect retrieved chunks**:
```python
for node in response.source_nodes:
    print(f"Claim: {node.metadata.get('claim_id')}")
    print(f"Score: {node.score}")
    print(f"Text: {node.text[:200]}...")
```

**Verify ChromaDB contents** (using MCP):
```python
collections = mcp_client.list_collections()
stats = mcp_client.get_collection_info("claims")
```

## Environment Configuration

Required in `.env`:
```
OPENAI_API_KEY=sk-...        # Required for agents
GOOGLE_API_KEY=...           # Required for unbiased evaluation
```

Optional (with defaults):
```
OPENAI_MODEL=gpt-4
OPENAI_MINI_MODEL=gpt-4o-mini
OPENAI_EMBEDDING_MODEL=text-embedding-3-small
GEMINI_EVAL_MODEL=gemini-2.5-flash
CHROMA_COLLECTION_NAME=insurance_claims

# MCP (Model Context Protocol) Configuration
MCP_MODE=direct                # "direct" for ChromaDB library access, "server" for future MCP server
MCP_ENABLED=true               # Enable/disable MCP integration
MCP_LOG_TOOL_CALLS=true        # Log MCP tool calls for visibility (demo mode)

# Table Extraction Configuration
EXTRACT_TABLES=true            # Enable table extraction from PDFs using pdfplumber
TABLE_MIN_ROWS=2               # Minimum rows for a valid table
TABLE_MIN_COLS=2               # Minimum columns for a valid table
```

## Critical Constraints

### DO

- Use LlamaIndex (not LangChain)
- Use LLM for metadata extraction (not regex)
- Store intermediate summaries (not just final)
- Use 3-way routing (structured/summary/needle)
- Use Gemini for evaluation (different provider)
- Follow hierarchical chunking (1024/256 tokens)
- Extract tables from PDFs using pdfplumber (when enabled)

### DO NOT

- Use regex for metadata extraction
- Skip intermediate summary storage
- Use same LLM provider for evaluation
- Hard-code API keys in source files
- Use NLTK without silencing download messages (use `src/utils/nltk_silencer.py`)

## Database Schema

### SQLite Metadata Store (`claims_metadata.db`)

```sql
CREATE TABLE claims (
    claim_id TEXT PRIMARY KEY,      -- e.g., "CLM-2024-001847"
    claim_type TEXT,                -- e.g., "Auto Accident", "Slip and Fall"
    claimant TEXT,                  -- Full name
    policy_number TEXT,
    claim_status TEXT,              -- OPEN/CLOSED/SETTLED
    total_value REAL,               -- Dollar amount
    incident_date DATE,             -- YYYY-MM-DD
    filing_date DATE,
    settlement_date DATE            -- Nullable
);
```

### ChromaDB Collections

- `insurance_claims` - Leaf nodes (small chunks) for needle retrieval
- Document metadata preserved in each node for filtering
