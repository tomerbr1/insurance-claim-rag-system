# System Architecture

This document provides detailed architectural information about the Insurance Claims RAG System.

## High-Level System Flow

```
┌─────────────────┐
│   User Query    │
│ "What was the   │
│  towing cost?"  │
└────────┬────────┘
         │
         ▼
┌─────────────────────────────────┐
│      Router Agent (GPT-4)       │
│  Classifies query into:         │
│  • STRUCTURED (SQL metadata)    │
│  • SUMMARY (high-level overview)│
│  • NEEDLE (precise fact)        │
└────────┬────────────────────────┘
         │
    ┌────┴────┬────────────┐
    │         │            │
    ▼         ▼            ▼
┌───────┐ ┌───────┐ ┌──────────┐
│STRUCT │ │SUMMARY│ │  NEEDLE  │
│ Agent │ │ Agent │ │  Agent   │
│(GPT-4)│ │(GPT-4)│ │ (GPT-4)  │
└───┬───┘ └───┬───┘ └────┬─────┘
    │         │          │
    │         │          │ Auto-merging
    ▼         ▼          ▼ retrieval
┌───────┐ ┌───────┐ ┌──────────┐
│SQLite │ │Summary│ │Hierarchi-│
│  DB   │ │ Index │ │cal Index │
│       │ │       │ │(ChromaDB)│
│10 rows│ │~200   │ │~150 leaf │
│       │ │nodes  │ │nodes     │
└───────┘ └───────┘ └──────────┘
    │         │          │
    │         │          │
    └─────────┴──────────┘
              │
              ▼
      ┌──────────────┐
      │   Response   │
      │ + Citations  │
      └──────────────┘
```

## Agent Routing Logic

| Query Type | Route To | Example |
|------------|----------|---------|
| Exact lookups | **Structured** | "Get claim CLM-2024-001847" |
| Filters & aggregations | **Structured** | "Claims over $50k", "Average value" |
| High-level overviews | **Summary** | "What happened in this claim?" |
| Timeline narratives | **Summary** | "Summarize the events" |
| Precise facts | **Needle** | "What was the exact towing cost?" |
| Reference IDs | **Needle** | "Wire transfer number?" |

## Data Flow

```
PDF Documents (13 claims, 3 with tables)
     │
     ▼
┌────────────────────────────────────────────┐
│           DATA LOADING                      │
│  • Load PDFs with SimpleDirectoryReader    │
│  • Extract tables using pdfplumber         │
│  • Convert tables to markdown format       │
│  • Extract metadata using LLM (GPT-4o-mini)│
│    (not regex - handles format variations) │
└────────────────┬───────────────────────────┘
                 │
        ┌────────┴────────┐
        │                 │
        ▼                 ▼
┌───────────────┐  ┌────────────────────────┐
│ METADATA STORE│  │ HIERARCHICAL CHUNKING  │
│   (SQLite)    │  │  2 levels with overlap │
│               │  │                        │
│ • claim_id    │  │  Large:  1024 tokens   │
│ • claim_type  │  │    └─ Small: 256       │
│ • total_value │  │                        │
│ • dates, etc. │  │  Overlap: 20 tokens    │
│               │  │                        │
│ Used by:      │  │  Creates parent-child  │
│ Structured    │  │  relationships in      │
│ Agent         │  │  DocumentStore         │
└───────────────┘              │
                      ┌────────┴────────┐
                      │                 │
                      ▼                 ▼
               ┌────────────┐    ┌────────────┐
               │  SUMMARY   │    │  VECTOR    │
               │   INDEX    │    │   INDEX    │
               │            │    │ (ChromaDB) │
               │ MapReduce: │    │            │
               │ • Chunk    │    │ Leaf nodes │
               │ • Section  │    │ only       │
               │ • Document │    │            │
               │            │    │ Auto-      │
               │ ~200 nodes │    │ merging    │
               │            │    │ retrieval  │
               │ Used by:   │    │            │
               │ Summary    │    │ Used by:   │
               │ Agent      │    │ Needle     │
               └────────────┘    │ Agent      │
                                 └────────────┘

                      ┌────────────────┐
                      │ Router Agent   │
                      │ (Coordinates   │
                      │  all 3 agents) │
                      └────────────────┘
```

## Processing Pipeline

| Phase | Component | Output | Notes |
|-------|-----------|--------|-------|
| 1. Load | `SimpleDirectoryReader` + pdfplumber | 13 Documents | Raw PDF text + tables |
| 2. Tables | pdfplumber | Markdown tables | Appended to text |
| 3. Extract | LLM (GPT-4o-mini) | Metadata dicts | ~$0.01 cost |
| 4. Store | SQLite + Chunking | DB + Nodes | Dual path |
| 5. Summarize | MapReduce (GPT-4) | Summary nodes | 3 levels |
| 6. Embed | ChromaDB + OpenAI | Vector index | Leaf nodes only |
| 7. Build | Agent creation | 4 agents | Router + 3 specialists |
| 8. Query | Runtime routing | Responses | Per-query execution |

**Total build time:** ~2-3 minutes | **Total cost:** ~$0.10-0.15

## Detailed Phase Breakdown

| Phase | Time | LLM Calls | Storage | Notes |
|-------|------|-----------|---------|-------|
| **1. Load PDFs** | ~5s | 0 | Memory | 13 PDFs → Document objects |
| **2. Extract Tables** | ~2s | 0 | Memory | pdfplumber → markdown |
| **3. Extract Metadata** | ~25s | 13 | SQLite | GPT-4o-mini, 1 call/doc |
| **4. Chunk Documents** | ~10s | 0 | Memory + DocStore | 2 levels, ~250-350 nodes |
| **5. Build Summary Index** | ~90-120s | ~180-220 | VectorStore | Most expensive phase |
| **6. Build Vector Index** | ~30-45s | ~180 | ChromaDB | Embedding generation |
| **7. Create Agents** | ~1s | 0 | Memory | Configure 4 agents |
| **8. Ready for Queries** | Instant | 1-2/query | - | Runtime queries |

## Storage Architecture

### Storage Footprint

- **SQLite database**: ~65 KB (13 rows of structured metadata)
- **ChromaDB vector store**: ~6-12 MB (embeddings + metadata)
- **In-memory objects**: ~15-20 MB (document store, nodes, agents)

### SQLite Schema

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

## Cost Analysis

### Per Full Rebuild

- Metadata extraction (GPT-4o-mini): ~$0.01-0.02
- Summary generation (GPT-4): ~$0.08-0.12
- Embedding generation (text-embedding-3-small): ~$0.01
- **Total per rebuild**: ~$0.10-0.15

### Runtime Query Costs

- Structured query: ~$0.001 (minimal LLM use)
- Summary query: ~$0.005-0.01 (routing + generation)
- Needle query: ~$0.01-0.02 (routing + retrieval + generation)

## MCP Integration

The system uses an MCP-style wrapper (`ChromaDBMCPClient`) for ChromaDB access:

| Tool | Description | Used By |
|------|-------------|---------|
| `list_collections` | List all vector store collections | Router Agent |
| `collection_stats` | Get collection document count | Router Agent |
| `collection_count` | Quick document count | Router Agent |
| `direct_search` | Semantic similarity search | Available |
| `peek_collection` | Preview collection documents | Available |

### Router Agent MCP Integration

The Router Agent calls MCP tools before making routing decisions:

```
💬 Query: What was the towing cost?

🔧 [MCP TOOL CALL] collection_stats()
   └─ Result: Collection 'insurance_claims': 150 documents
🔧 [MCP TOOL CALL] collection_stats('insurance_claims_summaries')
   └─ Result: Collection 'insurance_claims_summaries': 200 documents

🔀 Routed to: NEEDLE agent
💡 Answer: The towing cost was $185.00 (Tow Invoice #T-8827)
```

Use the `mcp` command in interactive mode to view MCP client status and test tools.
