# Design Decisions

This document explains the key architectural and implementation decisions made in the Insurance Claims RAG System.

## Summary of Trade-offs

| Decision | Benefit | Cost |
|----------|---------|------|
| LLM for metadata | Robust extraction | Higher cost, slower |
| Store all summaries | Flexible retrieval | More storage |
| 3-way routing | Better accuracy | More complexity |
| Small chunks (256) | Precise retrieval | May lose context |
| Hybrid SQL + RAG | Fast structured queries | Two storage systems |
| Table extraction | Preserve tabular data | Additional processing |

---

## 1. Hybrid Architecture: SQL + RAG

**Decision**: Use SQL for structured queries, RAG for narrative queries.

**Rationale**:
- Insurance claims contain both structured data (dates, IDs, amounts) and unstructured narratives
- SQL queries are **100x faster** for exact matches and filters
- No hallucination on structured data - SQL is deterministic
- RAG provides semantic understanding where SQL cannot

**Performance Comparison**:

| Query | Pure RAG | Hybrid |
|-------|----------|--------|
| "Get claim CLM-001847" | ~500ms | ~5ms |
| "Claims over $50k" | ~1000ms | ~10ms |
| "What happened?" | ~500ms | ~500ms |

---

## 2. LLM-Based Metadata Extraction (No Regex)

**Decision**: Use GPT-4o-mini to extract metadata from PDFs instead of regex patterns.

**Rationale**:
- **Format flexibility**: Handles date variations (Oct 15, 2024 / 2024-10-15 / 10/15/2024)
- **Semantic understanding**: Distinguishes incident_date from filing_date from settlement_date
- **Robustness**: No brittle regex patterns to maintain as document formats evolve
- **Consistency**: Aligns with RAG-first philosophy

**Trade-offs**:
- Higher cost (~$0.01 per document vs free regex)
- Slower (~2s per document vs instant)
- **BUT**: More reliable and production-ready

---

## 3. Multi-Level Summary Index (MapReduce)

**Decision**: Store intermediate summaries at chunk, section, and document levels.

**Rationale**:
- Different queries need different granularity
- "What were the repair costs?" → Section summary (detailed)
- "Overview of claim" → Document summary (high-level)
- Enables flexible retrieval at appropriate detail level

### MapReduce Implementation

The system uses a true MapReduce pattern with LLM calls:

**MAP Phase** (Per-chunk summarization):
- Input: ~10-20 small chunks per claim
- Process: Each chunk → GPT-4 → 2-3 sentence summary
- Output: Chunk-level summary nodes (stored as retrievable TextNodes)
- LLM calls: ~100-200 total (10-20 per claim × 10 claims)

**REDUCE Phase 1** (Section-level aggregation):
- Input: All chunk summaries for a claim
- Process: Combine summaries → GPT-4 → 3-5 sentence section summary
- Output: Section-level summary nodes
- LLM calls: ~10 (1 per claim)

**REDUCE Phase 2** (Document-level aggregation):
- Input: Section summaries for a claim
- Process: Create comprehensive summary → GPT-4 → 1 paragraph
- Output: Document-level summary node
- LLM calls: ~10 (1 per claim)

### Storage Structure

```
Document CLM-2024-001847
├── Document Summary (1 node) ← stored in VectorStoreIndex
│   metadata: {'summary_level': 'document', 'claim_id': 'CLM-2024-001847'}
│   text: "Auto accident claim by Robert Mitchell, total $14,050.33,
│          settled Nov 30, 2024. Intersection collision, vehicle towed..."
│
├── Section Summary (1 node) ← stored in VectorStoreIndex
│   metadata: {'summary_level': 'section', 'claim_id': 'CLM-2024-001847'}
│   text: "Incident occurred Oct 15, 2024 at Main St intersection.
│          Towing: $185 (Invoice #T-8827). Repairs: $8,500..."
│
└── Chunk Summaries (15 nodes) ← stored in VectorStoreIndex
    ├── metadata: {'summary_level': 'chunk', 'chunk_index': 0, ...}
    │   text: "Police report by Officer Thompson, Badge #4421..."
    ├── metadata: {'summary_level': 'chunk', 'chunk_index': 1, ...}
    │   text: "Tow Invoice #T-8827, dated Oct 15, amount $185.00..."
    └── ...
```

**All summary levels are vectorized and retrievable**, enabling the Summary Agent to find the most appropriate granularity based on the query.

---

## 4. Hierarchical Chunking with Auto-Merging

**Decision**: Use two chunk sizes (1024/256 tokens) with auto-merging retrieval.

**Rationale**:
- **Small chunks (256)**: Capture precise facts (amounts, dates, IDs)
- **Large chunks (1024)**: Full sections for comprehensive understanding
- **Auto-merge**: Automatically expands context when needed
- Two levels are sufficient for insurance claims documents

### Chunking Parameters

| Level | Tokens | Overlap | Purpose | Indexed? |
|-------|--------|---------|---------|----------|
| Small | 256 | 20 | Precise facts | Yes (in ChromaDB) |
| Large | 1024 | 20 | Full sections | No (in DocStore only) |

### How Auto-Merging Works

1. **Initial Retrieval**:
   - Query → ChromaDB vector search → Top-k small chunks (k=5)
   - Example: Query "towing cost" retrieves 5 small chunks about towing

2. **Merge Decision**:
   - For each retrieved small chunk, check its siblings (other children of same parent)
   - If >50% of siblings are also retrieved → merge to parent (medium chunk)
   - Example: 3 out of 5 small chunks share same medium parent → merge to medium

3. **Recursive Merging**:
   - Apply same logic to medium chunks
   - Can merge medium → large if threshold met
   - Example: If multiple medium chunks from same section retrieved → expand to large

4. **Context Expansion**:
   - User gets expanded context automatically
   - No need to manually adjust chunk size
   - Balances precision with context

**Auto-Merge Threshold**: 50% (configurable in `AutoMergingRetriever`)

### Example Retrieval Flow

```
Query: "What was the exact towing cost in CLM-2024-001847?"

Step 1 - Initial retrieval:
  → Retrieved 5 small chunks (256 tokens each)
  → chunk_0: "Tow Invoice #T-8827..."
  → chunk_1: "Amount: $185.00..."
  → chunk_2: "Date: Oct 15, 2024..."
  → chunk_8: "Other claim detail..."
  → chunk_15: "Another detail..."

Step 2 - Check merge conditions:
  → chunk_0, chunk_1, chunk_2 share parent: large_0
  → 3 out of 4 children retrieved (75% > 50% threshold)
  → MERGE to large_0

Step 3 - Return expanded context:
  → Original: 768 tokens (3 × 256)
  → After merge: 1024 tokens (1 large chunk)
  → Includes full section context about towing incident
```

This ensures precise retrieval with automatic context expansion when needed.

---

## 5. MCP Integration

**Decision**: Integrate with ChromaDB MCP server rather than custom tools.

**Rationale**:
- Industry-standard protocol for LLM tool integration
- Community-maintained servers
- Richer functionality than custom tools
- More relevant to production systems

**Implementation**:

The system uses an MCP-style wrapper (`ChromaDBMCPClient`) that provides:

| Tool | Description | Used By |
|------|-------------|---------|
| `list_collections` | List all vector store collections | Router Agent |
| `collection_stats` | Get collection document count | Router Agent |
| `collection_count` | Quick document count | Router Agent |
| `direct_search` | Semantic similarity search | Available |
| `peek_collection` | Preview collection documents | Available |

The Router Agent calls MCP tools before making routing decisions, demonstrating agent-tool interaction in action.

---

## 6. Table Extraction (pdfplumber)

**Decision**: Extract tables from PDFs using pdfplumber and convert to markdown for LLM consumption.

**Rationale**:
- Insurance documents often contain tabular data (financial breakdowns, coverage limits, treatment timelines)
- Standard PDF text extraction (pypdf) loses table structure, making data hard to interpret
- pdfplumber preserves table structure as 2D arrays
- Markdown format is well-understood by LLMs and maintains visual structure

### Implementation

The system uses a dual-extraction approach:
1. **Base text**: pypdf extracts narrative content
2. **Tables**: pdfplumber detects and extracts tables as 2D arrays
3. **Conversion**: Tables are converted to markdown format
4. **Integration**: Tables are appended to document text with markers

### Table-Enabled Documents

| Claim ID | Table Type | Key Data |
|----------|------------|----------|
| CLM-2024-006001 | Financial Breakdown | 8 line items, costs, dates |
| CLM-2024-006002 | Coverage/Policy | 7 coverage types with limits |
| CLM-2024-006003 | Treatment Timeline | 12 events with status |

### Example Extracted Table

```
[EXTRACTED TABLE - Page 1]
| Item             | Category   | Amount     | Date       |
|------------------|------------|------------|------------|
| Roof replacement | Structural | $12,500.00 | 2024-09-15 |
| Water cleanup    | Restoration| $3,200.00  | 2024-09-12 |
[END TABLE]
```

### Configuration

- `EXTRACT_TABLES=true` - Enable/disable table extraction
- `TABLE_MIN_ROWS=2` - Minimum rows for valid table
- `TABLE_MIN_COLS=2` - Minimum columns for valid table

### Test Queries for Tables

- "What was the exact cost of roof replacement?" → $12,500.00
- "What is the Medical Payments coverage limit?" → $5,000
- "How much did the MRI cost?" → $2,800.00
