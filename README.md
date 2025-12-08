# Insurance Claims RAG System

A multi-agent retrieval-augmented generation (RAG) system for insurance claim document analysis, implementing a **hybrid architecture** that combines SQL for structured queries with semantic search for unstructured narratives.

## 📋 Table of Contents

- [Overview](#overview)
- [Architecture](#architecture)
- [Key Design Decisions](#key-design-decisions)
- [Installation](#installation)
- [Usage](#usage)
- [Evaluation](#evaluation)
- [Project Structure](#project-structure)
- [Limitations & Trade-offs](#limitations--trade-offs)

---

## Overview

This system processes 10 insurance claim documents and provides intelligent query routing to answer both:
- **Structured queries**: "Show me all claims over $100k" → SQL
- **Narrative queries**: "What happened in claim X?" → RAG

### Features

✅ **Hybrid Architecture** - SQL + RAG for optimal performance  
✅ **3-Way Query Routing** - Intelligent classification to specialized agents  
✅ **Hierarchical Chunking** - Multi-level chunks (128/512/1536 tokens)  
✅ **Auto-Merging Retrieval** - Automatic context expansion  
✅ **Multi-Level Summaries** - Chunk → Section → Document summaries  
✅ **LLM-as-Judge Evaluation** - Automated quality assessment  
✅ **MCP Integration** - ChromaDB vector store access  

---

## Architecture

### High-Level System Flow

```
┌─────────────────┐
│   User Query    │
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│  Router Agent   │ ◄── Classifies query type
│  (3-way routing)│
└────────┬────────┘
         │
    ┌────┴────┬────────────┐
    │         │            │
    ▼         ▼            ▼
┌───────┐ ┌───────┐ ┌──────────┐
│STRUCT │ │SUMMARY│ │  NEEDLE  │
│ Agent │ │ Agent │ │  Agent   │
└───┬───┘ └───┬───┘ └────┬─────┘
    │         │          │
    ▼         ▼          ▼
┌───────┐ ┌───────┐ ┌──────────┐
│SQLite │ │Summary│ │Hierarchi-│
│  DB   │ │ Index │ │cal Index │
└───────┘ └───────┘ └──────────┘
    │         │          │
    └─────────┴──────────┘
              │
              ▼
      ┌──────────────┐
      │   Response   │
      └──────────────┘
```

### Agent Routing Logic

| Query Type | Route To | Example |
|------------|----------|---------|
| Exact lookups | **Structured** | "Get claim CLM-2024-001847" |
| Filters & aggregations | **Structured** | "Claims over $50k", "Average value" |
| High-level overviews | **Summary** | "What happened in this claim?" |
| Timeline narratives | **Summary** | "Summarize the events" |
| Precise facts | **Needle** | "What was the exact towing cost?" |
| Reference IDs | **Needle** | "Wire transfer number?" |

### Data Flow

```
PDF Documents
     │
     ▼
┌────────────────────────────────────────────┐
│           DATA LOADING                      │
│  • Load PDFs with SimpleDirectoryReader    │
│  • Extract metadata using LLM (not regex)  │
└────────────────┬───────────────────────────┘
                 │
        ┌────────┴────────┐
        │                 │
        ▼                 ▼
┌───────────────┐  ┌────────────────────────┐
│ METADATA STORE│  │ HIERARCHICAL CHUNKING  │
│   (SQLite)    │  │  Small: 128 tokens     │
│               │  │  Medium: 512 tokens    │
│ • claim_id    │  │  Large: 1536 tokens    │
│ • claim_type  │  └───────────┬────────────┘
│ • total_value │              │
│ • dates       │     ┌────────┴────────┐
└───────────────┘     │                 │
                      ▼                 ▼
               ┌────────────┐    ┌────────────┐
               │  SUMMARY   │    │  VECTOR    │
               │   INDEX    │    │   INDEX    │
               │            │    │ (ChromaDB) │
               │ MapReduce: │    │            │
               │ • Chunk    │    │ Leaf nodes │
               │ • Section  │    │ only       │
               │ • Document │    │            │
               └────────────┘    └────────────┘
```

---

## Key Design Decisions

### 1. Hybrid Architecture: SQL + RAG

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

### 2. LLM-Based Metadata Extraction (No Regex)

**Decision**: Use GPT-3.5-turbo to extract metadata from PDFs instead of regex patterns.

**Rationale**:
- **Format flexibility**: Handles date variations (Oct 15, 2024 / 2024-10-15 / 10/15/2024)
- **Semantic understanding**: Distinguishes incident_date from filing_date from settlement_date
- **Robustness**: No brittle regex patterns to maintain as document formats evolve
- **Consistency**: Aligns with RAG-first philosophy

**Trade-offs**:
- Higher cost (~$0.01 per document vs free regex)
- Slower (~2s per document vs instant)
- **BUT**: More reliable and production-ready

### 3. Multi-Level Summary Index (MapReduce)

**Decision**: Store intermediate summaries at chunk, section, and document levels.

**Rationale**:
- Different queries need different granularity
- "What were the repair costs?" → Section summary (detailed)
- "Overview of claim" → Document summary (high-level)
- Enables flexible retrieval at appropriate detail level

**Structure**:
```
Document CLM-2024-001847
├── Document Summary (1 node)
│   "Auto accident claim, Robert Mitchell, $14,050.33 settled..."
│
├── Section Summaries (3-5 nodes)
│   ├── "Incident: Oct 15, 2024, intersection collision..."
│   ├── "Costs: Towing $185, repairs $8,500..."
│   └── "Timeline: Filed Oct 16, settled Nov 30..."
│
└── Chunk Summaries (10-20 nodes)
    ├── "Police report by Officer Thompson, Badge #4421..."
    ├── "Tow Invoice #T-8827, $185.00..."
    └── ...
```

### 4. Hierarchical Chunking with Auto-Merging

**Decision**: Use three chunk sizes (128/512/1536 tokens) with auto-merging retrieval.

**Rationale**:
- **Small chunks (128)**: Capture precise facts (amounts, dates, IDs)
- **Medium chunks (512)**: Provide context for timeline events
- **Large chunks (1536)**: Full sections for comprehensive understanding
- **Auto-merge**: Automatically expands context when needed

**Parameters**:
| Level | Tokens | Overlap | Purpose |
|-------|--------|---------|---------|
| Small | 128 | 20 | Precise facts |
| Medium | 512 | 50 | Balanced context |
| Large | 1536 | 200 | Full sections |

**Auto-Merge Threshold**: 50% - If more than half of a parent's children are retrieved, merge to parent.

### 5. MCP Integration

**Decision**: Integrate with ChromaDB MCP server rather than custom tools.

**Rationale**:
- Industry-standard protocol for LLM tool integration
- Community-maintained servers
- Richer functionality than custom tools
- More relevant to production systems

---

## Installation

### Prerequisites

- **Python 3.11.14** (required)
- **OpenAI API key** (required) - Get from https://platform.openai.com/api-keys
- **Google API key** (required) - Get from https://makersuite.google.com/app/apikey

**⚠️ Both API keys are required!**
- OpenAI: For main agents, routing, and processing
- Google: For unbiased evaluation (Gemini evaluates OpenAI outputs)

### Setup

```bash
# Clone or navigate to project
cd insurance-claim-rag-system

# Ensure you have Python 3.11.14 installed
python --version  # Should show Python 3.11.14

# Create virtual environment
python -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Configure environment
cp env.example .env
# Edit .env and add BOTH API keys:
#   OPENAI_API_KEY=sk-your-key-here
#   GOOGLE_API_KEY=your-key-here
```

**Important:** The system will validate both API keys at startup and fail fast if either is missing or invalid. This ensures unbiased evaluation throughout.

---

## Usage

### Interactive Mode

```bash
python main.py
```

Example queries:
```
💬 Get claim CLM-2024-001847
🔀 Routed to: STRUCTURED agent
💡 Auto Accident claim by Robert J. Mitchell, total $14,050.33, SETTLED

💬 What happened in the slip and fall claim?
🔀 Routed to: SUMMARY agent
💡 Patricia Vaughn suffered a slip and fall at Sunny Days Cafe...

💬 What was the exact towing cost in CLM-2024-001847?
🔀 Routed to: NEEDLE agent
💡 The towing cost was $185.00 (Tow Invoice #T-8827)
```

### Evaluation Mode

```bash
python main.py --eval
```

Runs 11 test cases and reports:
- Correctness score (0-1)
- Relevancy score (0-1)
- Recall score (0-1)
- Routing accuracy

### Build Only

```bash
python main.py --build
```

Builds indexes without entering interactive mode.

---

## Evaluation

### Methodology

We use **LLM-as-Judge** evaluation with a completely different provider (Gemini 2.5 Flash) evaluating responses from the main system (OpenAI GPT-4). This avoids potential bias from having OpenAI evaluate its own outputs.

### Metrics

| Metric | Description | Scoring |
|--------|-------------|---------|
| **Correctness** | Does answer match ground truth? | 0.0 - 1.0 |
| **Relevancy** | Was retrieved context relevant? | 0.0 - 1.0 |
| **Recall** | Were correct documents retrieved? | 0.0 - 1.0 |
| **Routing** | Did router choose correct agent? | Boolean |

### Test Cases

**Structured Queries (3)**:
- Exact lookups
- Range filters
- Status checks

**Summary Queries (2)**:
- Claim overviews
- Multi-claim summaries

**Needle Queries (6)**:
- Exact amounts
- Reference numbers
- Precise times
- Person names

### Sample Results

```
📊 EVALUATION SUMMARY
════════════════════════════════
📈 OVERALL SCORES:
   Correctness:      0.85
   Relevancy:        0.90
   Recall:           0.82
   Routing Accuracy: 91%

📋 BY QUERY TYPE:
   STRUCTURED:
      Avg Correctness: 0.93
      Routing Accuracy: 100%
   
   SUMMARY:
      Avg Correctness: 0.88
      Routing Accuracy: 100%
   
   NEEDLE:
      Avg Correctness: 0.80
      Routing Accuracy: 83%
```

---

## Project Structure

```
insurance-claim-rag-system/
├── main.py                    # Entry point & orchestrator
├── requirements.txt           # Python dependencies
├── env.example               # Environment template
├── README.md                 # This file
│
├── insurance_claims_data/    # PDF claim documents
│   ├── CLM_2024_001847.pdf
│   ├── ... (10 PDFs)
│   └── README.md             # Data documentation
│
├── src/
│   ├── config.py             # Configuration & settings
│   ├── data_loader.py        # PDF loading + LLM metadata extraction
│   ├── metadata_store.py     # SQLite for structured queries
│   ├── chunking.py           # Hierarchical node parser
│   ├── indexing.py           # Summary & vector index builders
│   ├── retrieval.py          # Auto-merging retriever
│   ├── evaluation.py         # LLM-as-judge evaluation
│   │
│   ├── agents/
│   │   ├── router_agent.py   # 3-way query router
│   │   ├── structured_agent.py # SQL-based queries
│   │   ├── summary_agent.py  # High-level RAG
│   │   └── needle_agent.py   # Precise fact retrieval
│   │
│   └── mcp/
│       └── chromadb_client.py # MCP integration
│
└── tests/
    └── test_queries.py       # Test query definitions
```

---

## Limitations & Trade-offs

### Known Limitations

1. **Cold Start**: Initial index building takes 2-3 minutes due to LLM summarization
2. **Cost**: LLM-based metadata extraction costs ~$0.10 per full build
3. **Context Window**: Very large claims may exceed context limits
4. **Routing Accuracy**: Edge cases between summary/needle can misroute

### Trade-offs Made

| Decision | Benefit | Cost |
|----------|---------|------|
| LLM for metadata | Robust extraction | Higher cost, slower |
| Store all summaries | Flexible retrieval | More storage |
| 3-way routing | Better accuracy | More complexity |
| Small chunks (128) | Precise retrieval | May lose context |

### Future Improvements

- [ ] Add caching for LLM calls
- [ ] Implement streaming responses
- [ ] Add document section detection for better chunking
- [ ] Support multi-modal (images in PDFs)
- [ ] Add conversation history for follow-up queries

---

## Models Used

| Component | Model | Provider | Purpose |
|-----------|-------|----------|---------|
| **Agents & Routing** | GPT-4 | OpenAI | Query processing, response generation |
| **Metadata Extraction** | GPT-4o-mini | OpenAI | Extract structured data from PDFs (cost-effective) |
| **Embeddings** | text-embedding-3-small | OpenAI | Vector representations |
| **Evaluation Judge** | Gemini 2.5 Flash | Google | Score system responses (different provider for unbiased evaluation) |

---

## License

This project is part of an academic assignment.

---

## Author

Insurance Claims RAG System - Tomer Brami

