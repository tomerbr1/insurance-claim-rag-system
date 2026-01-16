# Insurance Claims RAG System

A multi-agent RAG system for insurance claim document analysis using a **hybrid SQL + semantic search architecture**.

## Features

- **Hybrid Architecture** - SQL for structured queries, RAG for narratives
- **3-Way Query Routing** - Intelligent classification to specialized agents
- **Hierarchical Chunking** - 2-level chunks (1024/256 tokens) with auto-merging
- **Multi-Level Summaries** - MapReduce: chunk → section → document
- **Table Extraction** - Automatic table detection from PDFs (pdfplumber)
- **LLM-as-Judge Evaluation** - Unbiased assessment (Gemini evaluates OpenAI)
- **MCP Integration** - ChromaDB vector store access

## Quick Start

### Prerequisites

- **Python 3.11.14** (required)
- **OpenAI API key** - Get from https://platform.openai.com/api-keys
- **Google API key** - Get from https://makersuite.google.com/app/apikey

### Setup

```bash
# Create virtual environment
python -m venv .venv
source .venv/bin/activate  # Windows: .venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Configure environment
cp env.example .env
# Edit .env and add BOTH API keys:
#   OPENAI_API_KEY=sk-your-key-here
#   GOOGLE_API_KEY=your-key-here

# Run the system
python main.py
```

## Architecture Overview

```
User Query → Router Agent → STRUCTURED (SQL) | SUMMARY (RAG) | NEEDLE (RAG)
                              ↓                   ↓               ↓
                           SQLite            Summary Index    Hierarchical Index
                           (metadata)        (~200 nodes)     (ChromaDB, ~150 nodes)
```

| Query Type | Route To | Example |
|------------|----------|---------|
| Exact lookups, filters | **Structured** | "Get claim CLM-2024-001847", "Claims over $50k" |
| High-level overviews | **Summary** | "What happened in this claim?" |
| Precise facts | **Needle** | "What was the exact towing cost?" |

See [Architecture Documentation](docs/ARCHITECTURE.md) for detailed diagrams and data flow.

## Usage

### Interactive Mode

```bash
python main.py              # Prompts to rebuild or reuse data
python main.py --no-cleanup # Instant startup (reuse existing data)
```

Example session:
```
💬 Get claim CLM-2024-001847
🔀 Routed to: STRUCTURED agent
💡 Auto Accident claim by Robert J. Mitchell, total $14,050.33, SETTLED

💬 What was the exact towing cost?
🔀 Routed to: NEEDLE agent
💡 The towing cost was $185.00 (Tow Invoice #T-8827)
```

### Evaluation

```bash
python main.py --eval                    # Basic model-based evaluation
python main.py --graders                 # Multi-grader with HTML report
python main.py --graders --eval-subset 5 # First 5 queries, all graders
```

See [Evaluation Documentation](docs/EVALUATION.md) for grading methodology.

## Documentation

| Document | Description |
|----------|-------------|
| [Architecture](docs/ARCHITECTURE.md) | System diagrams, data flow, storage details |
| [Design Decisions](docs/DESIGN_DECISIONS.md) | Rationale for key architectural choices |
| [Evaluation](docs/EVALUATION.md) | Grading system, metrics, running evals |

## Project Structure

```
insurance-claim-rag-system/
├── main.py                    # Entry point, interactive CLI
├── core.py                    # System orchestrator
├── requirements.txt
├── env.example
│
├── insurance_claims_data/     # PDF claim documents (13 total)
│
├── src/
│   ├── config.py              # Configuration & settings
│   ├── data_loader.py         # PDF loading + table extraction
│   ├── metadata_store.py      # SQLite for structured queries
│   ├── chunking.py            # Hierarchical node parser
│   ├── indexing.py            # Summary & vector index builders
│   ├── retrieval.py           # Auto-merging retriever
│   ├── evaluation.py          # LLM-as-judge evaluation
│   │
│   ├── agents/
│   │   ├── router_agent.py    # 3-way query router
│   │   ├── structured_agent.py
│   │   ├── summary_agent.py
│   │   └── needle_agent.py
│   │
│   ├── graders/               # Three-type grading system
│   └── mcp/                   # MCP integration
│
├── docs/                      # Detailed documentation
└── tests/
```

## Models

| Component | Model | Provider |
|-----------|-------|----------|
| Agents & Routing | GPT-4 | OpenAI |
| Metadata Extraction | GPT-4o-mini | OpenAI |
| Embeddings | text-embedding-3-small | OpenAI |
| Evaluation Judge | Gemini 2.5 Flash | Google |

## Limitations

| Limitation | Impact |
|------------|--------|
| Cold start (~2-3 min) | Initial build requires LLM summarization |
| Build cost (~$0.10) | LLM-based metadata and summary generation |
| Routing edge cases | Summary/needle boundary can misroute |

See [Design Decisions](docs/DESIGN_DECISIONS.md) for trade-off analysis.

---

**Author**: Tomer Brami | **License**: Academic project
