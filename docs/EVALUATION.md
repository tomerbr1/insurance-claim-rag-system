# Evaluation System

This document describes the evaluation methodology and grading system for the Insurance Claims RAG System.

## Overview

The system uses a **three-type grading approach** inspired by Anthropic's "Demystifying Evals for AI Agents":

| Grader Type | Implementation | Purpose |
|-------------|----------------|---------|
| **Code-Based** | Deterministic | Routing accuracy, claim retrieval, amounts, references |
| **Model-Based** | LLM (Gemini) | Semantic correctness, relevancy scoring |
| **Human** | Manual CLI | Calibration, edge case validation |

## Methodology

We use **LLM-as-Judge** evaluation with a completely different provider (Gemini 2.5 Flash) evaluating responses from the main system (OpenAI GPT-4). This avoids potential bias from having OpenAI evaluate its own outputs.

## Metrics

| Metric | Description | Scoring |
|--------|-------------|---------|
| **Correctness** | Does answer match ground truth? | 0.0 - 1.0 |
| **Relevancy** | Was retrieved context relevant? | 0.0 - 1.0 |
| **Recall** | Were correct documents retrieved? | 0.0 - 1.0 |
| **Routing** | Did router choose correct agent? | Boolean |

## Test Cases

Test queries are organized by expected routing in `src/test_data.py`:

**Structured Queries (12)**:
- Exact lookups ("Get claim CLM-2024-001847")
- Range filters ("Claims over $50k")
- Aggregations ("Average claim value")

**Summary Queries (11)**:
- Claim overviews ("What happened in this claim?")
- Timeline narratives ("Summarize the events")
- Multi-claim summaries

**Needle Queries (25)**:
- Exact amounts ("What was the towing cost?")
- Reference numbers ("Wire transfer number?")
- Precise times and dates
- Person names and identifiers

**Router Edge Cases (10)**:
- Ambiguous queries for routing accuracy testing

## Agent-Specific Graders

Each agent has specialized grading logic:

### StructuredAgentGrader
- Validates claim retrieval accuracy
- Checks aggregation results
- Verifies SQL query patterns

### SummaryAgentGrader
- Measures claim coverage in responses
- Checks response length appropriateness
- Validates presence of narrative elements

### NeedleAgentGrader
- Verifies exact amount extraction
- Checks reference number accuracy
- Validates numeric precision

### RouterGrader
- Measures routing accuracy
- Validates signal matching
- Tests edge case handling

## Running Evaluation

### Basic Evaluation (Model-Based)

```bash
python main.py --eval
```

### Multi-Grader Evaluation

```bash
# All graders with HTML report
python main.py --graders

# With subset and human grades
python main.py --graders --eval-subset 10 --include-human

# Rate limit handling
python main.py --eval --eval-delay 5
```

### Human Grading CLI

```bash
# Grade responses interactively
python -m src.graders.human_graders grade

# Compare human vs model grades
python -m src.graders.human_graders compare

# View grading statistics
python -m src.graders.human_graders stats

# Export grades to JSON
python -m src.graders.human_graders export
```

## Sample Results

```
EVALUATION SUMMARY
════════════════════════════════
OVERALL SCORES:
   Correctness:      0.85
   Relevancy:        0.90
   Recall:           0.82
   Routing Accuracy: 91%

BY QUERY TYPE:
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

## Output Files

Evaluation generates output in `eval_runs/`:

- `eval_YYYYMMDD_HHMMSS_report.html` - HTML evaluation reports
- `human_grades.db` - SQLite storage for human grades
- `responses_to_grade.json` - Responses queued for manual grading

## Adding Test Cases

Add new test queries to `src/test_data.py`:

```python
# For agent queries:
{
    "query": "Your new query",
    "ground_truth": "Expected answer",
    "expected_claims": ["CLM-2024-XXXXXX"]  # For recall scoring
}

# For router edge cases:
{
    "query": "Ambiguous or edge case query",
    "expected_agent": "structured",  # or "summary" or "needle"
    "routing_rationale": "Why this should route here"
}
```
