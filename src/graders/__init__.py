"""
Graders Package - Multi-type evaluation graders for RAG system.

This package implements three types of graders based on Anthropic's
"Demystifying Evals for AI Agents" article:

1. Code-Based Graders (code_graders.py)
   - Deterministic, fast, reproducible
   - Agent-specific grading logic
   - Checks routing, claim retrieval, amounts, references

2. Model-Based Graders (via LLMJudge in evaluation.py)
   - Semantic evaluation using Gemini
   - Correctness, relevancy scoring
   - Unbiased (different provider from agents)

3. Human Graders (human_graders.py)
   - CLI interface for manual grading
   - SQLite storage for grades
   - Calibration comparison with model grades

4. Combined Grader (combined_grader.py)
   - Orchestrates all three grader types
   - Computes consensus scores
   - Generates unified evaluation reports
"""

from src.graders.code_graders import (
    CodeGradeResult,
    StructuredAgentGrader,
    SummaryAgentGrader,
    NeedleAgentGrader,
    RouterGrader,
    InsuranceClaimsCodeGraders,
)

from src.graders.combined_grader import (
    CombinedGradeResult,
    CombinedGrader,
    EvaluationReport,
    print_report_summary,
)

from src.graders.report_generator import (
    generate_html_report,
    generate_html_from_dict,
)

from src.graders.human_graders import (
    HumanGrade,
    HumanGraderStore,
    HumanGraderCLI,
)

__all__ = [
    # Code graders
    'CodeGradeResult',
    'StructuredAgentGrader',
    'SummaryAgentGrader',
    'NeedleAgentGrader',
    'RouterGrader',
    'InsuranceClaimsCodeGraders',
    # Combined grader
    'CombinedGradeResult',
    'CombinedGrader',
    'EvaluationReport',
    'print_report_summary',
    # Report generator
    'generate_html_report',
    'generate_html_from_dict',
    # Human graders
    'HumanGrade',
    'HumanGraderStore',
    'HumanGraderCLI',
]
