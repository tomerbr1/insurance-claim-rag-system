"""
Combined Grader - Orchestrates all three grader types.

Combines results from:
1. Code-based graders (deterministic)
2. Model-based graders (LLM judge - Gemini)
3. Human graders (manual grades from SQLite)

Produces unified evaluation reports with:
- Per-agent breakdown
- Consensus scores
- Grader agreement metrics
"""

import logging
from dataclasses import dataclass, field, asdict
from typing import List, Dict, Any, Optional, Tuple
from datetime import datetime

from src.graders.code_graders import CodeGradeResult, InsuranceClaimsCodeGraders
from src.graders.human_graders import HumanGraderStore, HumanGrade

logger = logging.getLogger(__name__)


@dataclass
class ModelGradeResult:
    """Result from model-based grader (LLMJudge)."""
    correctness_score: float
    correctness_reasoning: str
    relevancy_score: float
    relevancy_reasoning: str
    recall_score: float
    recall_reasoning: str


@dataclass
class CombinedGradeResult:
    """Combined result from all grader types."""
    query: str
    response: str
    expected_agent: str
    actual_agent: str
    ground_truth: str

    # Individual grader results
    code_grade: Optional[CodeGradeResult] = None
    model_grade: Optional[ModelGradeResult] = None
    human_grade: Optional[HumanGrade] = None

    # Consensus metrics
    consensus_score: Optional[float] = None
    grader_agreement: Optional[float] = None

    # Metadata
    timestamp: str = field(default_factory=lambda: datetime.now().isoformat())
    eval_run_id: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        result = {
            'query': self.query,
            'response': self.response[:500],  # Truncate for readability
            'expected_agent': self.expected_agent,
            'actual_agent': self.actual_agent,
            'ground_truth': self.ground_truth,
            'consensus_score': self.consensus_score,
            'grader_agreement': self.grader_agreement,
            'timestamp': self.timestamp,
            'eval_run_id': self.eval_run_id,
        }

        if self.code_grade:
            result['code_grade'] = {
                'score': self.code_grade.score,
                'passed': self.code_grade.passed,
                'checks': self.code_grade.checks,
                'grader_type': self.code_grade.grader_type,
            }

        if self.model_grade:
            result['model_grade'] = {
                'correctness': self.model_grade.correctness_score,
                'relevancy': self.model_grade.relevancy_score,
                'recall': self.model_grade.recall_score,
            }

        if self.human_grade:
            result['human_grade'] = {
                'level': self.human_grade.grade_level,
                'label': self.human_grade.grade_label,
                'reasoning': self.human_grade.reasoning,
            }

        return result


@dataclass
class AgentEvalSummary:
    """Summary of evaluation for a single agent type."""
    agent_type: str
    total_queries: int
    routing_accuracy: float

    # Code grader metrics
    code_avg_score: float
    code_pass_rate: float

    # Model grader metrics
    model_avg_correctness: float
    model_avg_relevancy: float
    model_avg_recall: float

    # Human grader metrics (if available)
    human_avg_grade: Optional[float] = None
    human_count: int = 0

    # Agreement metrics
    code_model_agreement: Optional[float] = None


@dataclass
class EvaluationReport:
    """Full evaluation report with all metrics."""
    eval_run_id: str
    timestamp: str
    total_queries: int

    # Overall metrics
    overall_routing_accuracy: float
    overall_code_score: float
    overall_model_correctness: float
    overall_consensus_score: float

    # Per-agent summaries
    agent_summaries: Dict[str, AgentEvalSummary]

    # Detailed results
    results: List[CombinedGradeResult]

    # Grader agreement
    code_model_correlation: Optional[float] = None
    human_model_correlation: Optional[float] = None

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            'eval_run_id': self.eval_run_id,
            'timestamp': self.timestamp,
            'total_queries': self.total_queries,
            'overall': {
                'routing_accuracy': self.overall_routing_accuracy,
                'code_score': self.overall_code_score,
                'model_correctness': self.overall_model_correctness,
                'consensus_score': self.overall_consensus_score,
            },
            'by_agent': {
                name: asdict(summary)
                for name, summary in self.agent_summaries.items()
            },
            'correlations': {
                'code_model': self.code_model_correlation,
                'human_model': self.human_model_correlation,
            },
            'results': [r.to_dict() for r in self.results],
        }


class CombinedGrader:
    """
    Orchestrates all three grader types and produces combined reports.

    Usage:
        grader = CombinedGrader()
        result = grader.grade_single(query, response, expected, actual_agent, metadata)
        report = grader.generate_report(all_results)
    """

    def __init__(
        self,
        llm_judge=None,
        human_store: Optional[HumanGraderStore] = None,
        include_human: bool = True
    ):
        """
        Initialize the combined grader.

        Args:
            llm_judge: LLMJudge instance for model grading
            human_store: HumanGraderStore for retrieving human grades
            include_human: Whether to include human grades in results
        """
        self.code_graders = InsuranceClaimsCodeGraders()
        self.llm_judge = llm_judge
        self.human_store = human_store or HumanGraderStore()
        self.include_human = include_human

    def grade_single(
        self,
        query: str,
        response: str,
        expected: Dict[str, Any],
        actual_agent: str,
        metadata: Optional[Dict[str, Any]] = None,
        eval_run_id: Optional[str] = None
    ) -> CombinedGradeResult:
        """
        Grade a single query/response with all available graders.

        Args:
            query: Original query
            response: Agent's response
            expected: Expected values dict
            actual_agent: Agent that handled the query
            metadata: Optional response metadata
            eval_run_id: Optional evaluation run identifier

        Returns:
            CombinedGradeResult with all grader outputs
        """
        result = CombinedGradeResult(
            query=query,
            response=response,
            expected_agent=expected.get('expected_agent', ''),
            actual_agent=actual_agent,
            ground_truth=expected.get('ground_truth', ''),
            eval_run_id=eval_run_id
        )

        # 1. Code-based grading
        result.code_grade = self.code_graders.grade(
            query, response, expected, actual_agent, metadata
        )

        # 2. Model-based grading (if judge available)
        if self.llm_judge:
            try:
                correctness_score, correctness_reason = self.llm_judge.evaluate_correctness(
                    response, expected.get('ground_truth', '')
                )
                relevancy_score, relevancy_reason = self.llm_judge.evaluate_relevancy(
                    query, response
                )
                recall_score, recall_reason = self.llm_judge.evaluate_recall(
                    expected.get('expected_claims', []),
                    self._extract_claims(response)
                )

                result.model_grade = ModelGradeResult(
                    correctness_score=correctness_score,
                    correctness_reasoning=correctness_reason,
                    relevancy_score=relevancy_score,
                    relevancy_reasoning=relevancy_reason,
                    recall_score=recall_score,
                    recall_reasoning=recall_reason
                )

                # Save for human comparison
                self.human_store.save_model_grade(
                    query, correctness_score, relevancy_score, recall_score, eval_run_id
                )

            except Exception as e:
                logger.error(f"Model grading failed: {e}")

        # 3. Human grading (lookup existing)
        if self.include_human:
            result.human_grade = self.human_store.get_human_grade(query)

        # Calculate consensus
        result.consensus_score, result.grader_agreement = self._calculate_consensus(result)

        return result

    def _extract_claims(self, text: str) -> List[str]:
        """Extract claim IDs from text."""
        import re
        return re.findall(r'CLM-\d{4}-\d{6}', text)

    def _calculate_consensus(
        self,
        result: CombinedGradeResult
    ) -> Tuple[Optional[float], Optional[float]]:
        """
        Calculate consensus score and grader agreement.

        Returns:
            Tuple of (consensus_score, agreement_score)
        """
        scores = []

        if result.code_grade:
            scores.append(('code', result.code_grade.score))

        if result.model_grade:
            scores.append(('model', result.model_grade.correctness_score))

        if result.human_grade:
            # Convert 1-5 scale to 0-1
            human_normalized = (result.human_grade.grade_level - 1) / 4.0
            scores.append(('human', human_normalized))

        if not scores:
            return None, None

        # Consensus = weighted average (human weighted higher if available)
        weights = {'code': 1.0, 'model': 1.5, 'human': 2.0}
        total_weight = sum(weights[name] for name, _ in scores)
        consensus = sum(weights[name] * score for name, score in scores) / total_weight

        # Agreement = 1 - standard deviation (normalized)
        if len(scores) >= 2:
            mean = sum(s for _, s in scores) / len(scores)
            variance = sum((s - mean) ** 2 for _, s in scores) / len(scores)
            std_dev = variance ** 0.5
            agreement = 1.0 - min(std_dev, 1.0)  # Cap at 1
        else:
            agreement = 1.0

        return consensus, agreement

    def generate_report(
        self,
        results: List[CombinedGradeResult],
        eval_run_id: Optional[str] = None
    ) -> EvaluationReport:
        """
        Generate a full evaluation report from combined results.

        Args:
            results: List of CombinedGradeResult objects
            eval_run_id: Optional evaluation run identifier

        Returns:
            EvaluationReport with all metrics
        """
        if not results:
            raise ValueError("No results to generate report from")

        eval_run_id = eval_run_id or f"eval_{datetime.now().strftime('%Y%m%d_%H%M%S')}"

        # Calculate overall metrics
        routing_correct = sum(
            1 for r in results
            if r.expected_agent and r.actual_agent == r.expected_agent
        )
        total_with_expected = sum(1 for r in results if r.expected_agent)
        overall_routing = routing_correct / total_with_expected if total_with_expected else 0

        code_scores = [r.code_grade.score for r in results if r.code_grade]
        model_scores = [r.model_grade.correctness_score for r in results if r.model_grade]
        consensus_scores = [r.consensus_score for r in results if r.consensus_score is not None]

        overall_code = sum(code_scores) / len(code_scores) if code_scores else 0
        overall_model = sum(model_scores) / len(model_scores) if model_scores else 0
        overall_consensus = sum(consensus_scores) / len(consensus_scores) if consensus_scores else 0

        # Per-agent summaries
        agent_summaries = {}
        for agent_type in ['structured', 'summary', 'needle']:
            agent_results = [r for r in results if r.actual_agent == agent_type]
            if not agent_results:
                continue

            # Routing accuracy for this agent
            agent_routing_correct = sum(
                1 for r in agent_results
                if r.expected_agent == agent_type
            )
            agent_routing_expected = sum(
                1 for r in agent_results if r.expected_agent
            )

            # Code grader metrics
            agent_code_scores = [
                r.code_grade.score for r in agent_results if r.code_grade
            ]
            agent_code_passed = sum(
                1 for r in agent_results if r.code_grade and r.code_grade.passed
            )

            # Model grader metrics
            agent_model_correct = [
                r.model_grade.correctness_score for r in agent_results if r.model_grade
            ]
            agent_model_relevancy = [
                r.model_grade.relevancy_score for r in agent_results if r.model_grade
            ]
            agent_model_recall = [
                r.model_grade.recall_score for r in agent_results if r.model_grade
            ]

            # Human grader metrics
            agent_human_grades = [
                r.human_grade.grade_level for r in agent_results if r.human_grade
            ]

            agent_summaries[agent_type] = AgentEvalSummary(
                agent_type=agent_type,
                total_queries=len(agent_results),
                routing_accuracy=(
                    agent_routing_correct / agent_routing_expected
                    if agent_routing_expected else 0
                ),
                code_avg_score=(
                    sum(agent_code_scores) / len(agent_code_scores)
                    if agent_code_scores else 0
                ),
                code_pass_rate=(
                    agent_code_passed / len(agent_results)
                    if agent_results else 0
                ),
                model_avg_correctness=(
                    sum(agent_model_correct) / len(agent_model_correct)
                    if agent_model_correct else 0
                ),
                model_avg_relevancy=(
                    sum(agent_model_relevancy) / len(agent_model_relevancy)
                    if agent_model_relevancy else 0
                ),
                model_avg_recall=(
                    sum(agent_model_recall) / len(agent_model_recall)
                    if agent_model_recall else 0
                ),
                human_avg_grade=(
                    sum(agent_human_grades) / len(agent_human_grades)
                    if agent_human_grades else None
                ),
                human_count=len(agent_human_grades)
            )

        # Calculate correlations
        code_model_correlation = self._calculate_correlation(
            [(r.code_grade.score, r.model_grade.correctness_score)
             for r in results if r.code_grade and r.model_grade]
        )

        human_model_pairs = [
            ((r.human_grade.grade_level - 1) / 4.0, r.model_grade.correctness_score)
            for r in results if r.human_grade and r.model_grade
        ]
        human_model_correlation = self._calculate_correlation(human_model_pairs)

        return EvaluationReport(
            eval_run_id=eval_run_id,
            timestamp=datetime.now().isoformat(),
            total_queries=len(results),
            overall_routing_accuracy=overall_routing,
            overall_code_score=overall_code,
            overall_model_correctness=overall_model,
            overall_consensus_score=overall_consensus,
            agent_summaries=agent_summaries,
            results=results,
            code_model_correlation=code_model_correlation,
            human_model_correlation=human_model_correlation
        )

    def _calculate_correlation(
        self,
        pairs: List[Tuple[float, float]]
    ) -> Optional[float]:
        """Calculate Pearson correlation coefficient."""
        if len(pairs) < 3:
            return None

        n = len(pairs)
        x = [p[0] for p in pairs]
        y = [p[1] for p in pairs]

        sum_x = sum(x)
        sum_y = sum(y)
        sum_xy = sum(a * b for a, b in pairs)
        sum_x2 = sum(a ** 2 for a in x)
        sum_y2 = sum(b ** 2 for b in y)

        numerator = n * sum_xy - sum_x * sum_y
        denominator = ((n * sum_x2 - sum_x ** 2) * (n * sum_y2 - sum_y ** 2)) ** 0.5

        if denominator == 0:
            return None

        return numerator / denominator


def print_report_summary(report: EvaluationReport):
    """Print a formatted summary of the evaluation report."""
    print("\n" + "=" * 70)
    print(f"EVALUATION REPORT: {report.eval_run_id}")
    print(f"Generated: {report.timestamp}")
    print("=" * 70)

    print(f"\n OVERALL METRICS ({report.total_queries} queries)")
    print("-" * 40)
    print(f"  Routing Accuracy:    {report.overall_routing_accuracy:.1%}")
    print(f"  Code Grader Score:   {report.overall_code_score:.2f}")
    print(f"  Model Correctness:   {report.overall_model_correctness:.2f}")
    print(f"  Consensus Score:     {report.overall_consensus_score:.2f}")

    print(f"\n BY AGENT TYPE")
    print("-" * 40)
    for agent, summary in report.agent_summaries.items():
        print(f"\n  {agent.upper()} ({summary.total_queries} queries)")
        print(f"    Routing Accuracy: {summary.routing_accuracy:.1%}")
        print(f"    Code Score:       {summary.code_avg_score:.2f} ({summary.code_pass_rate:.0%} passed)")
        print(f"    Model Correct:    {summary.model_avg_correctness:.2f}")
        if summary.human_avg_grade:
            print(f"    Human Grade:      {summary.human_avg_grade:.1f}/5 ({summary.human_count} graded)")

    print(f"\n GRADER CORRELATIONS")
    print("-" * 40)
    if report.code_model_correlation is not None:
        print(f"  Code vs Model:  {report.code_model_correlation:.3f}")
    if report.human_model_correlation is not None:
        print(f"  Human vs Model: {report.human_model_correlation:.3f}")

    print("\n" + "=" * 70)


# Quick test
if __name__ == "__main__":
    from src.graders.code_graders import InsuranceClaimsCodeGraders

    grader = CombinedGrader(llm_judge=None, include_human=False)

    # Test single grading
    result = grader.grade_single(
        query="Get claim CLM-2024-001847",
        response="Claim CLM-2024-001847: Auto Accident, Robert Mitchell, $14,050.33",
        expected={
            'expected_agent': 'structured',
            'expected_claims': ['CLM-2024-001847'],
            'ground_truth': 'Auto Accident, Robert Mitchell, $14,050.33'
        },
        actual_agent='structured'
    )

    print("Single result:")
    print(f"  Code score: {result.code_grade.score:.2f}")
    print(f"  Consensus: {result.consensus_score}")
    print(f"  Agreement: {result.grader_agreement}")
