"""
Code-Based Graders - Deterministic grading for RAG system responses.

Each agent type has specialized grading logic:

- STRUCTURED: SQL correctness, metadata accuracy, aggregation results
- SUMMARY: Claim coverage, response length appropriateness, key facts
- NEEDLE: Exact value matching, reference number validation, precision
- ROUTER: Routing accuracy, classification correctness

Based on Anthropic's "Demystifying Evals for AI Agents" article.
"""

import re
import logging
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import List, Dict, Any, Optional, Set, Tuple

logger = logging.getLogger(__name__)


@dataclass
class CodeGradeResult:
    """Result from a code-based grader."""
    score: float  # 0.0 to 1.0
    passed: bool  # Binary pass/fail
    checks: Dict[str, bool]  # Individual check results
    details: Dict[str, Any]  # Additional details
    grader_type: str  # "structured", "summary", "needle", "router"

    def __post_init__(self):
        if not 0.0 <= self.score <= 1.0:
            raise ValueError(f"Score must be between 0.0 and 1.0, got {self.score}")


class BaseAgentGrader(ABC):
    """Abstract base class for agent-specific graders."""

    @property
    @abstractmethod
    def grader_type(self) -> str:
        """Return the grader type identifier."""
        pass

    @abstractmethod
    def grade(
        self,
        query: str,
        response: str,
        expected: Dict[str, Any],
        metadata: Optional[Dict[str, Any]] = None
    ) -> CodeGradeResult:
        """
        Grade a response for this agent type.

        Args:
            query: The original query
            response: The agent's response
            expected: Expected values (ground_truth, expected_claims, etc.)
            metadata: Optional metadata from the response

        Returns:
            CodeGradeResult with score and details
        """
        pass

    def _extract_claim_ids(self, text: str) -> Set[str]:
        """Extract claim IDs from text using regex."""
        pattern = r'CLM-\d{4}-\d{6}'
        return set(re.findall(pattern, text))

    def _extract_dollar_amounts(self, text: str) -> List[float]:
        """Extract dollar amounts from text."""
        pattern = r'\$[\d,]+(?:\.\d{2})?'
        matches = re.findall(pattern, text)
        amounts = []
        for match in matches:
            try:
                clean = match.replace('$', '').replace(',', '')
                amounts.append(float(clean))
            except ValueError:
                continue
        return amounts

    def _extract_reference_numbers(self, text: str) -> Set[str]:
        """Extract various reference numbers from text."""
        patterns = [
            r'WPT-\d{4}-\d+',  # Wire transfer refs
            r'#T-\d+',  # Tow invoice
            r'#\d{4,}',  # Badge numbers, generic refs
            r'\d{6}[A-Z]{2}',  # Rolex refs like 116618LB
        ]
        refs = set()
        for pattern in patterns:
            refs.update(re.findall(pattern, text))
        return refs


class StructuredAgentGrader(BaseAgentGrader):
    """
    Grader for STRUCTURED agent (SQL-based queries).

    Checks:
    - Claim ID accuracy in lookups
    - Correct claims returned for filters
    - Aggregation value reasonableness
    - Response contains expected metadata fields
    """

    @property
    def grader_type(self) -> str:
        return "structured"

    def grade(
        self,
        query: str,
        response: str,
        expected: Dict[str, Any],
        metadata: Optional[Dict[str, Any]] = None
    ) -> CodeGradeResult:
        checks = {}
        details = {}

        # Check 1: Claim ID retrieval
        expected_claims = set(expected.get('expected_claims', []))
        response_claims = self._extract_claim_ids(response)

        if expected_claims:
            claim_recall = len(expected_claims & response_claims) / len(expected_claims)
            claim_precision = (
                len(expected_claims & response_claims) / len(response_claims)
                if response_claims else 0.0
            )
            checks['claim_retrieval'] = claim_recall >= 0.8
            details['claim_recall'] = claim_recall
            details['claim_precision'] = claim_precision
            details['expected_claims'] = list(expected_claims)
            details['found_claims'] = list(response_claims)
        else:
            checks['claim_retrieval'] = True  # No specific claims expected
            details['claim_recall'] = 1.0

        # Check 2: Response is not an error
        error_indicators = ['error', 'failed', 'exception', 'could not']
        response_lower = response.lower()
        has_error = any(ind in response_lower for ind in error_indicators)
        checks['no_error'] = not has_error
        details['has_error'] = has_error

        # Check 3: SQL query executed (look for data patterns)
        has_data_pattern = bool(
            re.search(r'claim.*id|status|value|\$[\d,]+', response_lower) or
            re.search(r'\d+ claim', response_lower) or
            re.search(r'settled|open|closed', response_lower)
        )
        checks['has_structured_data'] = has_data_pattern
        details['has_structured_data'] = has_data_pattern

        # Check 4: For aggregation queries, check for numeric result
        aggregation_keywords = ['count', 'average', 'sum', 'total', 'how many']
        is_aggregation = any(kw in query.lower() for kw in aggregation_keywords)
        if is_aggregation:
            has_number = bool(re.search(r'\d+', response))
            checks['aggregation_result'] = has_number
            details['is_aggregation'] = True
            details['has_numeric_result'] = has_number
        else:
            details['is_aggregation'] = False

        # Calculate overall score
        passed_checks = sum(checks.values())
        total_checks = len(checks)
        score = passed_checks / total_checks if total_checks > 0 else 0.0

        return CodeGradeResult(
            score=score,
            passed=score >= 0.7,
            checks=checks,
            details=details,
            grader_type=self.grader_type
        )


class SummaryAgentGrader(BaseAgentGrader):
    """
    Grader for SUMMARY agent (high-level RAG).

    Checks:
    - Response covers expected claims
    - Response length is appropriate (not too short/long)
    - Contains narrative elements (not just data)
    - Key facts from ground truth appear
    """

    @property
    def grader_type(self) -> str:
        return "summary"

    def grade(
        self,
        query: str,
        response: str,
        expected: Dict[str, Any],
        metadata: Optional[Dict[str, Any]] = None
    ) -> CodeGradeResult:
        checks = {}
        details = {}

        # Check 1: Claim coverage
        expected_claims = set(expected.get('expected_claims', []))
        response_claims = self._extract_claim_ids(response)

        if expected_claims:
            claim_coverage = len(expected_claims & response_claims) / len(expected_claims)
            checks['claim_coverage'] = claim_coverage >= 0.5
            details['claim_coverage'] = claim_coverage
            details['expected_claims'] = list(expected_claims)
            details['found_claims'] = list(response_claims)
        else:
            checks['claim_coverage'] = True
            details['claim_coverage'] = 1.0

        # Check 2: Response length appropriateness
        word_count = len(response.split())
        checks['length_appropriate'] = 50 <= word_count <= 1000
        details['word_count'] = word_count
        details['length_status'] = (
            'too_short' if word_count < 50 else
            'too_long' if word_count > 1000 else
            'appropriate'
        )

        # Check 3: Contains narrative elements
        narrative_patterns = [
            r'\b(happened|occurred|incident|event)\b',
            r'\b(because|due to|resulted in|led to)\b',
            r'\b(settled|resolved|concluded)\b',
            r'\b(timeline|sequence|process)\b',
        ]
        narrative_matches = sum(
            1 for p in narrative_patterns
            if re.search(p, response.lower())
        )
        checks['has_narrative'] = narrative_matches >= 1
        details['narrative_elements'] = narrative_matches

        # Check 4: Ground truth key facts
        ground_truth = expected.get('ground_truth', '')
        if ground_truth:
            # Extract key facts from ground truth
            gt_lower = ground_truth.lower()
            response_lower = response.lower()

            # Check for key terms overlap
            gt_words = set(re.findall(r'\b\w{4,}\b', gt_lower))
            response_words = set(re.findall(r'\b\w{4,}\b', response_lower))

            if gt_words:
                fact_overlap = len(gt_words & response_words) / len(gt_words)
                checks['key_facts'] = fact_overlap >= 0.3
                details['fact_overlap'] = fact_overlap
            else:
                checks['key_facts'] = True
                details['fact_overlap'] = 1.0

        # Calculate overall score
        passed_checks = sum(checks.values())
        total_checks = len(checks)
        score = passed_checks / total_checks if total_checks > 0 else 0.0

        return CodeGradeResult(
            score=score,
            passed=score >= 0.6,
            checks=checks,
            details=details,
            grader_type=self.grader_type
        )


class NeedleAgentGrader(BaseAgentGrader):
    """
    Grader for NEEDLE agent (precise fact extraction).

    Checks:
    - Exact value matching for amounts
    - Reference number accuracy
    - Specific fact extraction
    - Precision over recall
    """

    @property
    def grader_type(self) -> str:
        return "needle"

    def grade(
        self,
        query: str,
        response: str,
        expected: Dict[str, Any],
        metadata: Optional[Dict[str, Any]] = None
    ) -> CodeGradeResult:
        checks = {}
        details = {}

        ground_truth = expected.get('ground_truth', '')
        response_lower = response.lower()
        gt_lower = ground_truth.lower()

        # Check 1: Exact value match for dollar amounts
        gt_amounts = self._extract_dollar_amounts(ground_truth)
        response_amounts = self._extract_dollar_amounts(response)

        if gt_amounts:
            amount_matches = sum(1 for a in gt_amounts if a in response_amounts)
            amount_accuracy = amount_matches / len(gt_amounts)
            checks['amount_match'] = amount_accuracy >= 0.8
            details['expected_amounts'] = gt_amounts
            details['found_amounts'] = response_amounts
            details['amount_accuracy'] = amount_accuracy
        else:
            details['has_amounts'] = False

        # Check 2: Reference number match
        gt_refs = self._extract_reference_numbers(ground_truth)
        response_refs = self._extract_reference_numbers(response)

        if gt_refs:
            ref_matches = len(gt_refs & response_refs)
            ref_accuracy = ref_matches / len(gt_refs)
            checks['reference_match'] = ref_accuracy >= 0.8
            details['expected_refs'] = list(gt_refs)
            details['found_refs'] = list(response_refs)
            details['ref_accuracy'] = ref_accuracy
        else:
            details['has_refs'] = False

        # Check 3: Specific numeric values (times, percentages, dates)
        # Extract numbers from ground truth
        gt_numbers = set(re.findall(r'\b\d+(?:\.\d+)?\b', ground_truth))
        response_numbers = set(re.findall(r'\b\d+(?:\.\d+)?\b', response))

        if gt_numbers:
            number_matches = len(gt_numbers & response_numbers)
            number_accuracy = number_matches / len(gt_numbers)
            checks['number_match'] = number_accuracy >= 0.5
            details['number_accuracy'] = number_accuracy
        else:
            checks['number_match'] = True
            details['number_accuracy'] = 1.0

        # Check 4: Key terms from ground truth
        # For needle, we want high precision on specific terms
        gt_key_terms = set(re.findall(r'\b[A-Z][a-z]+\b', ground_truth))  # Proper nouns
        if gt_key_terms:
            found_terms = sum(1 for t in gt_key_terms if t.lower() in response_lower)
            term_accuracy = found_terms / len(gt_key_terms)
            checks['key_terms'] = term_accuracy >= 0.5
            details['key_term_accuracy'] = term_accuracy
            details['expected_terms'] = list(gt_key_terms)
        else:
            checks['key_terms'] = True

        # Check 5: Response is focused (not overly verbose)
        word_count = len(response.split())
        checks['focused_response'] = word_count <= 300
        details['word_count'] = word_count

        # Check 6: Retrieved correct claim
        expected_claims = set(expected.get('expected_claims', []))
        response_claims = self._extract_claim_ids(response)

        if expected_claims:
            claim_match = len(expected_claims & response_claims) > 0
            checks['correct_claim'] = claim_match
            details['expected_claims'] = list(expected_claims)
            details['found_claims'] = list(response_claims)

        # Calculate overall score (weighted toward precision)
        if checks:
            passed_checks = sum(checks.values())
            total_checks = len(checks)
            score = passed_checks / total_checks
        else:
            score = 0.5  # No specific checks applicable

        return CodeGradeResult(
            score=score,
            passed=score >= 0.6,
            checks=checks,
            details=details,
            grader_type=self.grader_type
        )


class RouterGrader(BaseAgentGrader):
    """
    Grader for ROUTER agent (query classification).

    Checks:
    - Routing accuracy (expected vs actual agent)
    - Classification confidence
    - Edge case handling
    """

    @property
    def grader_type(self) -> str:
        return "router"

    def grade(
        self,
        query: str,
        response: str,
        expected: Dict[str, Any],
        metadata: Optional[Dict[str, Any]] = None
    ) -> CodeGradeResult:
        checks = {}
        details = {}

        expected_agent = expected.get('expected_agent', '').lower()
        actual_agent = (metadata or {}).get('routed_to', '').lower()

        # Check 1: Routing accuracy
        routing_correct = expected_agent == actual_agent
        checks['routing_correct'] = routing_correct
        details['expected_agent'] = expected_agent
        details['actual_agent'] = actual_agent

        # Check 2: Valid agent selection
        valid_agents = {'structured', 'summary', 'needle'}
        checks['valid_agent'] = actual_agent in valid_agents
        details['is_valid_agent'] = actual_agent in valid_agents

        # Check 3: Query type indicators (sanity check)
        query_lower = query.lower()

        # Structured indicators
        structured_signals = sum(1 for kw in [
            'count', 'average', 'sum', 'list all', 'get claim', 'show me',
            'which claims', 'how many', 'total value'
        ] if kw in query_lower)

        # Summary indicators
        summary_signals = sum(1 for kw in [
            'what happened', 'summarize', 'overview', 'explain', 'describe',
            'timeline', 'story', 'circumstances'
        ] if kw in query_lower)

        # Needle indicators
        needle_signals = sum(1 for kw in [
            'exact', 'specific', 'what was the', 'reference number',
            'invoice', 'how much', 'what time', 'who signed'
        ] if kw in query_lower)

        details['structured_signals'] = structured_signals
        details['summary_signals'] = summary_signals
        details['needle_signals'] = needle_signals

        # Check if routing matches strongest signal
        signal_map = {
            'structured': structured_signals,
            'summary': summary_signals,
            'needle': needle_signals
        }
        max_signal = max(signal_map.values())
        if max_signal > 0:
            strongest = [k for k, v in signal_map.items() if v == max_signal]
            checks['matches_signals'] = actual_agent in strongest or routing_correct
            details['strongest_signals'] = strongest
        else:
            checks['matches_signals'] = True  # No clear signal

        # Check 4: Routing rationale (if provided)
        if 'routing_rationale' in expected:
            details['routing_rationale'] = expected['routing_rationale']

        # Calculate score
        score = 1.0 if routing_correct else 0.0
        if not routing_correct and checks.get('matches_signals', False):
            score = 0.5  # Partial credit for reasonable routing

        return CodeGradeResult(
            score=score,
            passed=routing_correct,
            checks=checks,
            details=details,
            grader_type=self.grader_type
        )


class InsuranceClaimsCodeGraders:
    """
    Main grading interface that dispatches to agent-specific graders.

    Usage:
        graders = InsuranceClaimsCodeGraders()
        result = graders.grade(query, response, expected, actual_agent)
    """

    def __init__(self):
        self.graders = {
            'structured': StructuredAgentGrader(),
            'summary': SummaryAgentGrader(),
            'needle': NeedleAgentGrader(),
            'router': RouterGrader(),
        }

    def grade(
        self,
        query: str,
        response: str,
        expected: Dict[str, Any],
        actual_agent: str,
        metadata: Optional[Dict[str, Any]] = None
    ) -> CodeGradeResult:
        """
        Grade a response using the appropriate agent grader.

        Args:
            query: Original query
            response: Agent's response
            expected: Expected values dict
            actual_agent: The agent that handled the query
            metadata: Optional response metadata

        Returns:
            CodeGradeResult from the appropriate grader
        """
        grader = self.graders.get(actual_agent.lower())
        if not grader:
            logger.warning(f"Unknown agent type: {actual_agent}, using needle grader")
            grader = self.graders['needle']

        return grader.grade(query, response, expected, metadata)

    def grade_routing(
        self,
        query: str,
        expected_agent: str,
        actual_agent: str,
        metadata: Optional[Dict[str, Any]] = None
    ) -> CodeGradeResult:
        """
        Grade routing accuracy specifically.

        Args:
            query: Original query
            expected_agent: Expected agent type
            actual_agent: Actual agent that handled the query
            metadata: Optional routing metadata

        Returns:
            CodeGradeResult for routing
        """
        expected = {'expected_agent': expected_agent}
        metadata = metadata or {}
        metadata['routed_to'] = actual_agent

        return self.graders['router'].grade(query, '', expected, metadata)

    def grade_all(
        self,
        query: str,
        response: str,
        expected: Dict[str, Any],
        actual_agent: str,
        metadata: Optional[Dict[str, Any]] = None
    ) -> Dict[str, CodeGradeResult]:
        """
        Run both response grading and routing grading.

        Returns:
            Dict with 'response' and 'routing' CodeGradeResults
        """
        results = {}

        # Grade the response with agent-specific grader
        results['response'] = self.grade(
            query, response, expected, actual_agent, metadata
        )

        # Grade routing if expected_agent is provided
        expected_agent = expected.get('expected_agent')
        if expected_agent:
            results['routing'] = self.grade_routing(
                query, expected_agent, actual_agent, metadata
            )

        return results


# Quick test
if __name__ == "__main__":
    graders = InsuranceClaimsCodeGraders()

    # Test structured grader
    print("Testing STRUCTURED grader:")
    result = graders.grade(
        query="Get claim CLM-2024-001847",
        response="Claim CLM-2024-001847: Auto Accident, Robert Mitchell, $14,050.33, SETTLED",
        expected={
            'expected_claims': ['CLM-2024-001847'],
            'ground_truth': 'Auto Accident, Robert Mitchell'
        },
        actual_agent='structured'
    )
    print(f"  Score: {result.score:.2f}, Passed: {result.passed}")
    print(f"  Checks: {result.checks}")

    # Test needle grader
    print("\nTesting NEEDLE grader:")
    result = graders.grade(
        query="What was the towing cost?",
        response="The towing cost was $185.00 (Tow Invoice #T-8827)",
        expected={
            'expected_claims': ['CLM-2024-001847'],
            'ground_truth': '$185.00 (Tow Invoice #T-8827)'
        },
        actual_agent='needle'
    )
    print(f"  Score: {result.score:.2f}, Passed: {result.passed}")
    print(f"  Checks: {result.checks}")

    # Test router grader
    print("\nTesting ROUTER grader:")
    result = graders.grade_routing(
        query="Count all claims",
        expected_agent='structured',
        actual_agent='structured'
    )
    print(f"  Score: {result.score:.2f}, Passed: {result.passed}")
    print(f"  Checks: {result.checks}")
