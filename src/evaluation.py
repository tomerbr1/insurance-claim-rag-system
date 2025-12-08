"""
Evaluation Module - LLM-as-Judge evaluation system.

This module implements:
1. LLM-based evaluation of system responses using Gemini (different provider)
2. Three evaluation metrics: Correctness, Relevancy, Recall
3. Test suite with ground truth answers
4. Router accuracy evaluation

Evaluation Philosophy:
- Use a DIFFERENT LLM provider (Gemini) to avoid self-evaluation bias
- Compare against ground truth where available
- Score on 0.0-1.0 scale

Why Gemini for Evaluation?
- Different provider = unbiased evaluation (not OpenAI evaluating OpenAI)
- Gemini 2.5 Flash is fast and cost-effective
- Good reasoning capabilities for quality assessment
"""

import logging
from typing import List, Dict, Any, Optional, Tuple
from dataclasses import dataclass

import google.generativeai as genai
from llama_index.llms.openai import OpenAI

from src.config import GOOGLE_API_KEY, GEMINI_EVAL_MODEL, OPENAI_API_KEY, OPENAI_MINI_MODEL

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


# Evaluation prompts
CORRECTNESS_PROMPT = """
You are evaluating an RAG system's answer for CORRECTNESS.

Ground Truth Answer: {ground_truth}

System's Answer: {answer}

Score from 0.0 to 1.0:
- 1.0: Answer is fully correct and matches ground truth
- 0.7-0.9: Answer is mostly correct with minor omissions or extra details
- 0.4-0.6: Answer is partially correct but missing key information
- 0.1-0.3: Answer has some relevant info but is mostly wrong
- 0.0: Answer is completely wrong or irrelevant

Return ONLY a JSON object with this format:
{{"score": 0.X, "reasoning": "brief explanation"}}
"""

RELEVANCY_PROMPT = """
You are evaluating whether the RETRIEVED CONTEXT is relevant to the query.

User Query: {query}

Retrieved Context:
{context}

Score from 0.0 to 1.0:
- 1.0: Context is highly relevant and contains information to answer the query
- 0.7-0.9: Context is mostly relevant with some tangential information
- 0.4-0.6: Context is somewhat relevant but missing key information
- 0.1-0.3: Context is mostly irrelevant to the query
- 0.0: Context is completely irrelevant

Return ONLY a JSON object with this format:
{{"score": 0.X, "reasoning": "brief explanation"}}
"""

RECALL_PROMPT = """
You are evaluating RETRIEVAL RECALL - whether the correct source documents were retrieved.

Expected Source Claims: {expected_claims}

Actually Retrieved From: {retrieved_claims}

Score from 0.0 to 1.0 based on overlap:
- 1.0: All expected claims were retrieved
- 0.5-0.9: Most expected claims were retrieved
- 0.1-0.4: Some expected claims were retrieved
- 0.0: None of the expected claims were retrieved

Return ONLY a JSON object with this format:
{{"score": 0.X, "reasoning": "brief explanation"}}
"""


@dataclass
class TestCase:
    """A single test case for evaluation."""
    query: str
    expected_agent: str  # "structured", "summary", or "needle"
    ground_truth: str
    expected_claims: List[str]  # Claim IDs that should be retrieved
    category: str = "general"  # For grouping results


@dataclass
class EvaluationResult:
    """Result of evaluating a single test case."""
    query: str
    expected_agent: str
    actual_agent: str
    answer: str
    correctness_score: float
    relevancy_score: float
    recall_score: float
    routing_correct: bool
    reasoning: Dict[str, str]


# Test cases based on the data README
TEST_CASES = [
    # Structured queries
    TestCase(
        query="Get claim CLM-2024-001847",
        expected_agent="structured",
        ground_truth="Auto Accident claim by Robert J. Mitchell, total value $14,050.33, status SETTLED",
        expected_claims=["CLM-2024-001847"],
        category="structured_lookup"
    ),
    TestCase(
        query="Show me all claims over $100,000",
        expected_agent="structured",
        ground_truth="CLM-2024-003012 ($142,500), CLM-2024-004583 ($500,547.95), CLM-2024-004891 ($1,550,000)",
        expected_claims=["CLM-2024-003012", "CLM-2024-004583", "CLM-2024-004891"],
        category="structured_filter"
    ),
    TestCase(
        query="Which claims are still open?",
        expected_agent="structured",
        ground_truth="Open claims based on status field",
        expected_claims=[],
        category="structured_status"
    ),
    
    # Summary queries
    TestCase(
        query="What happened in claim CLM-2024-003012?",
        expected_agent="summary",
        ground_truth="Slip and fall incident at Sunny Days Cafe. Patricia Vaughn fell due to coffee spill that was on the floor for 8 minutes. Required hip replacement surgery. Settled for $142,500.",
        expected_claims=["CLM-2024-003012"],
        category="summary_overview"
    ),
    TestCase(
        query="Give me an overview of all auto-related claims",
        expected_agent="summary",
        ground_truth="Two auto claims: CLM-2024-001847 (Robert Mitchell accident, $14,050.33) and CLM-2024-003458 (Michelle Torres total loss, $24,255.00)",
        expected_claims=["CLM-2024-001847", "CLM-2024-003458"],
        category="summary_multi"
    ),
    
    # Needle queries
    TestCase(
        query="What was the exact towing cost in claim CLM-2024-001847?",
        expected_agent="needle",
        ground_truth="$185.00 (Tow Invoice #T-8827)",
        expected_claims=["CLM-2024-001847"],
        category="needle_amount"
    ),
    TestCase(
        query="How long was the coffee spill on the floor before the slip and fall incident?",
        expected_agent="needle",
        ground_truth="8 minutes",
        expected_claims=["CLM-2024-003012"],
        category="needle_duration"
    ),
    TestCase(
        query="What is the wire transfer reference number for the life insurance payout?",
        expected_agent="needle",
        ground_truth="WPT-2024-889234",
        expected_claims=["CLM-2024-004583"],
        category="needle_reference"
    ),
    TestCase(
        query="What was the impairment rating in the workers comp claim?",
        expected_agent="needle",
        ground_truth="8% left upper extremity",
        expected_claims=["CLM-2024-004127"],
        category="needle_medical"
    ),
    TestCase(
        query="Who signed off on the UAT for the DataCore project?",
        expected_agent="needle",
        ground_truth="Tom Henderson on March 28, 2024",
        expected_claims=["CLM-2024-004891"],
        category="needle_person"
    ),
    TestCase(
        query="What time did the appendectomy surgery start?",
        expected_agent="needle",
        ground_truth="8:30 AM on October 28, 2024",
        expected_claims=["CLM-2024-005234"],
        category="needle_time"
    ),
]


class LLMJudge:
    """
    LLM-based judge for evaluating RAG system responses.
    
    Uses Gemini (Google) ONLY as the evaluation model to avoid bias
    from having OpenAI evaluate OpenAI responses.
    
    NO FALLBACK: Both OpenAI and Google API keys are required.
    This ensures unbiased evaluation throughout.
    """
    
    def __init__(self):
        """
        Initialize the judge with Gemini.
        
        Raises:
            ValueError: If Google API key is not set
        """
        if not GOOGLE_API_KEY:
            raise ValueError(
                "Google API key is required for evaluation. "
                "We use Gemini to evaluate OpenAI outputs to avoid bias. "
                "Please set GOOGLE_API_KEY in your .env file."
            )
        
        # Configure Gemini
        genai.configure(api_key=GOOGLE_API_KEY)
        self.gemini_model = genai.GenerativeModel(GEMINI_EVAL_MODEL)
        logger.info(f"✅ Judge initialized with Gemini: {GEMINI_EVAL_MODEL}")
        logger.info("   Using different provider (Google) to avoid OpenAI self-evaluation bias")
    
    def _call_llm(self, prompt: str) -> str:
        """Call Gemini for evaluation."""
        try:
            response = self.gemini_model.generate_content(prompt)
            return response.text
        except Exception as e:
            logger.error(f"Gemini API call failed: {e}")
            raise RuntimeError(
                f"Failed to call Gemini for evaluation: {str(e)}\n"
                "Please check your Google API key and internet connection."
            )
    
    def _parse_score_response(self, response: str) -> Tuple[float, str]:
        """Parse LLM response to extract score and reasoning."""
        import json
        
        try:
            # Clean up response
            response = response.strip()
            if response.startswith("```"):
                lines = response.split("\n")
                # Handle ```json or just ```
                start = 1
                end = -1
                if lines[-1].strip() == "```":
                    end = -1
                response = "\n".join(lines[start:end])
                response = response.strip()
            
            # Try to find JSON in the response
            if "{" in response:
                start_idx = response.find("{")
                end_idx = response.rfind("}") + 1
                response = response[start_idx:end_idx]
            
            result = json.loads(response)
            return float(result.get('score', 0.0)), result.get('reasoning', '')
        except Exception as e:
            logger.warning(f"Failed to parse judge response: {e}")
            logger.debug(f"Response was: {response[:200]}")
            return 0.5, "Could not parse evaluation"
    
    def evaluate_correctness(
        self,
        answer: str,
        ground_truth: str
    ) -> Tuple[float, str]:
        """
        Evaluate answer correctness against ground truth.
        
        Args:
            answer: System's answer
            ground_truth: Expected correct answer
        
        Returns:
            Tuple of (score, reasoning)
        """
        prompt = CORRECTNESS_PROMPT.format(
            ground_truth=ground_truth,
            answer=answer
        )
        
        response = self._call_llm(prompt)
        return self._parse_score_response(response)
    
    def evaluate_relevancy(
        self,
        query: str,
        context: str
    ) -> Tuple[float, str]:
        """
        Evaluate context relevancy to query.
        
        Args:
            query: User's query
            context: Retrieved context
        
        Returns:
            Tuple of (score, reasoning)
        """
        prompt = RELEVANCY_PROMPT.format(
            query=query,
            context=context[:2000]  # Truncate long contexts
        )
        
        response = self._call_llm(prompt)
        return self._parse_score_response(response)
    
    def evaluate_recall(
        self,
        expected_claims: List[str],
        retrieved_claims: List[str]
    ) -> Tuple[float, str]:
        """
        Evaluate retrieval recall.
        
        Args:
            expected_claims: Claims that should be retrieved
            retrieved_claims: Claims actually retrieved
        
        Returns:
            Tuple of (score, reasoning)
        """
        # Simple overlap calculation (no LLM needed for this)
        if not expected_claims:
            return 1.0, "No expected claims specified"
        
        expected_set = set(expected_claims)
        retrieved_set = set(retrieved_claims)
        
        overlap = expected_set.intersection(retrieved_set)
        recall = len(overlap) / len(expected_set)
        
        reasoning = f"Retrieved {len(overlap)}/{len(expected_set)} expected claims"
        return recall, reasoning


def run_evaluation(
    router,
    test_cases: List[TestCase] = None,
    judge: LLMJudge = None,
    verbose: bool = True
) -> List[EvaluationResult]:
    """
    Run full evaluation suite.
    
    Args:
        router: The router agent to evaluate
        test_cases: Test cases to run (default: TEST_CASES)
        judge: LLM judge instance
        verbose: Whether to print progress
    
    Returns:
        List of EvaluationResult objects
    """
    test_cases = test_cases or TEST_CASES
    judge = judge or LLMJudge()
    
    results = []
    
    for i, test in enumerate(test_cases, 1):
        if verbose:
            print(f"\n[{i}/{len(test_cases)}] Testing: {test.query[:50]}...")
        
        try:
            # Run query
            response, metadata = router.query_with_metadata(test.query)
            actual_agent = metadata.get('routed_to', 'unknown')
            
            # Extract retrieved claims from response (simplified)
            # In production, you'd extract from source nodes
            retrieved_claims = []
            for claim_id in ["CLM-2024-001847", "CLM-2024-002156", "CLM-2024-002589",
                            "CLM-2024-003012", "CLM-2024-003458", "CLM-2024-003891",
                            "CLM-2024-004127", "CLM-2024-004583", "CLM-2024-004891",
                            "CLM-2024-005234"]:
                if claim_id in response:
                    retrieved_claims.append(claim_id)
            
            # Evaluate
            correctness_score, correctness_reason = judge.evaluate_correctness(
                response, test.ground_truth
            )
            
            relevancy_score, relevancy_reason = judge.evaluate_relevancy(
                test.query, response
            )
            
            recall_score, recall_reason = judge.evaluate_recall(
                test.expected_claims, retrieved_claims
            )
            
            result = EvaluationResult(
                query=test.query,
                expected_agent=test.expected_agent,
                actual_agent=actual_agent,
                answer=response[:500],
                correctness_score=correctness_score,
                relevancy_score=relevancy_score,
                recall_score=recall_score,
                routing_correct=(actual_agent == test.expected_agent),
                reasoning={
                    'correctness': correctness_reason,
                    'relevancy': relevancy_reason,
                    'recall': recall_reason
                }
            )
            results.append(result)
            
            if verbose:
                routing_status = "✅" if result.routing_correct else "❌"
                print(f"   Routing: {routing_status} ({actual_agent})")
                print(f"   Correctness: {correctness_score:.2f}")
                print(f"   Relevancy: {relevancy_score:.2f}")
                print(f"   Recall: {recall_score:.2f}")
                
        except Exception as e:
            logger.error(f"Error evaluating test case: {e}")
            import traceback
            traceback.print_exc()
    
    return results


def print_evaluation_summary(results: List[EvaluationResult]):
    """Print formatted evaluation summary."""
    print("\n" + "=" * 70)
    print("📊 EVALUATION SUMMARY")
    print("=" * 70)
    
    # Calculate averages
    avg_correctness = sum(r.correctness_score for r in results) / len(results)
    avg_relevancy = sum(r.relevancy_score for r in results) / len(results)
    avg_recall = sum(r.recall_score for r in results) / len(results)
    routing_accuracy = sum(1 for r in results if r.routing_correct) / len(results)
    
    print(f"\n📈 OVERALL SCORES:")
    print(f"   Correctness:      {avg_correctness:.2f}")
    print(f"   Relevancy:        {avg_relevancy:.2f}")
    print(f"   Recall:           {avg_recall:.2f}")
    print(f"   Routing Accuracy: {routing_accuracy:.1%}")
    
    # Group by category
    print(f"\n📋 BY QUERY TYPE:")
    for agent_type in ["structured", "summary", "needle"]:
        type_results = [r for r in results if r.expected_agent == agent_type]
        if type_results:
            avg = sum(r.correctness_score for r in type_results) / len(type_results)
            routing = sum(1 for r in type_results if r.routing_correct) / len(type_results)
            print(f"\n   {agent_type.upper()}:")
            print(f"      Avg Correctness: {avg:.2f}")
            print(f"      Routing Accuracy: {routing:.1%}")
    
    # Low scores
    print(f"\n⚠️  LOW SCORING QUERIES (< 0.7):")
    for r in results:
        if r.correctness_score < 0.7:
            print(f"   - {r.query[:50]}... (score: {r.correctness_score:.2f})")
    
    print("\n" + "=" * 70)


# Quick test
if __name__ == "__main__":
    print("Evaluation module loaded successfully")
    print(f"Loaded {len(TEST_CASES)} test cases")
    print("\nTest case categories:")
    categories = set(t.category for t in TEST_CASES)
    for cat in categories:
        count = len([t for t in TEST_CASES if t.category == cat])
        print(f"  - {cat}: {count}")

