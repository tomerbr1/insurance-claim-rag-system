"""
Test Data Module - Query definitions for evaluation and testing.

This module contains the canonical test query definitions used by:
- Evaluation suite (main.py --eval, --graders)
- Code graders (src/graders/)
- Pytest tests (tests/)

Query Categories:
- STRUCTURED: SQL-based lookups, filters, aggregations
- SUMMARY: High-level overviews, timelines, narratives
- NEEDLE: Precise fact extraction, specific numbers, references
- ROUTER: Edge cases for routing accuracy evaluation

Note: This module lives in src/ (not tests/) because it's used by
production code (evaluation, graders). Tests import from here.
"""

from typing import List, Dict, Any


# =============================================================================
# STRUCTURED QUERIES (SQL Agent)
# Tests: exact lookups, filters, aggregations, date ranges, status checks
# =============================================================================
STRUCTURED_QUERIES: List[Dict[str, Any]] = [
    {
        "query": "Get claim CLM-2024-001847",
        "ground_truth": "Auto Accident, Robert J. Mitchell, $14,050.33, SETTLED",
        "expected_claims": ["CLM-2024-001847"]
    },
    {
        "query": "Show me all claims over $100,000",
        "ground_truth": "CLM-2024-003012, CLM-2024-004583, CLM-2024-004891",
        "expected_claims": ["CLM-2024-003012", "CLM-2024-004583", "CLM-2024-004891"]
    },
    {
        "query": "Which claims are still open?",
        "ground_truth": "Claims with status OPEN",
        "expected_claims": []
    },
    {
        "query": "What is the average claim value?",
        "ground_truth": "Average across all claims",
        "expected_claims": []
    },
    {
        "query": "List all auto-related claims",
        "ground_truth": "CLM-2024-001847 (Auto Accident), CLM-2024-003458 (Auto - Total Loss), CLM-2024-006002 (Auto - Multi-Vehicle Collision)",
        "expected_claims": ["CLM-2024-001847", "CLM-2024-003458", "CLM-2024-006002"]
    },
    {
        "query": "Count how many claims are settled",
        "ground_truth": "Count of claims with status SETTLED",
        "expected_claims": []
    },
    {
        "query": "What is the total value of all claims?",
        "ground_truth": "Sum of all claim values",
        "expected_claims": []
    },
    {
        "query": "Show claims filed in October 2024",
        "ground_truth": "Claims with filing_date in October 2024",
        "expected_claims": []
    },
    {
        "query": "Which claim has the highest value?",
        "ground_truth": "CLM-2024-004891 ($2,800,000.00)",
        "expected_claims": ["CLM-2024-004891"]
    },
    {
        "query": "Get all claims between $20,000 and $50,000",
        "ground_truth": "CLM-2024-003891 ($21,762.99), CLM-2024-002156 ($23,450), CLM-2024-003458 ($24,655), CLM-2024-006001 ($26,700), CLM-2024-002589 ($29,055), CLM-2024-006002 ($46,170)",
        "expected_claims": ["CLM-2024-003891", "CLM-2024-002156", "CLM-2024-003458", "CLM-2024-006001", "CLM-2024-002589", "CLM-2024-006002"]
    },
    {
        "query": "List claims by claimant name containing 'Mitchell'",
        "ground_truth": "CLM-2024-001847 (Robert J. Mitchell)",
        "expected_claims": ["CLM-2024-001847"]
    },
    {
        "query": "How many different claim types are there?",
        "ground_truth": "Count of distinct claim types",
        "expected_claims": []
    },
]


# =============================================================================
# SUMMARY QUERIES (Summary RAG Agent)
# Tests: high-level overviews, timelines, multi-section understanding
# =============================================================================
SUMMARY_QUERIES: List[Dict[str, Any]] = [
    {
        "query": "What happened in claim CLM-2024-003012?",
        "ground_truth": "Slip and fall at Sunny Days Cafe. Patricia Vaughn fell due to coffee spill, required hip replacement surgery. Settled for $142,500.",
        "expected_claims": ["CLM-2024-003012"]
    },
    {
        "query": "Summarize Daniel Harrison's auto collision claim (CLM-2024-006002)",
        "ground_truth": "CLM-2024-006002: Daniel Harrison multi-vehicle collision on Highway 101, BMW totaled, settlement for $46,170, subrogation against at-fault driver",
        "expected_claims": ["CLM-2024-006002"]
    },
    {
        "query": "Summarize the water damage claim",
        "ground_truth": "CLM-2024-002156: Jennifer & Thomas Blackwood, water damage from pipe failure, $22,450.00",
        "expected_claims": ["CLM-2024-002156"]
    },
    {
        "query": "What is the timeline of James Rodriguez's workers compensation claim (CLM-2024-004127)?",
        "ground_truth": "James Rodriguez injury on August 5, 2024, treatment including surgery, TTD benefits, return to modified duty",
        "expected_claims": []
    },
    {
        "query": "Explain what led to the slip and fall incident",
        "ground_truth": "Coffee spill at Sunny Days Cafe was on the floor for 8 minutes before Patricia Vaughn slipped",
        "expected_claims": ["CLM-2024-003012"]
    },
    {
        "query": "Describe the circumstances of Robert Mitchell's auto accident (CLM-2024-001847)",
        "ground_truth": "Robert Mitchell vehicle accident at Oak Street and Main Avenue in Springfield, third-party driver ran red light, settlement for $14,050.33",
        "expected_claims": []
    },
    {
        "query": "What's the story behind the life insurance claim?",
        "ground_truth": "Life insurance payout of $500,547.95, wire transfer settlement to Linda Harrison",
        "expected_claims": []
    },
    {
        "query": "Give me an overview of the theft claim",
        "ground_truth": "Theft claim for Gregory and Susan Palmer, including Gold Rolex Submariner valued at $38,500, settlement for $21,762.99",
        "expected_claims": []
    },
    {
        "query": "Summarize the appendectomy surgery claim (CLM-2024-005234)",
        "ground_truth": "CLM-2024-005234: Dr. and Mrs. Foster's trip cancellation due to son Kevin's appendectomy at Stanford Medical Center, surgery by Dr. Michael Torres",
        "expected_claims": ["CLM-2024-005234"]
    },
    {
        "query": "What happened with the DataCore project claim?",
        "ground_truth": "CLM-2024-004891: Professional liability claim for $2,800,000 related to SAP S/4HANA implementation issues by DataCore Consulting for Vertex Technologies",
        "expected_claims": ["CLM-2024-004891"]
    },
    {
        "query": "Describe the property damage from the Blackwood pipe burst (CLM-2024-002156)",
        "ground_truth": "CLM-2024-002156: Water damage from pipe burst in second-floor bathroom at Jennifer & Thomas Blackwood's property, $23,450 total claim",
        "expected_claims": ["CLM-2024-002156"]
    },
]


# =============================================================================
# NEEDLE QUERIES (Needle RAG Agent)
# Tests: precise facts, specific numbers, reference IDs, quotes
# =============================================================================
NEEDLE_QUERIES: List[Dict[str, Any]] = [
    {
        "query": "What was the exact towing cost in claim CLM-2024-001847?",
        "ground_truth": "$185.00 (Tow Invoice #T-8827)",
        "expected_claims": ["CLM-2024-001847"]
    },
    {
        "query": "How long was the coffee spill on the floor before the slip and fall incident?",
        "ground_truth": "8 minutes",
        "expected_claims": ["CLM-2024-003012"]
    },
    {
        "query": "What is the wire transfer reference number for the life insurance payout?",
        "ground_truth": "WPT-2024-889234",
        "expected_claims": ["CLM-2024-004583"]
    },
    {
        "query": "What was the impairment rating in the workers comp claim?",
        "ground_truth": "8% left upper extremity",
        "expected_claims": ["CLM-2024-004127"]
    },
    {
        "query": "Who signed off on the UAT for the DataCore project?",
        "ground_truth": "Tom Henderson",
        "expected_claims": ["CLM-2024-004891"]
    },
    {
        "query": "What time did the appendectomy surgery start?",
        "ground_truth": "8:30 AM on October 28, 2024",
        "expected_claims": ["CLM-2024-005234"]
    },
    {
        "query": "What was Officer Daniel Thompson's badge number from the Springfield police report (CLM-2024-001847)?",
        "ground_truth": "#4421",
        "expected_claims": ["CLM-2024-001847"]
    },
    {
        "query": "When did ServiceMaster arrive for the water damage?",
        "ground_truth": "8:45 PM",
        "expected_claims": ["CLM-2024-002156"]
    },
    {
        "query": "What was the Copart salvage auction winning bid for Michelle Torres' totaled RAV4?",
        "ground_truth": "$4,200.00",
        "expected_claims": ["CLM-2024-003458"]
    },
    {
        "query": "What is the value of the Gold Rolex Submariner in the theft claim?",
        "ground_truth": "$38,500.00",
        "expected_claims": ["CLM-2024-003891"]
    },
    {
        "query": "What was the deductible amount in the water damage claim?",
        "ground_truth": "$1,000.00",
        "expected_claims": ["CLM-2024-002156"]
    },
    {
        "query": "What is the policy number for the slip and fall claim?",
        "ground_truth": "CGL-2234789",
        "expected_claims": ["CLM-2024-003012"]
    },
    {
        "query": "What was the name of the surgeon who performed the appendectomy?",
        "ground_truth": "Dr. Michael Torres",
        "expected_claims": ["CLM-2024-005234"]
    },
    {
        "query": "What was the reference number of the stolen Rolex?",
        "ground_truth": "116618LB",
        "expected_claims": ["CLM-2024-003891"]
    },
    {
        "query": "What was the emergency mitigation cost in the water damage claim?",
        "ground_truth": "$4,200.00",
        "expected_claims": ["CLM-2024-002156"]
    },
    {
        "query": "What date was James Rodriguez's workers comp injury reported?",
        "ground_truth": "August 5, 2024",
        "expected_claims": ["CLM-2024-004127"]
    },
    {
        "query": "What was the VIN of the totaled vehicle?",
        "ground_truth": "2T3RFREV8KW024891",
        "expected_claims": ["CLM-2024-003458"]
    },
    # Additional needle queries for comprehensive coverage (8 new)
    {
        "query": "What is Dr. William Foster's NPI number in the knee surgery claim?",
        "ground_truth": "1234567890",
        "expected_claims": ["CLM-2024-002589"]
    },
    {
        "query": "What was the pre-authorization number for David Chen's surgery?",
        "ground_truth": "PA-2024-88472",
        "expected_claims": ["CLM-2024-002589"]
    },
    {
        "query": "What was adjuster Rebecca Martinez's ID number in the Thompson storm damage claim?",
        "ground_truth": "ADJ-5587",
        "expected_claims": ["CLM-2024-006001"]
    },
    {
        "query": "What check number was used to settle the Thompson storm claim?",
        "ground_truth": "Check #48827",
        "expected_claims": ["CLM-2024-006001"]
    },
    {
        "query": "What is the CHP report number for the Highway 101 collision?",
        "ground_truth": "2024-CA-87234",
        "expected_claims": ["CLM-2024-006002"]
    },
    {
        "query": "What is the VIN of Daniel Harrison's BMW?",
        "ground_truth": "5UXCR6C05N9K78234",
        "expected_claims": ["CLM-2024-006002"]
    },
    {
        "query": "What is the OSHA incident report number for the forklift accident?",
        "ground_truth": "OSHA-2024-CA-44821",
        "expected_claims": ["CLM-2024-006003"]
    },
    {
        "query": "What is the weekly TTD rate for Christopher Martinez's workers comp claim?",
        "ground_truth": "$920.00/week",
        "expected_claims": ["CLM-2024-006003"]
    },
]


# =============================================================================
# ROUTER EDGE CASES
# Tests routing accuracy with ambiguous or tricky queries
# =============================================================================
ROUTER_QUERIES: List[Dict[str, Any]] = [
    # Clear structured - aggregation keywords
    {
        "query": "Count all claims",
        "expected_agent": "structured",
        "routing_rationale": "Aggregation query (count) - clear structured signal"
    },
    # Clear summary - narrative keywords
    {
        "query": "Tell me what happened in the cafe incident",
        "expected_agent": "summary",
        "routing_rationale": "Narrative question (what happened) - clear summary signal"
    },
    # Clear needle - exact value request
    {
        "query": "What's the exact invoice number for the towing service?",
        "expected_agent": "needle",
        "routing_rationale": "Exact reference number request - clear needle signal"
    },
    # Ambiguous: could be structured (lookup) or summary (overview)
    {
        "query": "Tell me about claim CLM-2024-001847",
        "expected_agent": "summary",
        "routing_rationale": "Open-ended 'tell me about' suggests narrative, not just data lookup"
    },
    # Ambiguous: amount could be structured (filter) or needle (exact value)
    {
        "query": "How much was the settlement?",
        "expected_agent": "needle",
        "routing_rationale": "Asking for specific amount from document text, not filtering"
    },
    # Ambiguous: 'what' could be summary or needle depending on context
    {
        "query": "What caused the accident?",
        "expected_agent": "summary",
        "routing_rationale": "Causation question requires narrative understanding"
    },
    # Edge case: multiple keywords
    {
        "query": "List all the exact costs in the water damage claim",
        "expected_agent": "needle",
        "routing_rationale": "'exact costs' signals needle despite 'list' which suggests structured"
    },
    # Edge case: temporal question
    {
        "query": "When was the claim filed?",
        "expected_agent": "structured",
        "routing_rationale": "Filing date is structured metadata, not buried in document"
    },
    # Edge case: comparison query
    {
        "query": "Compare the auto claims",
        "expected_agent": "summary",
        "routing_rationale": "Comparison requires narrative synthesis across claims"
    },
    # Edge case: status + detail
    {
        "query": "Is the slip and fall claim settled and what was the payout?",
        "expected_agent": "summary",
        "routing_rationale": "Combination query best served by summary with context"
    },
]


# =============================================================================
# HELPER FUNCTIONS
# =============================================================================

def get_all_test_queries() -> List[Dict[str, Any]]:
    """
    Get all test queries combined with expected agent type.

    Returns:
        List of query dicts with 'expected_agent' field added
    """
    all_queries = []

    for q in STRUCTURED_QUERIES:
        q_copy = q.copy()
        q_copy['expected_agent'] = 'structured'
        all_queries.append(q_copy)

    for q in SUMMARY_QUERIES:
        q_copy = q.copy()
        q_copy['expected_agent'] = 'summary'
        all_queries.append(q_copy)

    for q in NEEDLE_QUERIES:
        q_copy = q.copy()
        q_copy['expected_agent'] = 'needle'
        all_queries.append(q_copy)

    return all_queries


def get_router_test_queries() -> List[Dict[str, Any]]:
    """Get router edge case queries for routing accuracy evaluation."""
    return ROUTER_QUERIES.copy()


def get_query_stats() -> Dict[str, int]:
    """Get statistics about test query distribution."""
    return {
        'structured': len(STRUCTURED_QUERIES),
        'summary': len(SUMMARY_QUERIES),
        'needle': len(NEEDLE_QUERIES),
        'router': len(ROUTER_QUERIES),
        'total': len(STRUCTURED_QUERIES) + len(SUMMARY_QUERIES) + len(NEEDLE_QUERIES) + len(ROUTER_QUERIES)
    }


# =============================================================================
# CLI for quick reference
# =============================================================================

def print_test_queries():
    """Print all test queries for reference."""
    print("=" * 70)
    print("TEST QUERIES FOR INSURANCE CLAIMS RAG SYSTEM")
    print("=" * 70)

    print("\n STRUCTURED QUERIES (SQL):")
    for i, q in enumerate(STRUCTURED_QUERIES, 1):
        print(f"  {i}. {q['query']}")

    print("\n SUMMARY QUERIES (RAG):")
    for i, q in enumerate(SUMMARY_QUERIES, 1):
        print(f"  {i}. {q['query']}")

    print("\n NEEDLE QUERIES (Precise RAG):")
    for i, q in enumerate(NEEDLE_QUERIES, 1):
        print(f"  {i}. {q['query']}")

    print("\n ROUTER EDGE CASES:")
    for i, q in enumerate(ROUTER_QUERIES, 1):
        print(f"  {i}. {q['query']} -> {q['expected_agent']}")

    stats = get_query_stats()
    print(f"\nTotal: {stats['total']} queries")
    print(f"  Structured: {stats['structured']}, Summary: {stats['summary']}, "
          f"Needle: {stats['needle']}, Router: {stats['router']}")
    print("=" * 70)


if __name__ == "__main__":
    print_test_queries()
