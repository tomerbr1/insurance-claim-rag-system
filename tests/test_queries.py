"""
Test Queries Module - Sample queries for testing the RAG system.

This module contains:
1. Test queries organized by type (structured/summary/needle)
2. Ground truth answers for evaluation
3. Helper functions for running quick tests
"""

from typing import List, Dict


# Structured queries (should route to SQL agent)
STRUCTURED_QUERIES = [
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
        "ground_truth": "CLM-2024-001847 and CLM-2024-003458",
        "expected_claims": ["CLM-2024-001847", "CLM-2024-003458"]
    }
]

# Summary queries (should route to Summary RAG agent)
SUMMARY_QUERIES = [
    {
        "query": "What happened in claim CLM-2024-003012?",
        "ground_truth": "Slip and fall at Sunny Days Cafe. Patricia Vaughn fell due to coffee spill, required hip replacement surgery. Settled for $142,500.",
        "expected_claims": ["CLM-2024-003012"]
    },
    {
        "query": "Give me an overview of all auto-related claims",
        "ground_truth": "Two auto claims: CLM-2024-001847 (Robert Mitchell, $14,050.33) and CLM-2024-003458 (Michelle Torres, $24,255.00)",
        "expected_claims": ["CLM-2024-001847", "CLM-2024-003458"]
    },
    {
        "query": "Summarize the water damage claim",
        "ground_truth": "CLM-2024-002156: Jennifer & Thomas Blackwood, water damage from pipe failure, $22,450.00",
        "expected_claims": ["CLM-2024-002156"]
    },
    {
        "query": "What is the timeline of the workers compensation claim?",
        "ground_truth": "CLM-2024-004127: James Rodriguez injury, treatment timeline, settlement",
        "expected_claims": ["CLM-2024-004127"]
    }
]

# Needle queries (should route to Needle RAG agent)
NEEDLE_QUERIES = [
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
        "ground_truth": "Tom Henderson on March 28, 2024",
        "expected_claims": ["CLM-2024-004891"]
    },
    {
        "query": "What time did the appendectomy surgery start?",
        "ground_truth": "8:30 AM on October 28, 2024",
        "expected_claims": ["CLM-2024-005234"]
    },
    {
        "query": "What was Officer Thompson's badge number?",
        "ground_truth": "Badge #4421",
        "expected_claims": ["CLM-2024-001847"]
    },
    {
        "query": "When did ServiceMaster arrive for the water damage?",
        "ground_truth": "8:45 PM",
        "expected_claims": ["CLM-2024-002156"]
    },
    {
        "query": "What was the salvage winning bid amount?",
        "ground_truth": "$4,200.00 by JM Auto Parts, Miami FL",
        "expected_claims": ["CLM-2024-003458"]
    },
    {
        "query": "What is the value of the Gold Rolex Submariner in the theft claim?",
        "ground_truth": "$38,500.00 (ref. 116618LB)",
        "expected_claims": ["CLM-2024-003891"]
    }
]


def get_all_test_queries() -> List[Dict]:
    """Get all test queries combined."""
    all_queries = []
    
    for q in STRUCTURED_QUERIES:
        q['expected_agent'] = 'structured'
        all_queries.append(q)
    
    for q in SUMMARY_QUERIES:
        q['expected_agent'] = 'summary'
        all_queries.append(q)
    
    for q in NEEDLE_QUERIES:
        q['expected_agent'] = 'needle'
        all_queries.append(q)
    
    return all_queries


def print_test_queries():
    """Print all test queries for reference."""
    print("=" * 70)
    print("TEST QUERIES FOR INSURANCE CLAIMS RAG SYSTEM")
    print("=" * 70)
    
    print("\n📊 STRUCTURED QUERIES (SQL):")
    for i, q in enumerate(STRUCTURED_QUERIES, 1):
        print(f"  {i}. {q['query']}")
    
    print("\n📝 SUMMARY QUERIES (RAG):")
    for i, q in enumerate(SUMMARY_QUERIES, 1):
        print(f"  {i}. {q['query']}")
    
    print("\n🔍 NEEDLE QUERIES (Precise RAG):")
    for i, q in enumerate(NEEDLE_QUERIES, 1):
        print(f"  {i}. {q['query']}")
    
    total = len(STRUCTURED_QUERIES) + len(SUMMARY_QUERIES) + len(NEEDLE_QUERIES)
    print(f"\nTotal: {total} test queries")
    print("=" * 70)


if __name__ == "__main__":
    print_test_queries()

