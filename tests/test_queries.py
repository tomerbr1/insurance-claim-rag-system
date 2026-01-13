"""
Test Queries Module - Re-exports from src/test_data.py for pytest convenience.

This module re-exports the canonical test query definitions from src/test_data.py.
The actual definitions live in src/ because they're used by production code
(evaluation suite, graders). This file provides a convenient import path for tests.

Usage in tests:
    from tests.test_queries import STRUCTURED_QUERIES, get_all_test_queries
    # or
    from src.test_data import STRUCTURED_QUERIES, get_all_test_queries
"""

# Re-export everything from the canonical source
from src.test_data import (
    # Query lists
    STRUCTURED_QUERIES,
    SUMMARY_QUERIES,
    NEEDLE_QUERIES,
    ROUTER_QUERIES,
    # Helper functions
    get_all_test_queries,
    get_router_test_queries,
    get_query_stats,
    print_test_queries,
)

# For backward compatibility, expose __all__
__all__ = [
    'STRUCTURED_QUERIES',
    'SUMMARY_QUERIES',
    'NEEDLE_QUERIES',
    'ROUTER_QUERIES',
    'get_all_test_queries',
    'get_router_test_queries',
    'get_query_stats',
    'print_test_queries',
]


if __name__ == "__main__":
    print_test_queries()
    print("\nQuery Statistics:")
    for agent, count in get_query_stats().items():
        print(f"  {agent}: {count}")
