#!/usr/bin/env python3
"""
Insurance Claims RAG System - Core Orchestrator

This module contains the core system initialization, evaluation runners,
and orchestration logic. It is imported by main.py (the CLI entry point).

Architecture (Instructor-Approved):
1. Hybrid approach: SQL for structured + RAG for unstructured queries
2. 3-way routing: Structured → Summary → Needle agents
3. LLM-based metadata extraction
4. Multi-level summary index with intermediate summaries
5. MCP integration for ChromaDB access

Note: Use main.py as the entry point, not this file directly.
    python main.py              # Interactive CLI
    python main.py --eval       # Run evaluation
    python main.py --graders    # Multi-grader evaluation
"""

import argparse
import json
import logging
import os
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import List, Dict, Any, Optional, Tuple

from src.utils.nltk_silencer import silence_nltk_downloads

# Silence NLTK download chatter before LlamaIndex initializes tokenizers.
silence_nltk_downloads()

from llama_index.llms.openai import OpenAI

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)
# Silence noisy Chroma telemetry chatter (INFO/ERROR) to keep startup clean.
for chroma_logger_name in [
    "chromadb.telemetry",
    "chromadb.telemetry.product.posthog",
]:
    chroma_logger = logging.getLogger(chroma_logger_name)
    chroma_logger.setLevel(logging.CRITICAL)
    chroma_logger.propagate = False


def display_data_summary():
    """
    Display a summary of existing data including timestamps and statistics.

    Called during instant startup to show users what data is loaded.
    """
    from src.config import CHROMA_DIR, METADATA_DB
    from src.cleanup import check_existing_data
    from src.metadata_store import MetadataStore

    # Get file modification times
    def get_file_mtime(path: Path) -> str:
        """Get human-readable modification time of a file."""
        if path.exists():
            mtime = os.path.getmtime(path)
            return datetime.fromtimestamp(mtime).strftime('%Y-%m-%d %H:%M:%S')
        return "N/A"

    # Get data info from cleanup module
    data_info = check_existing_data()

    # Get timestamps
    metadata_mtime = get_file_mtime(METADATA_DB)
    chroma_db_path = CHROMA_DIR / "chroma.sqlite3"
    chroma_mtime = get_file_mtime(chroma_db_path)
    docstore_path = CHROMA_DIR / "docstore.json"
    docstore_mtime = get_file_mtime(docstore_path)

    # Use the most recent timestamp as "data created" time
    mtimes = [metadata_mtime, chroma_mtime, docstore_mtime]
    valid_mtimes = [m for m in mtimes if m != "N/A"]
    data_created = max(valid_mtimes) if valid_mtimes else "N/A"

    # Get statistics from metadata store
    stats = None
    try:
        metadata_store = MetadataStore()
        stats = metadata_store.get_statistics()
        metadata_store.close()
    except Exception as e:
        logger.warning(f"Could not load statistics: {e}")

    # Display formatted output
    width = 70
    print("\n" + "=" * width)
    print("  EXISTING DATA SUMMARY")
    print("=" * width)

    # Timestamp section
    print(f"\n  Data Created: {data_created}")

    # Contents section
    print("\n  Contents:")

    # Claims info
    if stats:
        print(f"     Claims: {stats['total_claims']} total")

        # By type
        if stats.get('by_type'):
            for claim_type, count in sorted(stats['by_type'].items()):
                print(f"       - {claim_type}: {count}")

        # By status
        if stats.get('by_status'):
            status_parts = [f"{status} ({count})" for status, count in sorted(stats['by_status'].items())]
            print(f"     Status: {', '.join(status_parts)}")

        # Financial summary
        if stats.get('value_stats') and stats['value_stats'].get('total'):
            total_val = stats['value_stats']['total']
            min_val = stats['value_stats'].get('min', 0)
            max_val = stats['value_stats'].get('max', 0)
            print(f"     Total Value: ${total_val:,.2f}")
            print(f"     Range: ${min_val:,.2f} - ${max_val:,.2f}")
    else:
        print(f"     Claims: {data_info.get('metadata_count', 0)} total")

    # Vector store info
    print(f"     Vector Store: {data_info.get('chromadb_count', 0)} chunks indexed")
    print(f"     Summary Nodes: {data_info.get('summary_count', 0)} summaries")

    # Docstore info
    docstore_nodes = data_info.get('details', {}).get('docstore_nodes', 0)
    print(f"     Docstore: {docstore_nodes} hierarchical nodes")

    print("\n" + "=" * width)


def load_existing_system():
    """
    Load the RAG system from existing persisted data.
    
    This is called when the user chooses to reuse existing data,
    enabling instant startup without re-processing documents.
    
    Returns:
        Dictionary with all system components, or None if loading fails
    """
    from src.config import validate_config, OPENAI_API_KEY, OPENAI_MODEL
    from src.metadata_store import MetadataStore
    from src.indexing import load_all_indexes
    from src.agents.structured_agent import create_structured_agent
    from src.agents.summary_agent import create_summary_agent
    from src.agents.needle_agent import create_needle_agent
    from src.agents.router_agent import create_router_agent
    from src.mcp.chromadb_client import ChromaDBMCPClient, create_mcp_tools
    
    print("\n" + "=" * 70)
    print("⚡ LOADING EXISTING DATA - INSTANT STARTUP")
    print("=" * 70)

    # Display data summary
    display_data_summary()

    # Validate configuration
    print("\n🔧 Validating configuration...")
    validate_config()
    print("✅ Configuration valid")
    
    # Validate API keys
    print("\n🔑 Validating API keys...")
    from src.config import validate_api_keys
    try:
        validate_api_keys()
        print("✅ Both OpenAI and Google API keys are valid")
    except ValueError as e:
        print(f"\n{e}")
        raise
    
    # Initialize LLM
    llm = OpenAI(model=OPENAI_MODEL, api_key=OPENAI_API_KEY, temperature=0)
    
    # Load all indexes from persistence
    print("\n📂 Loading persisted indexes...")
    summary_index, hierarchical_index, docstore, summary_nodes = load_all_indexes()
    
    if not all([summary_index, hierarchical_index, docstore]):
        print("\n❌ Failed to load all indexes - falling back to full build")
        return None
    
    print(f"   ✅ Summary index loaded")
    print(f"   ✅ Hierarchical index loaded")
    print(f"   ✅ Docstore loaded ({len(docstore.docs)} nodes)")
    
    # Load metadata store (it already exists on disk)
    print("\n💾 Loading metadata store...")
    metadata_store = MetadataStore()
    stats = metadata_store.get_statistics()
    print(f"   ✅ Loaded {stats['total_claims']} claims from SQLite")
    
    # Create agents with loaded indexes
    print("\n🤖 Creating agents...")
    
    structured_agent = create_structured_agent(metadata_store, llm)
    print("   ✅ Structured agent (SQL queries)")
    
    summary_agent = create_summary_agent(summary_index, llm)
    print("   ✅ Summary agent (high-level RAG)")
    
    needle_agent = create_needle_agent(hierarchical_index, docstore, llm)
    print("   ✅ Needle agent (precise RAG)")
    
    # Initialize MCP client FIRST (needed for router)
    print("\n🔌 Initializing MCP client...")
    mcp_client = ChromaDBMCPClient()
    mcp_client.connect()
    mcp_tools = create_mcp_tools(mcp_client)
    print(f"✅ MCP client connected with {len(mcp_tools)} tools")
    tool_names = [t.metadata.name for t in mcp_tools]
    print(f"   Tools: {', '.join(tool_names)}")

    # Create router with MCP tools for system introspection
    print("\n🔀 Creating router with MCP integration...")
    router = create_router_agent(
        structured_engine=structured_agent,
        summary_engine=summary_agent,
        needle_engine=needle_agent,
        llm=llm,
        verbose=True,
        mcp_tools=mcp_tools  # Router can use MCP tools
    )
    print("✅ Router created (3-way: structured/summary/needle)")
    print("   Router will call MCP tools during query processing")
    
    print("\n" + "=" * 70)
    print("⚡ INSTANT STARTUP COMPLETE!")
    print("   (Loaded from existing data - no re-indexing needed)")
    print("=" * 70)
    
    return {
        'router': router,
        'structured_agent': structured_agent,
        'summary_agent': summary_agent,
        'needle_agent': needle_agent,
        'metadata_store': metadata_store,
        'summary_index': summary_index,
        'hierarchical_index': hierarchical_index,
        'docstore': docstore,
        'mcp_client': mcp_client,
        'mcp_tools': mcp_tools,
        'documents': None  # Not loaded when reusing
    }


def initialize_system(skip_cleanup_prompt: bool = False):
    """
    Initialize the RAG system (wrapper for cli.py compatibility).

    Args:
        skip_cleanup_prompt: If True, skips cleanup prompt and reuses existing data if available

    Returns:
        Dictionary with all system components
    """
    return build_system(skip_cleanup=skip_cleanup_prompt)


def build_system(skip_cleanup: bool = False):
    """
    Build the complete RAG system.
    
    Steps:
    0. Cleanup existing data (ChromaDB, SQLite) for fresh start
       - If user chooses to keep data AND data is complete, load instead of build
    1. Load documents with LLM metadata extraction
    2. Populate metadata store (SQL)
    3. Create hierarchical chunks
    4. Build summary index (with intermediate summaries)
    5. Build hierarchical vector index
    6. Create all agents
    7. Create router
    
    Args:
        skip_cleanup: If True, skip the cleanup step (keep existing data)
    
    Returns:
        Dictionary with all system components
    """
    from src.config import validate_config, OPENAI_API_KEY, OPENAI_MODEL
    from src.data_loader import load_claim_documents, get_documents_summary
    from src.metadata_store import MetadataStore
    from src.chunking import create_hierarchical_nodes
    from src.indexing import build_summary_index, build_hierarchical_index
    from src.agents.structured_agent import create_structured_agent
    from src.agents.summary_agent import create_summary_agent
    from src.agents.needle_agent import create_needle_agent
    from src.agents.router_agent import create_router_agent
    from src.mcp.chromadb_client import ChromaDBMCPClient, create_mcp_tools
    from src.cleanup import cleanup_all, check_existing_data
    
    print("\n" + "=" * 70)
    print("🚀 INSURANCE CLAIMS RAG SYSTEM - INITIALIZATION")
    print("=" * 70)
    
    # Step 0: Cleanup for fresh start (interactive by default)
    cleanup_results = None
    if not skip_cleanup:
        cleanup_results = cleanup_all(verbose=True, interactive=True)
        
        # Check if user chose to keep data AND data can be reused
        if cleanup_results.get('skipped') and cleanup_results.get('can_reuse'):
            print("\n✨ Existing data is complete - loading instead of rebuilding...")
            loaded_system = load_existing_system()
            if loaded_system:
                return loaded_system
            print("\n⚠️  Loading failed - proceeding with full build...")
    else:
        # --no-cleanup flag: check if we can reuse existing data
        print("\n⏭️  Skipping cleanup prompt (--no-cleanup flag set)")
        data_info = check_existing_data()
        if data_info['can_reuse']:
            print("✨ Existing data is complete - loading instead of rebuilding...")
            loaded_system = load_existing_system()
            if loaded_system:
                return loaded_system
            print("\n⚠️  Loading failed - proceeding with full build...")
    
    # Validate configuration
    print("\n🔧 Step 0/7: Validating configuration...")
    validate_config()
    print("✅ Configuration valid")
    
    # Validate API keys with actual API calls
    print("\n🔑 Validating API keys (making test calls)...")
    from src.config import validate_api_keys
    try:
        validate_api_keys()
        print("✅ Both OpenAI and Google API keys are valid")
    except ValueError as e:
        print(f"\n{e}")
        print("\n💡 Tips:")
        print("   • OpenAI: Get key from https://platform.openai.com/api-keys")
        print("   • Google: Get key from https://makersuite.google.com/app/apikey")
        raise
    
    # Initialize LLM
    llm = OpenAI(model=OPENAI_MODEL, api_key=OPENAI_API_KEY, temperature=0)
    
    # Step 1: Load documents
    print("\n📄 Step 1/7: Loading claim documents...")
    documents = load_claim_documents()
    print(f"✅ Loaded {len(documents)} documents")
    print(get_documents_summary(documents))
    
    # Step 2: Populate metadata store
    print("\n💾 Step 2/7: Building metadata store...")
    metadata_store = MetadataStore()
    for doc in documents:
        metadata_store.insert_claim(doc.metadata)
    stats = metadata_store.get_statistics()
    print(f"✅ Stored {stats['total_claims']} claims in SQLite")
    print(f"   By status: {stats['by_status']}")
    
    # Step 3: Create hierarchical chunks
    print("\n✂️  Step 3/7: Creating hierarchical chunks...")
    all_nodes, leaf_nodes, docstore = create_hierarchical_nodes(documents)
    print(f"✅ Created {len(all_nodes)} total nodes, {len(leaf_nodes)} leaf nodes")
    
    # Step 4: Build summary index
    print("\n📚 Step 4/7: Building summary index (MapReduce)...")
    print("   ℹ️  This creates 3 levels of summaries for each claim:")
    print("      • Chunk-level summaries (precise facts)")
    print("      • Section-level summaries (grouped context)")
    print("      • Document-level summaries (full overview)")
    print(f"   ⏳ Processing ~{len([n for n in all_nodes if n.metadata.get('chunk_level') == 'small'])} chunks across all claims...")
    print("      (This may take a few minutes)\n")
    summary_index, summary_nodes = build_summary_index(all_nodes, llm)
    print(f"\n✅ Built summary index with {len(summary_nodes)} summary nodes")
    
    # Step 5: Build hierarchical vector index
    print("\n🔍 Step 5/7: Building hierarchical vector index...")
    hierarchical_index = build_hierarchical_index(leaf_nodes, docstore)
    print("✅ Built hierarchical index in ChromaDB")
    
    # Step 6: Create agents
    print("\n🤖 Step 6/7: Creating agents...")
    
    structured_agent = create_structured_agent(metadata_store, llm)
    print("   ✅ Structured agent (SQL queries)")
    
    summary_agent = create_summary_agent(summary_index, llm)
    print("   ✅ Summary agent (high-level RAG)")
    
    needle_agent = create_needle_agent(hierarchical_index, docstore, llm)
    print("   ✅ Needle agent (precise RAG)")
    
    # Initialize MCP client FIRST (needed for router)
    print("\n🔌 Initializing MCP client...")
    mcp_client = ChromaDBMCPClient()
    mcp_client.connect()
    mcp_tools = create_mcp_tools(mcp_client)
    print(f"✅ MCP client connected with {len(mcp_tools)} tools")
    tool_names = [t.metadata.name for t in mcp_tools]
    print(f"   Tools: {', '.join(tool_names)}")

    # Step 7: Create router with MCP integration
    print("\n🔀 Step 7/7: Creating router with MCP integration...")
    router = create_router_agent(
        structured_engine=structured_agent,
        summary_engine=summary_agent,
        needle_engine=needle_agent,
        llm=llm,
        verbose=True,
        mcp_tools=mcp_tools  # Router can use MCP tools
    )
    print("✅ Router created (3-way: structured/summary/needle)")
    print("   Router will call MCP tools during query processing")
    
    print("\n" + "=" * 70)
    print("✅ SYSTEM INITIALIZATION COMPLETE!")
    print("=" * 70)
    
    return {
        'router': router,
        'structured_agent': structured_agent,
        'summary_agent': summary_agent,
        'needle_agent': needle_agent,
        'metadata_store': metadata_store,
        'summary_index': summary_index,
        'hierarchical_index': hierarchical_index,
        'docstore': docstore,
        'mcp_client': mcp_client,
        'mcp_tools': mcp_tools,
        'documents': documents
    }


def run_interactive(system: dict):
    """
    Run interactive query mode.
    
    Args:
        system: Dictionary with system components
    """
    router = system['router']
    
    def show_menu():
        """Display the main menu."""
        print("\n" + "=" * 70)
        print("🎯 INTERACTIVE MODE - MAIN MENU")
        print("=" * 70)
        print("\nHybrid Architecture Active:")
        print("   • STRUCTURED: Exact lookups, filters, aggregations (SQL)")
        print("   • SUMMARY: High-level overviews, timelines (RAG)")
        print("   • NEEDLE: Precise fact retrieval (RAG + auto-merge)")
        print("   • MCP: Router uses MCP tools for system introspection")
        print("\nAvailable Commands:")
        print("   'query' - Send a query to the system")
        print("   'eval'  - Run evaluation suite")
        print("   'stats' - Show system statistics")
        print("   'mcp'   - Show MCP client status and tools")
        print("   'quit'  - Exit the system")
        print("=" * 70)
    
    # Show menu initially
    show_menu()
    
    while True:
        print("\n" + "-" * 70)
        try:
            command = input("💬 Enter command: ").strip().lower()
        except (EOFError, KeyboardInterrupt):
            break
        
        if not command:
            continue
        
        if command == 'quit':
            break
        
        if command == 'eval':
            run_evaluation_mode(system, delay=2.0)  # Default delay for interactive mode
            show_menu()
            continue
        
        if command == 'stats':
            show_statistics(system)
            show_menu()
            continue

        if command == 'mcp':
            show_mcp_status(system)
            show_menu()
            continue

        if command == 'query':
            # Enter query mode - stay in loop until user types 'back'
            print("\n" + "=" * 70)
            print("🔍 QUERY MODE")
            print("=" * 70)
            print("💡 Enter your queries below. Type 'back' to return to the main menu.")
            print("-" * 70)
            
            while True:
                try:
                    user_query = input("\n💭 Query: ").strip()
                except (EOFError, KeyboardInterrupt):
                    break
                
                if not user_query:
                    continue
                
                # Check if user wants to go back to menu
                if user_query.lower() == 'back':
                    break
                
                # Process query
                try:
                    print(f"\n🔍 Processing query...")
                    response, metadata = router.query_with_metadata(user_query)
                    
                    print(f"\n💡 Answer:")
                    print("-" * 40)
                    print(response)
                    print("-" * 40)
                    
                    print(f"\n🎯 Routed to: {metadata.get('routed_to', 'unknown').upper()}")
                    
                except Exception as e:
                    logger.error(f"Error processing query: {e}")
                    print(f"\n❌ Error: {str(e)}")
            
            # Return to main menu
            show_menu()
            continue
        
        # Unknown command
        print(f"❌ Unknown command: '{command}'")
        print("💡 Available commands: query, eval, stats, mcp, quit")
    
    print("\n👋 Goodbye!")


def run_evaluation_mode(system: dict, subset: int = None, delay: float = 2.0):
    """
    Run the evaluation suite.

    Args:
        system: System components dictionary
        subset: Number of test cases to run (None = all)
        delay: Delay between queries in seconds
    """
    from src.evaluation import run_evaluation, print_evaluation_summary, TEST_CASES, LLMJudge

    test_cases = TEST_CASES[:subset] if subset else TEST_CASES

    print("\n📊 Running evaluation suite...")
    print(f"   Test cases: {len(test_cases)}" + (f" (subset of {len(TEST_CASES)})" if subset else ""))
    print(f"   Delay between queries: {delay}s")
    print("   💡 Tip: Use --eval-delay 5 if you hit rate limits")

    router = system['router']
    judge = LLMJudge()

    results = run_evaluation(router, test_cases, judge, verbose=True, delay_between_queries=delay)
    print_evaluation_summary(results)


# =============================================================================
# MULTI-GRADER EVALUATION HELPERS
# =============================================================================

# Constants for display formatting
DISPLAY_QUERY_LENGTH = 50
ROUTER_SUBSET_DEFAULT = 3


def _initialize_graders(include_human: bool) -> Tuple[Any, Any]:
    """
    Initialize the LLM judge and combined grader.

    Args:
        include_human: Whether to include human grades

    Returns:
        Tuple of (llm_judge, combined_grader)
    """
    from src.evaluation import LLMJudge
    from src.graders.combined_grader import CombinedGrader

    print("\n🔧 Initializing graders...")

    llm_judge = None
    try:
        llm_judge = LLMJudge()
        print("   ✅ Model grader (Gemini LLMJudge)")
    except Exception as e:
        print(f"   ⚠️  Model grader unavailable: {e}")

    combined_grader = CombinedGrader(
        llm_judge=llm_judge,
        include_human=include_human
    )
    print("   ✅ Code grader (agent-specific)")

    if include_human:
        print("   ✅ Human grader (from SQLite)")
    else:
        print("   ⏭️  Human grader (skipped - use --include-human to enable)")

    return llm_judge, combined_grader


def _evaluate_agent_queries(
    router,
    queries: List[Dict],
    grader,
    eval_run_id: str,
    delay: float,
    total_queries: int
) -> List[Any]:
    """
    Evaluate agent queries and collect results.

    Args:
        router: The router agent
        queries: List of query dicts with expected values
        grader: CombinedGrader instance
        eval_run_id: Evaluation run identifier
        delay: Delay between queries in seconds
        total_queries: Total number of queries (for progress display)

    Returns:
        List of CombinedGradeResult objects
    """
    results = []

    for i, test in enumerate(queries, 1):
        print(f"\n[{i}/{total_queries}] {test['query'][:DISPLAY_QUERY_LENGTH]}...")

        try:
            response, metadata = router.query_with_metadata(test['query'])
            actual_agent = metadata.get('routed_to', 'unknown')

            result = grader.grade_single(
                query=test['query'],
                response=response,
                expected={
                    'expected_agent': test.get('expected_agent'),
                    'expected_claims': test.get('expected_claims', []),
                    'ground_truth': test.get('ground_truth', '')
                },
                actual_agent=actual_agent,
                metadata=metadata,
                eval_run_id=eval_run_id
            )
            results.append(result)

            # Display progress
            routing_status = "✅" if result.expected_agent == actual_agent else "❌"
            code_score = result.code_grade.score if result.code_grade else 0
            model_score = result.model_grade.correctness_score if result.model_grade else 0

            print(f"   Routing: {routing_status} ({actual_agent})")
            print(f"   Code: {code_score:.2f} | Model: {model_score:.2f} | Consensus: {result.consensus_score:.2f}")

        except Exception as e:
            logger.error(f"Error evaluating query: {e}")
            print(f"   ❌ Error: {str(e)}")

        if i < total_queries and delay > 0:
            time.sleep(delay)

    return results


def _evaluate_router_queries(
    router,
    queries: List[Dict],
    delay: float,
    start_index: int,
    total_queries: int
) -> None:
    """
    Evaluate router edge case queries (routing accuracy only).

    Args:
        router: The router agent
        queries: List of router edge case query dicts
        delay: Delay between queries in seconds
        start_index: Starting index for progress display
        total_queries: Total number of queries (for progress display)
    """
    print(f"\n🔀 Evaluating router edge cases...")

    for i, test in enumerate(queries, start_index):
        print(f"\n[{i}/{total_queries}] ROUTER: {test['query'][:DISPLAY_QUERY_LENGTH]}...")

        try:
            response, metadata = router.query_with_metadata(test['query'])
            actual_agent = metadata.get('routed_to', 'unknown')
            expected_agent = test.get('expected_agent', '')

            routing_correct = actual_agent == expected_agent
            status = "✅" if routing_correct else "❌"
            print(f"   Expected: {expected_agent} | Actual: {actual_agent} {status}")
            print(f"   Rationale: {test.get('routing_rationale', 'N/A')}")

        except Exception as e:
            print(f"   ❌ Error: {str(e)}")

        if i < total_queries and delay > 0:
            time.sleep(delay)


def _generate_evaluation_report(grader, results: List, eval_run_id: str, output_html: bool):
    """
    Generate and save the evaluation report.

    Args:
        grader: CombinedGrader instance
        results: List of evaluation results
        eval_run_id: Evaluation run identifier
        output_html: Whether to generate HTML report
    """
    from src.graders.combined_grader import print_report_summary
    from src.graders.report_generator import generate_html_report

    print("\n" + "-" * 70)
    print("📊 Generating evaluation report...")

    report = grader.generate_report(results, eval_run_id)
    print_report_summary(report)

    if output_html:
        output_dir = Path("eval_runs")
        output_dir.mkdir(exist_ok=True)
        html_path = output_dir / f"{eval_run_id}_report.html"

        generate_html_report(report, html_path)
        print(f"\n🌐 HTML report: {html_path}")
        print(f"   Open with: open {html_path}")

    # Export responses for human grading
    _export_responses_for_grading(results, eval_run_id)

    return report


def _export_responses_for_grading(results: List, eval_run_id: str):
    """
    Export responses to JSON for human grading workflow.

    Creates eval_runs/responses_to_grade.json with all responses
    from the evaluation run in the format expected by human_graders.py.
    """
    try:
        output_dir = Path("eval_runs")
        output_dir.mkdir(exist_ok=True)
        responses_path = output_dir / "responses_to_grade.json"

        responses = []
        for result in results:
            responses.append({
                "query": result.query,
                "response": result.response,
                "expected_agent": result.expected_agent,
                "actual_agent": result.actual_agent,
                "ground_truth": result.ground_truth,
                "eval_run_id": eval_run_id
            })

        with open(responses_path, 'w') as f:
            json.dump(responses, f, indent=2)

        print(f"\n📝 Responses exported: {responses_path}")
        print(f"   {len(responses)} responses ready for human grading")
        print(f"   Run: python -m src.graders.human_graders grade")
    except Exception as e:
        print(f"\n❌ Error exporting responses: {e}")
        import traceback
        traceback.print_exc()


def run_graders_evaluation(
    system: dict,
    subset: int = None,
    delay: float = 2.0,
    include_human: bool = False,
    output_html: bool = True
):
    """
    Run evaluation with all three grader types (code, model, human).

    This implements the multi-grader evaluation based on Anthropic's
    "Demystifying Evals for AI Agents" article.

    Args:
        system: System components dictionary
        subset: Number of test cases to run (None = all)
        delay: Delay between queries in seconds
        include_human: Whether to include human grades (if available)
        output_html: Whether to generate HTML report

    Returns:
        EvaluationReport with all results
    """
    from src.test_data import get_all_test_queries, get_router_test_queries

    print("\n" + "=" * 70)
    print("🎯 MULTI-GRADER EVALUATION")
    print("   Based on Anthropic's 'Demystifying Evals for AI Agents'")
    print("=" * 70)

    # Get test queries
    all_queries = get_all_test_queries()
    router_queries = get_router_test_queries()

    if subset:
        all_queries = all_queries[:subset]
        router_queries = router_queries[:min(ROUTER_SUBSET_DEFAULT, len(router_queries))]

    total_queries = len(all_queries) + len(router_queries)

    print(f"\n📋 Test Configuration:")
    print(f"   Agent queries: {len(all_queries)}")
    print(f"   Router edge cases: {len(router_queries)}")
    print(f"   Total: {total_queries}")
    print(f"   Delay between queries: {delay}s")

    # Initialize graders
    _, combined_grader = _initialize_graders(include_human)

    # Generate run ID
    eval_run_id = f"eval_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    print(f"\n🚀 Starting evaluation (run ID: {eval_run_id})...")
    print("-" * 70)

    # Evaluate agent queries
    router = system['router']
    results = _evaluate_agent_queries(
        router, all_queries, combined_grader, eval_run_id, delay, total_queries
    )

    # Evaluate router edge cases
    _evaluate_router_queries(
        router, router_queries, delay,
        start_index=len(all_queries) + 1,
        total_queries=total_queries
    )

    # Generate report
    return _generate_evaluation_report(combined_grader, results, eval_run_id, output_html)


def show_mcp_status(system: dict):
    """Show MCP client status and available tools."""
    mcp_client = system['mcp_client']
    mcp_tools = system['mcp_tools']

    print("\n🔌 MCP CLIENT STATUS")
    print("=" * 60)

    # Server info
    server_info = mcp_client.server_info
    print(f"\n📡 Server Information:")
    print(f"   Name: {server_info.get('name', 'unknown')}")
    print(f"   Version: {server_info.get('version', 'unknown')}")
    print(f"   Protocol: {server_info.get('protocolVersion', 'unknown')}")
    print(f"   Status: {server_info.get('status', 'unknown')}")
    print(f"   Mode: direct (using ChromaDB library directly)")

    # Available tools
    print(f"\n🔧 Available MCP Tools ({len(mcp_tools)}):")
    for tool in mcp_tools:
        print(f"   • {tool.metadata.name}: {tool.metadata.description}")

    # Test tool calls
    print(f"\n🧪 Testing MCP Tools:")
    for tool in mcp_tools[:3]:  # Test first 3 tools
        try:
            result = tool.fn()
            display_result = result if len(str(result)) < 60 else str(result)[:57] + "..."
            print(f"   ✅ {tool.metadata.name}(): {display_result}")
        except Exception as e:
            print(f"   ❌ {tool.metadata.name}(): Error - {e}")

    # Router integration status
    router = system['router']
    if hasattr(router, '_mcp_tools') and router._mcp_tools:
        print(f"\n🔀 Router MCP Integration:")
        print(f"   ✅ Router has {len(router._mcp_tools)} MCP tools configured")
        print(f"   ✅ Router will call MCP tools during query processing")
        print(f"   Log tool calls: {'enabled' if router._log_tool_calls else 'disabled'}")
    else:
        print(f"\n🔀 Router MCP Integration:")
        print(f"   ⚠️  Router does not have MCP tools configured")

    print("=" * 60)


def show_statistics(system: dict):
    """Show system statistics."""
    metadata_store = system['metadata_store']
    mcp_client = system['mcp_client']

    print("\n📊 SYSTEM STATISTICS")
    print("=" * 50)
    
    # Metadata store stats
    stats = metadata_store.get_statistics()
    print(f"\n💾 Metadata Store:")
    print(f"   Total claims: {stats['total_claims']}")
    print(f"   By status: {stats['by_status']}")
    print(f"   By type: {stats['by_type']}")
    
    if stats['value_stats']['total']:
        print(f"\n💰 Financial Summary:")
        print(f"   Total value: ${stats['value_stats']['total']:,.2f}")
        print(f"   Average: ${stats['value_stats']['average']:,.2f}")
        print(f"   Min: ${stats['value_stats']['min']:,.2f}")
        print(f"   Max: ${stats['value_stats']['max']:,.2f}")
    
    # Vector store stats
    print(f"\n🔍 Vector Store:")
    collections = mcp_client.list_collections()
    print(f"   Collections: {collections}")
    for name in collections:
        info = mcp_client.get_collection_info(name)
        print(f"   {name}: {info.get('count', 'unknown')} documents")
    
    print("=" * 50)


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="Insurance Claims RAG System",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python main.py                     # Interactive mode
  python main.py --eval              # Run basic evaluation
  python main.py --graders           # Run multi-grader evaluation with HTML report
  python main.py --graders --eval-subset 5  # Run 5 test cases with all graders
  python main.py --eval --eval-delay 5    # 5 second delay between queries
  python main.py --build             # Build indexes and exit
  python main.py --no-cleanup        # Instant startup if data exists

Multi-Grader Evaluation (--graders):
  Based on Anthropic's "Demystifying Evals for AI Agents" article.
  Runs three types of graders:
  - Code-based: Deterministic checks (routing, amounts, references)
  - Model-based: LLM evaluation using Gemini (unbiased provider)
  - Human: Manual grades from SQLite (use --include-human)
  Outputs a beautiful HTML report to eval_runs/

Rate Limit Mitigation:
  If you hit OpenAI rate limits (429 errors), try:
  1. --eval-delay 5        # Increase delay between queries
  2. --eval-subset 3       # Run fewer test cases
  3. Check your OpenAI billing/usage at platform.openai.com
        """
    )
    
    parser.add_argument(
        '--eval', action='store_true',
        help='Run evaluation suite and exit'
    )
    parser.add_argument(
        '--eval-subset', type=int, default=None, metavar='N',
        help='Run only first N test cases (useful for rate limit issues)'
    )
    parser.add_argument(
        '--eval-delay', type=float, default=2.0, metavar='SECONDS',
        help='Delay between queries in seconds (default: 2.0, increase if hitting rate limits)'
    )
    parser.add_argument(
        '--build', action='store_true',
        help='Build indexes only and exit'
    )
    parser.add_argument(
        '--verbose', '-v', action='store_true',
        help='Verbose output'
    )
    parser.add_argument(
        '--no-cleanup', action='store_true',
        help='Skip cleanup prompt; if data exists, enables instant startup by reusing indexes'
    )
    parser.add_argument(
        '--graders', action='store_true',
        help='Run multi-grader evaluation (code + model + human) with HTML report'
    )
    parser.add_argument(
        '--include-human', action='store_true',
        help='Include human grades in evaluation (requires prior manual grading)'
    )

    args = parser.parse_args()
    
    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)
    
    try:
        # Build system
        system = build_system(skip_cleanup=args.no_cleanup)
        
        if args.build:
            print("\n✅ Build complete. Exiting.")
            return 0
        
        if args.graders:
            run_graders_evaluation(
                system,
                subset=args.eval_subset,
                delay=args.eval_delay,
                include_human=args.include_human,
                output_html=True
            )
            return 0

        if args.eval:
            run_evaluation_mode(system, subset=args.eval_subset, delay=args.eval_delay)
            return 0
        
        # Interactive mode
        run_interactive(system)
        return 0
        
    except KeyboardInterrupt:
        print("\n\n👋 Interrupted. Goodbye!")
        return 1
    except Exception as e:
        logger.error(f"Fatal error: {e}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    sys.exit(main())

