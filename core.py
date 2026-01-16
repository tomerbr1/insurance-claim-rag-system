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

from rich.console import Console
from rich.panel import Panel
from rich.table import Table
from rich.text import Text
from rich import box

from src.utils.nltk_silencer import silence_nltk_downloads

# Initialize Rich console for colored output
console = Console()

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


def _count_documents_with_tables(chroma_dir: Path) -> tuple:
    """
    Count documents with tables from the docstore.

    Returns:
        Tuple of (docs_with_tables, total_docs, total_tables)
    """
    docstore_path = chroma_dir / "docstore.json"
    if not docstore_path.exists():
        return 0, 0, 0

    try:
        with open(docstore_path, 'r') as f:
            docstore_data = json.load(f)

        nodes = docstore_data.get('nodes', {})

        # Track unique documents (by source_file) with tables
        docs_seen = set()
        docs_with_tables = set()
        total_tables = 0

        for node_id, node_data in nodes.items():
            metadata = node_data.get('metadata', {})
            source_file = metadata.get('source_file') or metadata.get('file_name')

            if source_file:
                docs_seen.add(source_file)
                if metadata.get('has_tables'):
                    docs_with_tables.add(source_file)
                    # Count tables only once per document (use table_count from first occurrence)
                    if source_file not in docs_with_tables or total_tables == 0:
                        total_tables += metadata.get('table_count', 0)

        # Recalculate total tables correctly
        total_tables = 0
        for source_file in docs_with_tables:
            # Find first node with this source file to get table count
            for node_data in nodes.values():
                metadata = node_data.get('metadata', {})
                node_source = metadata.get('source_file') or metadata.get('file_name')
                if node_source == source_file and metadata.get('has_tables'):
                    total_tables += metadata.get('table_count', 0)
                    break

        return len(docs_with_tables), len(docs_seen), total_tables
    except Exception as e:
        logger.warning(f"Error counting documents with tables: {e}")
        return 0, 0, 0


def display_data_summary():
    """
    Display a summary of existing data including timestamps and statistics.

    Called during instant startup to show users what data is loaded.
    Uses Rich for colorful, user-friendly output.
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

    # Get documents with tables count
    docs_with_tables, total_docs, total_tables = _count_documents_with_tables(CHROMA_DIR)

    # Build the display using Rich
    text = Text()

    # Header
    text.append("\n")

    # Timestamp
    text.append("🕐 ", style="bold")
    text.append("Data Created: ", style="white")
    text.append(f"{data_created}\n\n", style="cyan")

    # Claims section
    text.append("📋 ", style="bold")
    text.append("CLAIMS\n", style="bold yellow")

    if stats:
        text.append(f"   Total: ", style="white")
        text.append(f"{stats['total_claims']} documents\n", style="bold green")

        # Status with colored badges
        if stats.get('by_status'):
            text.append("   Status: ", style="white")
            status_colors = {'SETTLED': 'green', 'OPEN': 'yellow', 'CLOSED': 'red'}
            first = True
            for status, count in sorted(stats['by_status'].items()):
                if not first:
                    text.append(" • ", style="white")
                first = False
                color = status_colors.get(status, 'white')
                text.append(f"{status}", style=f"bold {color}")
                text.append(f" ({count})", style="white")
            text.append("\n")

        # Financial summary
        if stats.get('value_stats') and stats['value_stats'].get('total'):
            total_val = stats['value_stats']['total']
            min_val = stats['value_stats'].get('min', 0)
            max_val = stats['value_stats'].get('max', 0)
            text.append("\n💰 ", style="bold")
            text.append("FINANCIAL\n", style="bold yellow")
            text.append("   Total Value: ", style="white")
            text.append(f"${total_val:,.2f}\n", style="bold green")
            text.append("   Range: ", style="white")
            text.append(f"${min_val:,.2f}", style="cyan")
            text.append(" → ", style="white")
            text.append(f"${max_val:,.2f}\n", style="cyan")
    else:
        text.append(f"   Total: {data_info.get('metadata_count', 0)} documents\n", style="white")

    # Index section
    text.append("\n📁 ", style="bold")
    text.append("INDEXES\n", style="bold yellow")

    chromadb_count = data_info.get('chromadb_count', 0)
    summary_count = data_info.get('summary_count', 0)
    docstore_nodes = data_info.get('details', {}).get('docstore_nodes', 0)

    text.append("   Vector Store: ", style="white")
    text.append(f"{chromadb_count:,} chunks\n", style="cyan")
    text.append("   Summary Nodes: ", style="white")
    text.append(f"{summary_count:,} summaries\n", style="cyan")
    text.append("   Docstore: ", style="white")
    text.append(f"{docstore_nodes:,} hierarchical nodes\n", style="cyan")

    # Tables section
    if total_docs > 0:
        text.append("\n📊 ", style="bold")
        text.append("TABLES\n", style="bold yellow")
        text.append("   Documents with tables: ", style="white")
        if docs_with_tables > 0:
            text.append(f"{docs_with_tables}/{total_docs}", style="bold magenta")
            text.append(f" ({total_tables} tables extracted)\n", style="white")
        else:
            text.append(f"0/{total_docs}\n", style="white")

    # Create the panel
    panel = Panel(
        text,
        title="[bold cyan]📦 EXISTING DATA SUMMARY[/bold cyan]",
        border_style="cyan",
        box=box.ROUNDED,
        padding=(0, 2),
    )

    # Print newline first to ensure panel starts on a clean line
    console.print()
    console.print(panel)


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


def generate_report_from_saved(output_html: bool = True):
    """
    Generate evaluation report from saved data (snapshot-based).

    Uses:
    - responses_to_grade.json (query, response, agents)
    - model_grades table (LLM evaluation scores)
    - human_grades table (manual grades)

    Re-runs code graders (deterministic) but does NOT re-query agents.
    This ensures human grades match the exact responses they were graded on.

    Returns:
        EvaluationReport or None if no saved data found.
    """
    from src.graders.combined_grader import CombinedGrader, CombinedGradeResult, ModelGradeResult
    from src.graders.human_graders import HumanGraderStore
    from src.graders.code_graders import InsuranceClaimsCodeGraders
    from src.graders.report_generator import generate_html_report
    from src.graders.combined_grader import print_report_summary

    responses_path = Path("eval_runs/responses_to_grade.json")
    if not responses_path.exists():
        print("❌ No saved responses found. Run evaluation first.")
        print("   Use: python main.py --graders")
        return None

    with open(responses_path) as f:
        responses = json.load(f)

    if not responses:
        print("❌ No responses in saved file.")
        return None

    eval_run_id = responses[0].get('eval_run_id', 'unknown')
    print(f"\n📂 Loading {len(responses)} responses from {eval_run_id}")

    # Initialize graders
    code_graders = InsuranceClaimsCodeGraders()
    human_store = HumanGraderStore()
    combined_grader = CombinedGrader(llm_judge=None, include_human=True)

    results = []
    human_applied = 0
    model_found = 0

    for resp in responses:
        # Build expected dict for code grader
        expected = {
            'expected_agent': resp['expected_agent'],
            'ground_truth': resp['ground_truth'],
            'expected_claims': []  # Could extract from ground_truth if needed
        }

        # 1. Code grading (re-run, deterministic)
        code_grade = code_graders.grade(
            resp['query'], resp['response'], expected, resp['actual_agent'], {}
        )

        # 2. Model grading (lookup from DB)
        model_data = human_store.get_model_grade(resp['query'])
        model_grade = None
        if model_data:
            model_grade = ModelGradeResult(
                correctness_score=model_data['correctness'],
                correctness_reasoning="(from saved evaluation)",
                relevancy_score=model_data['relevancy'],
                relevancy_reasoning="(from saved evaluation)",
                recall_score=model_data['recall'],
                recall_reasoning="(from saved evaluation)"
            )
            model_found += 1

        # 3. Human grading (lookup from DB)
        human_grade = human_store.get_human_grade(resp['query'])
        if human_grade:
            human_applied += 1

        # Create result
        result = CombinedGradeResult(
            query=resp['query'],
            response=resp['response'],
            expected_agent=resp['expected_agent'],
            actual_agent=resp['actual_agent'],
            ground_truth=resp['ground_truth'],
            code_grade=code_grade,
            model_grade=model_grade,
            human_grade=human_grade,
            eval_run_id=eval_run_id
        )

        # Calculate consensus using CombinedGrader's method
        result.consensus_score, result.grader_agreement = combined_grader._calculate_consensus(result)
        results.append(result)

    print(f"   ✅ Code grades: {len(results)} (re-calculated)")
    print(f"   ✅ Model grades: {model_found} (from DB)")
    print(f"   ✅ Human grades: {human_applied} (from DB)")

    # Generate report
    print("\n📊 Generating evaluation report...")
    report = combined_grader.generate_report(results, eval_run_id)
    print_report_summary(report)

    if output_html:
        output_dir = Path("eval_runs")
        output_dir.mkdir(exist_ok=True)
        html_path = output_dir / f"{eval_run_id}_with_human_report.html"

        generate_html_report(report, html_path)
        print(f"\n🌐 HTML report: {html_path}")
        print(f"   Open with: open {html_path}")

    return report


def compare_evaluation_reports():
    """
    Interactive comparison of two HTML evaluation reports.

    Lists available reports, lets user select two, compares them,
    and generates a comparison HTML report.
    """
    from src.graders.report_comparison import (
        list_available_reports,
        select_reports_interactive,
        parse_html_report,
        compare_reports,
        generate_comparison_html,
        print_comparison_summary
    )

    # List available reports
    reports = list_available_reports()

    if len(reports) < 2:
        console.print("[yellow]Need at least 2 reports to compare.[/yellow]")
        console.print(f"[dim]Found {len(reports)} report(s) in eval_runs/[/dim]")
        console.print("[dim]Run Multi-Grader Eval first to generate reports.[/dim]\n")
        return

    console.print(f"[dim]Found {len(reports)} reports in eval_runs/[/dim]\n")

    try:
        # Interactive selection
        baseline_path, comparison_path = select_reports_interactive(reports)

        console.print(f"\n[dim]Parsing reports...[/dim]")

        # Parse both reports
        baseline = parse_html_report(baseline_path)
        comparison = parse_html_report(comparison_path)

        console.print(f"[dim]Baseline: {baseline.total_queries} queries[/dim]")
        console.print(f"[dim]Comparison: {comparison.total_queries} queries[/dim]")

        # Compare
        result = compare_reports(baseline, comparison)

        # Print console summary
        print_comparison_summary(result)

        # Generate HTML report
        output_dir = Path("eval_runs_comparisons")
        output_dir.mkdir(exist_ok=True)

        output_path = output_dir / f"comparison_{datetime.now():%Y%m%d_%H%M%S}.html"
        generate_comparison_html(result, output_path)

        console.print(f"\n[green]Comparison report saved: {output_path}[/green]")
        console.print(f"[dim]Open with: open {output_path}[/dim]\n")

    except KeyboardInterrupt:
        console.print("\n[dim]Comparison cancelled.[/dim]\n")
    except FileNotFoundError as e:
        console.print(f"\n[red]Error: {e}[/red]\n")
    except ValueError as e:
        console.print(f"\n[red]Error parsing report: {e}[/red]\n")
    except Exception as e:
        console.print(f"\n[red]Unexpected error: {e}[/red]")
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
        subset: Number of queries PER AGENT TYPE (None = all). Ensures all agents are tested.
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

    # Clear previous grades (new eval run = fresh start)
    from src.graders.human_graders import HumanGraderStore
    grade_store = HumanGraderStore()
    human_count, model_count = grade_store.get_grade_counts()
    if human_count > 0 or model_count > 0:
        print(f"\n🗑️  Clearing previous grades (new eval run):")
        print(f"   Human grades: {human_count}")
        print(f"   Model grades: {model_count}")
        grade_store.clear_all_grades()
        print("   ✅ Cleared")

    # Get test queries
    all_queries = get_all_test_queries()
    router_queries = get_router_test_queries()

    if subset:
        # Apply subset per agent type to ensure all agents are tested
        filtered_queries = []
        for agent_type in ['structured', 'summary', 'needle']:
            agent_queries = [q for q in all_queries if q.get('expected_agent') == agent_type]
            filtered_queries.extend(agent_queries[:subset])
        all_queries = filtered_queries
        router_queries = router_queries[:subset]

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

