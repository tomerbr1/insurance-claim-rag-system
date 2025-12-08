#!/usr/bin/env python3
"""
Insurance Claims RAG System - Main Orchestrator

This is the entry point for the hybrid insurance claims RAG system.

Architecture (Instructor-Approved):
1. Hybrid approach: SQL for structured + RAG for unstructured queries
2. 3-way routing: Structured → Summary → Needle agents
3. LLM-based metadata extraction
4. Multi-level summary index with intermediate summaries
5. MCP integration for ChromaDB access

Usage:
    python main.py              # Interactive mode (prompts to clean if data exists)
    python main.py --eval       # Run evaluation
    python main.py --build      # Build indexes only
    python main.py --no-cleanup # Instant startup - reuse existing data

Startup Behavior:
- First run: Full build (loads PDFs, extracts metadata, builds indexes) ~5-10 min
- Subsequent runs: Prompts to clean or reuse existing data
  - [y] Clean: Fresh build from scratch
  - [N] Keep: Instant startup (loads persisted indexes)
- --no-cleanup flag: Automatically reuses existing data for instant startup
"""

import argparse
import logging
import sys
from pathlib import Path

from src.utils.nltk_silencer import silence_nltk_downloads

# Silence NLTK download chatter before LlamaIndex initializes tokenizers.
silence_nltk_downloads()

from llama_index.llms.openai import OpenAI
from llama_index.core import Settings

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


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
    
    # Create router
    print("\n🔀 Creating router...")
    router = create_router_agent(
        structured_engine=structured_agent,
        summary_engine=summary_agent,
        needle_engine=needle_agent,
        llm=llm,
        verbose=True
    )
    print("✅ Router created (3-way: structured/summary/needle)")
    
    # Initialize MCP client
    print("\n🔌 Initializing MCP client...")
    mcp_client = ChromaDBMCPClient()
    mcp_client.connect()
    mcp_tools = create_mcp_tools(mcp_client)
    print(f"✅ MCP client connected with {len(mcp_tools)} tools")
    
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
    
    # Step 7: Create router
    print("\n🔀 Step 7/7: Creating router...")
    router = create_router_agent(
        structured_engine=structured_agent,
        summary_engine=summary_agent,
        needle_engine=needle_agent,
        llm=llm,
        verbose=True
    )
    print("✅ Router created (3-way: structured/summary/needle)")
    
    # Initialize MCP client
    print("\n🔌 Initializing MCP client...")
    mcp_client = ChromaDBMCPClient()
    mcp_client.connect()
    mcp_tools = create_mcp_tools(mcp_client)
    print(f"✅ MCP client connected with {len(mcp_tools)} tools")
    
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
        print("\nAvailable Commands:")
        print("   'query' - Send a query to the system")
        print("   'eval'  - Run evaluation suite")
        print("   'stats' - Show system statistics")
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
        print("💡 Available commands: query, eval, stats, quit")
    
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
  python main.py --eval              # Run full evaluation
  python main.py --eval --eval-subset 3   # Run only 3 test cases
  python main.py --eval --eval-delay 5    # 5 second delay between queries
  python main.py --build             # Build indexes and exit
  python main.py --no-cleanup        # Instant startup if data exists

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
    
    args = parser.parse_args()
    
    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)
    
    try:
        # Build system
        system = build_system(skip_cleanup=args.no_cleanup)
        
        if args.build:
            print("\n✅ Build complete. Exiting.")
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

