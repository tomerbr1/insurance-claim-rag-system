"""
Cleanup Module - Reset ChromaDB and SQLite databases for fresh runs.

This module ensures a clean slate before each system initialization
by removing persisted data from:
1. ChromaDB vector store (chroma_db/)
2. SQLite metadata store (claims_metadata.db)

Also provides functions to:
- Check if existing data is valid and complete
- Determine if indexes can be reused vs need rebuilding
"""

import logging
import shutil
import json
import sqlite3
from pathlib import Path
from typing import List, Dict, Optional, Tuple

import chromadb

from src.config import CHROMA_DIR, METADATA_DB, CHROMA_COLLECTION_NAME

# Setup logging
logger = logging.getLogger(__name__)

# Constants for persistence
SUMMARY_COLLECTION_NAME = "insurance_claims_summaries"
DOCSTORE_FILE = "docstore.json"
SUMMARY_NODES_FILE = "summary_nodes.json"


def check_existing_data() -> Dict[str, any]:
    """
    Check what existing data is available and if it's valid for reuse.
    
    Returns:
        Dictionary with:
        - has_chromadb: bool - ChromaDB exists with data
        - has_metadata_db: bool - SQLite exists with claims
        - has_summary_data: bool - Summary nodes and docstore exist
        - chromadb_count: int - Number of documents in ChromaDB
        - metadata_count: int - Number of claims in SQLite
        - summary_count: int - Number of summary nodes
        - is_complete: bool - All data present and consistent
        - can_reuse: bool - Data is valid for reuse (skipping build)
    """
    result = {
        'has_chromadb': False,
        'has_metadata_db': False,
        'has_summary_data': False,
        'chromadb_count': 0,
        'metadata_count': 0,
        'summary_count': 0,
        'is_complete': False,
        'can_reuse': False,
        'details': {}
    }
    
    # Check ChromaDB
    if CHROMA_DIR.exists():
        try:
            chroma_client = chromadb.PersistentClient(path=str(CHROMA_DIR))
            collections = chroma_client.list_collections()
            collection_names = [c.name for c in collections]
            
            if CHROMA_COLLECTION_NAME in collection_names:
                collection = chroma_client.get_collection(CHROMA_COLLECTION_NAME)
                count = collection.count()
                result['has_chromadb'] = count > 0
                result['chromadb_count'] = count
                result['details']['chromadb_collection'] = CHROMA_COLLECTION_NAME
            
            # Check for summary collection
            if SUMMARY_COLLECTION_NAME in collection_names:
                summary_collection = chroma_client.get_collection(SUMMARY_COLLECTION_NAME)
                summary_count = summary_collection.count()
                result['summary_count'] = summary_count
                result['has_summary_data'] = summary_count > 0
                result['details']['summary_collection'] = SUMMARY_COLLECTION_NAME
                
        except Exception as e:
            logger.warning(f"Error checking ChromaDB: {e}")
            result['details']['chromadb_error'] = str(e)
    
    # Check for docstore file
    docstore_path = CHROMA_DIR / DOCSTORE_FILE
    if docstore_path.exists():
        try:
            with open(docstore_path, 'r') as f:
                docstore_data = json.load(f)
                result['details']['docstore_nodes'] = len(docstore_data.get('nodes', {}))
        except Exception as e:
            logger.warning(f"Error checking docstore: {e}")
            result['details']['docstore_error'] = str(e)
    
    # Check SQLite metadata
    if METADATA_DB.exists():
        try:
            conn = sqlite3.connect(str(METADATA_DB))
            cursor = conn.execute("SELECT COUNT(*) FROM claims")
            count = cursor.fetchone()[0]
            conn.close()
            
            result['has_metadata_db'] = count > 0
            result['metadata_count'] = count
            result['details']['metadata_db'] = str(METADATA_DB)
            
        except Exception as e:
            logger.warning(f"Error checking metadata DB: {e}")
            result['details']['metadata_error'] = str(e)
    
    # Determine if data is complete and can be reused
    # Complete means ALL of these must exist:
    # 1. ChromaDB hierarchical index (for needle agent vector search)
    # 2. SQLite metadata (for structured agent SQL queries)
    # 3. Summary index in ChromaDB (for summary agent)
    # 4. Docstore JSON (for auto-merging retriever in needle agent)
    
    has_docstore = result['details'].get('docstore_nodes', 0) > 0
    
    if (result['has_chromadb'] and 
        result['has_metadata_db'] and 
        result['has_summary_data'] and
        has_docstore):
        
        # All components exist - can reuse
        if result['metadata_count'] > 0:
            result['is_complete'] = True
            result['can_reuse'] = True
    
    # Store docstore status for detailed reporting
    result['has_docstore'] = has_docstore
    
    return result


def print_existing_data_summary(data_info: Dict) -> None:
    """
    Print a human-readable summary of existing data.
    
    Args:
        data_info: Dictionary from check_existing_data()
    """
    print("\n📊 EXISTING DATA STATUS")
    print("=" * 50)
    
    # ChromaDB status
    if data_info['has_chromadb']:
        print(f"   ✅ ChromaDB: {data_info['chromadb_count']} vectors indexed")
    else:
        print(f"   ❌ ChromaDB: No data found")
    
    # Summary index status
    if data_info['has_summary_data']:
        print(f"   ✅ Summary Index: {data_info['summary_count']} summary nodes")
    else:
        print(f"   ❌ Summary Index: No data found (needs rebuild)")
    
    # SQLite status
    if data_info['has_metadata_db']:
        print(f"   ✅ SQLite Metadata: {data_info['metadata_count']} claims")
    else:
        print(f"   ❌ SQLite Metadata: No data found")
    
    # Docstore status
    has_docstore = data_info.get('has_docstore', False)
    docstore_nodes = data_info.get('details', {}).get('docstore_nodes', 0)
    if has_docstore:
        print(f"   ✅ Docstore: {docstore_nodes} hierarchical nodes")
    else:
        print(f"   ❌ Docstore: No data found (needs rebuild)")
    
    # Overall status
    print("-" * 50)
    if data_info['can_reuse']:
        print("   ✨ STATUS: Data is COMPLETE - can skip indexing!")
    elif data_info['has_chromadb'] or data_info['has_metadata_db']:
        missing = []
        if not data_info['has_chromadb']:
            missing.append("ChromaDB vectors")
        if not data_info['has_metadata_db']:
            missing.append("SQLite metadata")
        if not data_info['has_summary_data']:
            missing.append("summary index")
        if not has_docstore:
            missing.append("docstore")
        if missing:
            print(f"   ⚠️  STATUS: INCOMPLETE - missing: {', '.join(missing)}")
            print(f"   💡 TIP: Do one full rebuild to enable instant startup")
        else:
            print("   ⚠️  STATUS: Data is INCOMPLETE")
    else:
        print("   📭 STATUS: No existing data - full build required")
    
    print("=" * 50)


def cleanup_chromadb(chroma_dir: Path = None) -> bool:
    """
    Remove the ChromaDB persistence directory.
    
    Args:
        chroma_dir: Path to ChromaDB directory (default: from config)
    
    Returns:
        True if cleanup was performed, False if nothing to clean
    """
    chroma_dir = chroma_dir or CHROMA_DIR
    
    if chroma_dir.exists():
        try:
            shutil.rmtree(chroma_dir)
            logger.info(f"✓ Removed ChromaDB directory: {chroma_dir}")
            return True
        except Exception as e:
            logger.error(f"Failed to remove ChromaDB directory: {e}")
            raise
    else:
        logger.debug(f"ChromaDB directory does not exist: {chroma_dir}")
        return False


def cleanup_metadata_db(db_path: Path = None) -> bool:
    """
    Remove the SQLite metadata database file.
    
    Args:
        db_path: Path to SQLite database file (default: from config)
    
    Returns:
        True if cleanup was performed, False if nothing to clean
    """
    db_path = db_path or METADATA_DB
    
    if db_path.exists():
        try:
            db_path.unlink()
            logger.info(f"✓ Removed metadata database: {db_path}")
            return True
        except Exception as e:
            logger.error(f"Failed to remove metadata database: {e}")
            raise
    else:
        logger.debug(f"Metadata database does not exist: {db_path}")
        return False


def cleanup_all(verbose: bool = True, force: bool = False, interactive: bool = True) -> dict:
    """
    Perform full cleanup of all persisted data.
    
    This removes:
    - ChromaDB vector store directory
    - SQLite metadata database
    
    Args:
        verbose: Whether to print status messages
        force: If True, skip confirmation and clean immediately
        interactive: If True, prompt user when data exists (default: True)
    
    Returns:
        Dictionary with cleanup results for each component:
        - chromadb: bool - was chromadb cleaned
        - metadata_db: bool - was metadata cleaned
        - errors: list - any errors encountered
        - skipped: bool - user chose to keep data
        - can_reuse: bool - existing data can be fully reused (skip all build steps)
        - data_info: dict - details about existing data
    """
    results = {
        'chromadb': False,
        'metadata_db': False,
        'errors': [],
        'skipped': False,
        'can_reuse': False,
        'data_info': None
    }
    
    # Check existing data thoroughly
    data_info = check_existing_data()
    results['data_info'] = data_info
    
    # Check if there's anything to clean
    has_chromadb = CHROMA_DIR.exists()
    has_metadata = METADATA_DB.exists()
    
    if not has_chromadb and not has_metadata:
        if verbose:
            print("\n✨ No existing data found - starting fresh")
        return results
    
    # If data exists and interactive mode is on, ask user
    if interactive and not force:
        print("\n⚠️  EXISTING DATA FOUND")
        print("=" * 60)
        
        # Show detailed data info
        print_existing_data_summary(data_info)
        
        # Show disk usage
        print("\n💾 DISK USAGE:")
        if has_chromadb:
            try:
                total_size = sum(f.stat().st_size for f in CHROMA_DIR.rglob('*') if f.is_file())
                size_mb = round(total_size / (1024 * 1024), 2)
                print(f"   • ChromaDB: {size_mb} MB")
            except Exception:
                print(f"   • ChromaDB: exists")
        
        if has_metadata:
            try:
                size_mb = round(METADATA_DB.stat().st_size / (1024 * 1024), 2)
                print(f"   • SQLite: {size_mb} MB")
            except Exception:
                print(f"   • SQLite: exists")
        
        print("\n" + "=" * 60)
        print("\n🤔 What would you like to do?")
        print()
        print("   [y] Yes - Clean and rebuild everything")
        print("           (Slower - re-processes all PDFs, ~5-10 min)")
        print()
        
        if data_info['can_reuse']:
            print("   [N] No  - REUSE existing data (RECOMMENDED)")
            print("           ⚡ INSTANT startup - skip all indexing!")
            print("           (Uses existing ChromaDB + SQLite + summaries)")
        else:
            print("   [N] No  - Keep partial data")
            print("           (May still need some rebuilding)")
        
        print()
        print("   💡 TIP: Use --no-cleanup flag to skip this prompt")
        
        try:
            response = input("\nYour choice [y/N]: ").strip().lower()
        except (EOFError, KeyboardInterrupt):
            print("\n\n⏭️  Keeping existing data")
            results['skipped'] = True
            results['can_reuse'] = data_info['can_reuse']
            return results
        
        if response not in ['y', 'yes']:
            if data_info['can_reuse']:
                print("\n⚡ REUSING existing data - instant startup!")
            else:
                print("\n⏭️  Keeping existing data")
            results['skipped'] = True
            results['can_reuse'] = data_info['can_reuse']
            return results
        
        print("\n🧹 Cleaning existing data...")
    elif verbose:
        print("\n🧹 CLEANUP: Removing existing data for fresh start...")
    
    # Clean ChromaDB
    try:
        results['chromadb'] = cleanup_chromadb()
        if verbose and results['chromadb']:
            print(f"   ✓ ChromaDB directory removed")
    except Exception as e:
        results['errors'].append(f"ChromaDB: {e}")
        if verbose:
            print(f"   ✗ ChromaDB cleanup failed: {e}")
    
    # Clean metadata database
    try:
        results['metadata_db'] = cleanup_metadata_db()
        if verbose and results['metadata_db']:
            print(f"   ✓ Metadata database removed")
    except Exception as e:
        results['errors'].append(f"Metadata DB: {e}")
        if verbose:
            print(f"   ✗ Metadata database cleanup failed: {e}")
    
    # Summary
    cleaned_count = sum([results['chromadb'], results['metadata_db']])
    
    if verbose:
        if cleaned_count > 0:
            print(f"   ✅ Cleanup complete ({cleaned_count} items removed)")
        else:
            print(f"   ℹ️  Nothing to clean (fresh state)")
    
    if results['errors']:
        logger.warning(f"Cleanup completed with errors: {results['errors']}")
    
    return results


def get_storage_info() -> dict:
    """
    Get information about current storage state.
    
    Returns:
        Dictionary with storage information
    """
    info = {
        'chromadb': {
            'exists': CHROMA_DIR.exists(),
            'path': str(CHROMA_DIR),
            'size_mb': None
        },
        'metadata_db': {
            'exists': METADATA_DB.exists(),
            'path': str(METADATA_DB),
            'size_mb': None
        }
    }
    
    # Get sizes if they exist
    if CHROMA_DIR.exists():
        total_size = sum(f.stat().st_size for f in CHROMA_DIR.rglob('*') if f.is_file())
        info['chromadb']['size_mb'] = round(total_size / (1024 * 1024), 2)
    
    if METADATA_DB.exists():
        info['metadata_db']['size_mb'] = round(METADATA_DB.stat().st_size / (1024 * 1024), 2)
    
    return info


# Quick test
if __name__ == "__main__":
    print("=== Storage Cleanup Utility ===\n")
    
    # Show current state
    print("📊 Current storage state:")
    info = get_storage_info()
    
    for store, data in info.items():
        status = "✓ exists" if data['exists'] else "✗ not found"
        size = f"({data['size_mb']} MB)" if data['size_mb'] else ""
        print(f"   {store}: {status} {size}")
        print(f"      Path: {data['path']}")
    
    # Ask before cleanup
    print("\n⚠️  This will delete all persisted data!")
    response = input("Proceed with cleanup? (y/N): ").strip().lower()
    
    if response == 'y':
        results = cleanup_all(verbose=True)
        print(f"\nResults: {results}")
    else:
        print("Cleanup cancelled.")

