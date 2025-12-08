"""
Cleanup Module - Reset ChromaDB and SQLite databases for fresh runs.

This module ensures a clean slate before each system initialization
by removing persisted data from:
1. ChromaDB vector store (chroma_db/)
2. SQLite metadata store (claims_metadata.db)
"""

import logging
import shutil
from pathlib import Path
from typing import List

from src.config import CHROMA_DIR, METADATA_DB

# Setup logging
logger = logging.getLogger(__name__)


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
        Dictionary with cleanup results for each component
    """
    results = {
        'chromadb': False,
        'metadata_db': False,
        'errors': [],
        'skipped': False
    }
    
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
        print("=" * 50)
        
        if has_chromadb:
            try:
                total_size = sum(f.stat().st_size for f in CHROMA_DIR.rglob('*') if f.is_file())
                size_mb = round(total_size / (1024 * 1024), 2)
                print(f"   • ChromaDB vector store: {size_mb} MB")
            except Exception:
                print(f"   • ChromaDB vector store: exists")
        
        if has_metadata:
            try:
                size_mb = round(METADATA_DB.stat().st_size / (1024 * 1024), 2)
                print(f"   • SQLite metadata: {size_mb} MB")
            except Exception:
                print(f"   • SQLite metadata: exists")
        
        print("=" * 50)
        print("\n🤔 Clean existing data and rebuild from scratch?")
        print()
        print("   [y] Yes - Clean and rebuild everything")
        print("           (Slower, ensures fresh data)")
        print()
        print("   [N] No  - Keep existing data")
        print("           (System will still rebuild indexes - takes a few minutes)")
        print("           (Keeps ChromaDB vectors, avoids re-embedding)")
        print()
        print("   💡 TIP: Use --no-cleanup flag to skip this prompt")
        
        try:
            response = input("\nYour choice [y/N]: ").strip().lower()
        except (EOFError, KeyboardInterrupt):
            print("\n\n⏭️  Keeping existing data")
            results['skipped'] = True
            return results
        
        if response not in ['y', 'yes']:
            print("\n⏭️  Keeping existing ChromaDB data - will reuse embeddings")
            results['skipped'] = True
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

