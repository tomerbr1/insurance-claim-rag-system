"""
MCP ChromaDB Client - Enhanced Integration with ChromaDB.

This module provides an MCP-style wrapper for ChromaDB operations.
While using direct ChromaDB access (not the full MCP protocol), it
maintains an interface matching MCP server tools for future migration.

MCP (Model Context Protocol):
- Standard protocol for LLM tool integration
- This wrapper matches MCP server tool signatures
- Ready for future migration to true MCP protocol

Why MCP-style Interface?
- Industry-standard approach
- Compatible with MCP ecosystem patterns
- Enables agents to use familiar tool signatures
- Experience with MCP integration patterns
"""

import logging
from typing import List, Dict, Any, Optional
from pathlib import Path

from llama_index.core.tools import FunctionTool

from src.config import CHROMA_DIR, CHROMA_COLLECTION_NAME

# Setup logging
logger = logging.getLogger(__name__)


class MCPError(Exception):
    """Base exception for MCP client errors."""
    pass


class MCPConnectionError(MCPError):
    """Raised when connection to ChromaDB fails."""
    pass


class MCPCollectionError(MCPError):
    """Raised when collection operations fail."""
    pass


class ChromaDBMCPClient:
    """
    MCP-style wrapper for ChromaDB operations.

    Provides an interface matching MCP server tools while using
    direct ChromaDB access. Ready for future migration to true MCP.

    Features:
    - Connection state management
    - Collection operations (list, stats, peek, query)
    - Document operations (get by ID, search)
    - MCP-style response formatting

    Example:
        client = ChromaDBMCPClient()
        client.connect()

        # List collections
        collections = client.list_collections()

        # Get stats
        info = client.get_collection_info()

        # Query
        results = client.query_collection("towing cost")
    """

    # Server info for MCP compatibility
    SERVER_NAME = "chromadb-mcp-direct"
    SERVER_VERSION = "1.0.0"
    PROTOCOL_VERSION = "2024-11-05"

    def __init__(
        self,
        persist_dir: str = None,
        collection_name: str = None
    ):
        """
        Initialize ChromaDB MCP client.

        Args:
            persist_dir: Directory for ChromaDB persistence
            collection_name: Default collection name to use
        """
        self.persist_dir = persist_dir or str(CHROMA_DIR)
        self.collection_name = collection_name or CHROMA_COLLECTION_NAME
        self._client = None
        self._connected = False

    @property
    def is_connected(self) -> bool:
        """Check if client is connected to ChromaDB."""
        return self._connected and self._client is not None

    @property
    def server_info(self) -> Dict[str, Any]:
        """
        Get MCP server information.

        Returns mock server info matching MCP protocol format.
        """
        return {
            "name": self.SERVER_NAME,
            "version": self.SERVER_VERSION,
            "protocolVersion": self.PROTOCOL_VERSION,
            "capabilities": {
                "tools": True,
                "resources": False,
                "prompts": False
            },
            "status": "connected" if self.is_connected else "disconnected",
            "persistDir": self.persist_dir,
            "defaultCollection": self.collection_name
        }

    def connect(self) -> None:
        """
        Connect to ChromaDB.

        In MCP version, this would connect to the MCP server.
        Here we connect directly to ChromaDB.

        Raises:
            MCPConnectionError: If connection fails
        """
        try:
            import chromadb

            self._client = chromadb.PersistentClient(path=self.persist_dir)
            self._connected = True
            logger.info(f"Connected to ChromaDB: {self.persist_dir}")

        except Exception as e:
            logger.error(f"Failed to connect to ChromaDB: {e}")
            raise MCPConnectionError(f"Connection failed: {e}")

    def disconnect(self) -> None:
        """
        Disconnect from ChromaDB.

        Cleans up connection resources.
        """
        self._client = None
        self._connected = False
        logger.info("Disconnected from ChromaDB")

    def _ensure_connected(self) -> None:
        """Ensure client is connected, auto-connect if not."""
        if not self.is_connected:
            self.connect()

    def _get_collection(self, collection_name: str = None):
        """Get a ChromaDB collection by name."""
        self._ensure_connected()
        name = collection_name or self.collection_name

        try:
            return self._client.get_collection(name)
        except Exception as e:
            raise MCPCollectionError(f"Collection '{name}' not found: {e}")

    # ========== Collection Operations ==========

    def list_collections(self) -> List[str]:
        """
        List all ChromaDB collections.

        MCP Tool: chroma_list_collections

        Returns:
            List of collection names
        """
        self._ensure_connected()

        try:
            collections = self._client.list_collections()
            return [c.name for c in collections]
        except Exception as e:
            logger.error(f"Failed to list collections: {e}")
            return []

    def get_collection_info(self, collection_name: str = None) -> Dict[str, Any]:
        """
        Get detailed information about a collection.

        MCP Tool: chroma_get_collection_info

        Args:
            collection_name: Name of collection (default: configured collection)

        Returns:
            Dictionary with collection info including name, count, metadata
        """
        try:
            collection = self._get_collection(collection_name)
            return {
                'name': collection.name,
                'count': collection.count(),
                'metadata': collection.metadata or {}
            }
        except MCPCollectionError as e:
            return {'error': str(e)}
        except Exception as e:
            return {'error': f"Failed to get collection info: {e}"}

    def get_collection_count(self, collection_name: str = None) -> int:
        """
        Get document count for a collection.

        MCP Tool: chroma_get_collection_count

        Args:
            collection_name: Name of collection

        Returns:
            Number of documents in collection, or -1 on error
        """
        try:
            collection = self._get_collection(collection_name)
            return collection.count()
        except Exception as e:
            logger.error(f"Failed to get count: {e}")
            return -1

    def peek_collection(
        self,
        collection_name: str = None,
        n: int = 5
    ) -> Dict[str, Any]:
        """
        Preview first N documents in a collection.

        MCP Tool: chroma_peek_collection

        Args:
            collection_name: Name of collection
            n: Number of documents to preview (default 5)

        Returns:
            Dictionary with documents, metadatas, and ids
        """
        try:
            collection = self._get_collection(collection_name)
            results = collection.peek(limit=n)

            return {
                'documents': results.get('documents', []),
                'metadatas': results.get('metadatas', []),
                'ids': results.get('ids', []),
                'count': len(results.get('ids', []))
            }
        except MCPCollectionError as e:
            return {'error': str(e)}
        except Exception as e:
            return {'error': f"Failed to peek collection: {e}"}

    # ========== Document Operations ==========

    def query_collection(
        self,
        query_text: str,
        collection_name: str = None,
        n_results: int = 5
    ) -> Dict[str, Any]:
        """
        Query a collection with text using semantic similarity.

        MCP Tool: chroma_query_documents

        Args:
            query_text: Text to search for
            collection_name: Collection to search
            n_results: Number of results to return

        Returns:
            Query results with documents, metadatas, distances, ids
        """
        try:
            collection = self._get_collection(collection_name)

            results = collection.query(
                query_texts=[query_text],
                n_results=n_results
            )

            return {
                'documents': results.get('documents', [[]])[0],
                'metadatas': results.get('metadatas', [[]])[0],
                'distances': results.get('distances', [[]])[0],
                'ids': results.get('ids', [[]])[0]
            }

        except MCPCollectionError as e:
            return {'error': str(e)}
        except Exception as e:
            logger.error(f"Query error: {e}")
            return {'error': f"Query failed: {e}"}

    def get_document_by_id(
        self,
        doc_id: str,
        collection_name: str = None
    ) -> Dict[str, Any]:
        """
        Get a specific document by its ID.

        MCP Tool: chroma_get_document

        Args:
            doc_id: Document ID to retrieve
            collection_name: Collection to search

        Returns:
            Document with its text, metadata, and id
        """
        try:
            collection = self._get_collection(collection_name)

            results = collection.get(ids=[doc_id])

            if not results['ids']:
                return {'error': f"Document '{doc_id}' not found"}

            return {
                'id': results['ids'][0],
                'document': results['documents'][0] if results['documents'] else None,
                'metadata': results['metadatas'][0] if results['metadatas'] else {}
            }

        except MCPCollectionError as e:
            return {'error': str(e)}
        except Exception as e:
            return {'error': f"Failed to get document: {e}"}

    # ========== System Operations ==========

    def get_all_stats(self) -> Dict[str, Any]:
        """
        Get statistics about all collections.

        Returns:
            Dictionary with stats for all collections
        """
        self._ensure_connected()

        stats = {
            'server': self.server_info,
            'collections': {}
        }

        for collection_name in self.list_collections():
            stats['collections'][collection_name] = self.get_collection_info(collection_name)

        return stats


def create_mcp_tools(client: ChromaDBMCPClient) -> List[FunctionTool]:
    """
    Create LlamaIndex tools that wrap MCP/ChromaDB operations.

    These tools can be used by agents for direct vector store access.
    Returns 5 tools matching MCP server capabilities.

    Args:
        client: ChromaDBMCPClient instance

    Returns:
        List of 5 FunctionTool instances
    """

    def list_collections() -> str:
        """List all available vector store collections."""
        collections = client.list_collections()
        if collections:
            return f"Available collections: {', '.join(collections)}"
        return "No collections found."

    def collection_stats(collection_name: str = None) -> str:
        """Get statistics about a collection (count, metadata)."""
        name = collection_name or client.collection_name
        info = client.get_collection_info(name)
        if 'error' in info:
            return f"Error: {info['error']}"
        return f"Collection '{info['name']}': {info['count']} documents"

    def collection_count(collection_name: str = None) -> str:
        """Get quick document count for a collection."""
        name = collection_name or client.collection_name
        count = client.get_collection_count(name)
        if count < 0:
            return f"Error getting count for '{name}'"
        return f"Collection '{name}' has {count} documents"

    def direct_search(query: str, n_results: int = 5) -> str:
        """Search the claims collection directly using semantic similarity."""
        results = client.query_collection(query, n_results=n_results)
        if 'error' in results:
            return f"Error: {results['error']}"

        if not results['documents']:
            return "No matching documents found."

        output = []
        for i, (doc, meta) in enumerate(zip(results['documents'], results['metadatas']), 1):
            claim_id = meta.get('claim_id', 'Unknown')
            preview = doc[:200] + '...' if len(doc) > 200 else doc
            output.append(f"{i}. [{claim_id}] {preview}")

        return "\n".join(output)

    def peek_collection(collection_name: str = None, n: int = 5) -> str:
        """Preview first N documents in a collection."""
        name = collection_name or client.collection_name
        results = client.peek_collection(name, n)
        if 'error' in results:
            return f"Error: {results['error']}"

        if not results['documents']:
            return f"Collection '{name}' is empty."

        output = [f"Preview of '{name}' ({results['count']} documents shown):"]
        for i, (doc, meta, doc_id) in enumerate(
            zip(results['documents'], results['metadatas'], results['ids']), 1
        ):
            claim_id = meta.get('claim_id', doc_id[:20])
            preview = doc[:100] + '...' if doc and len(doc) > 100 else (doc or '[empty]')
            output.append(f"  {i}. [{claim_id}] {preview}")

        return "\n".join(output)

    # Create tools list (5 tools)
    tools = [
        FunctionTool.from_defaults(
            fn=list_collections,
            name="list_collections",
            description="List all available vector store collections in the system"
        ),
        FunctionTool.from_defaults(
            fn=collection_stats,
            name="collection_stats",
            description="Get statistics about a vector collection (document count, metadata)"
        ),
        FunctionTool.from_defaults(
            fn=collection_count,
            name="collection_count",
            description="Get quick document count for a collection"
        ),
        FunctionTool.from_defaults(
            fn=direct_search,
            name="direct_search",
            description="Perform direct vector similarity search on the claims collection"
        ),
        FunctionTool.from_defaults(
            fn=peek_collection,
            name="peek_collection",
            description="Preview first N documents in a collection"
        )
    ]

    return tools


# Quick test
if __name__ == "__main__":
    print("Testing Enhanced MCP ChromaDB client...")

    client = ChromaDBMCPClient()

    try:
        client.connect()

        print(f"\n📡 Server Info:")
        for key, value in client.server_info.items():
            print(f"   {key}: {value}")

        print(f"\n📚 Collections: {client.list_collections()}")
        print(f"\n📊 All Stats: {client.get_all_stats()}")

        # Create and test tools
        tools = create_mcp_tools(client)
        print(f"\n🔧 Created {len(tools)} MCP tools:")
        for tool in tools:
            print(f"   - {tool.metadata.name}: {tool.metadata.description[:50]}...")

        # Test tool functions
        print("\n🧪 Testing tools:")
        print(f"   list_collections: {tools[0].fn()}")
        print(f"   collection_stats: {tools[1].fn()}")
        print(f"   collection_count: {tools[2].fn()}")

        print("\n✅ MCP client test complete!")

    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback
        traceback.print_exc()
