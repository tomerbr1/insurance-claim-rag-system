"""
MCP ChromaDB Client - Integration with ChromaDB MCP Server.

This module integrates with an existing ChromaDB MCP server to enable
direct vector store operations from the LLM agents.

MCP (Model Context Protocol):
- Standard protocol for LLM tool integration
- We connect to existing servers, not build from scratch
- ChromaDB MCP provides collection management, queries, stats

Why MCP?
- Industry-standard approach
- Community-maintained servers
- Richer functionality than custom tools
- Experience with real MCP integration
"""

import logging
from typing import List, Dict, Any, Optional
import asyncio

from llama_index.core.tools import FunctionTool

from src.config import CHROMA_DIR, CHROMA_COLLECTION_NAME

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class ChromaDBMCPClient:
    """
    Client for interfacing with ChromaDB MCP server.
    
    If a full MCP server is not available, this provides a simplified
    direct ChromaDB interface with the same API.
    
    In production, this would connect to an actual MCP server.
    For this implementation, we provide direct ChromaDB access as fallback.
    """
    
    def __init__(
        self,
        persist_dir: str = None,
        collection_name: str = None
    ):
        """
        Initialize ChromaDB client.
        
        Args:
            persist_dir: Directory for ChromaDB persistence
            collection_name: Name of the collection to use
        """
        self.persist_dir = persist_dir or str(CHROMA_DIR)
        self.collection_name = collection_name or CHROMA_COLLECTION_NAME
        self._client = None
        self._collection = None
        self._connected = False
    
    def connect(self):
        """
        Connect to ChromaDB.
        
        In MCP version, this would connect to the MCP server.
        Here we connect directly to ChromaDB.
        """
        try:
            import chromadb
            
            self._client = chromadb.PersistentClient(path=self.persist_dir)
            
            # Try to get existing collection
            try:
                self._collection = self._client.get_collection(self.collection_name)
            except Exception:
                # Collection doesn't exist yet, that's OK
                self._collection = None
            
            self._connected = True
            logger.info(f"Connected to ChromaDB: {self.persist_dir}")
            
        except Exception as e:
            logger.error(f"Failed to connect to ChromaDB: {e}")
            raise
    
    def list_collections(self) -> List[str]:
        """
        List all ChromaDB collections.
        
        Returns:
            List of collection names
        """
        if not self._connected:
            self.connect()
        
        collections = self._client.list_collections()
        return [c.name for c in collections]
    
    def get_collection_info(self, collection_name: str = None) -> Dict[str, Any]:
        """
        Get information about a collection.
        
        Args:
            collection_name: Name of collection (default: configured collection)
        
        Returns:
            Dictionary with collection info
        """
        if not self._connected:
            self.connect()
        
        collection_name = collection_name or self.collection_name
        
        try:
            collection = self._client.get_collection(collection_name)
            return {
                'name': collection.name,
                'count': collection.count(),
                'metadata': collection.metadata or {}
            }
        except Exception as e:
            return {'error': str(e)}
    
    def query_collection(
        self,
        query_text: str,
        collection_name: str = None,
        n_results: int = 5
    ) -> Dict[str, Any]:
        """
        Query a collection with text.
        
        Note: In MCP, this would call the server's query tool.
        Here we use direct ChromaDB query.
        
        Args:
            query_text: Text to search for
            collection_name: Collection to search
            n_results: Number of results to return
        
        Returns:
            Query results
        """
        if not self._connected:
            self.connect()
        
        collection_name = collection_name or self.collection_name
        
        try:
            collection = self._client.get_collection(collection_name)
            
            # ChromaDB query
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
            
        except Exception as e:
            logger.error(f"Query error: {e}")
            return {'error': str(e)}
    
    def get_all_stats(self) -> Dict[str, Any]:
        """
        Get statistics about all collections.
        
        Returns:
            Dictionary with stats for all collections
        """
        if not self._connected:
            self.connect()
        
        stats = {}
        for collection_name in self.list_collections():
            stats[collection_name] = self.get_collection_info(collection_name)
        
        return stats


def create_mcp_tools(client: ChromaDBMCPClient) -> List[FunctionTool]:
    """
    Create LlamaIndex tools that wrap MCP/ChromaDB operations.
    
    These tools can be used by agents for direct vector store access.
    
    Args:
        client: ChromaDBMCPClient instance
    
    Returns:
        List of FunctionTool instances
    """
    
    def list_collections() -> str:
        """List all available vector store collections."""
        collections = client.list_collections()
        if collections:
            return f"Available collections: {', '.join(collections)}"
        return "No collections found."
    
    def get_collection_stats(collection_name: str = None) -> str:
        """Get statistics about a collection (count, metadata)."""
        info = client.get_collection_info(collection_name)
        if 'error' in info:
            return f"Error: {info['error']}"
        return f"Collection '{info['name']}': {info['count']} documents"
    
    def search_collection(query: str, n_results: int = 5) -> str:
        """Search the claims collection directly."""
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
    
    # Create tools
    tools = [
        FunctionTool.from_defaults(
            fn=list_collections,
            name="list_collections",
            description="List all available vector store collections in the system"
        ),
        FunctionTool.from_defaults(
            fn=get_collection_stats,
            name="collection_stats",
            description="Get statistics about a vector collection (document count, metadata)"
        ),
        FunctionTool.from_defaults(
            fn=search_collection,
            name="direct_search",
            description="Perform direct vector similarity search on the claims collection"
        )
    ]
    
    return tools


# Quick test
if __name__ == "__main__":
    print("Testing MCP ChromaDB client...")
    
    client = ChromaDBMCPClient()
    
    try:
        client.connect()
        
        print(f"\n📚 Collections: {client.list_collections()}")
        print(f"\n📊 Stats: {client.get_all_stats()}")
        
        # Create tools
        tools = create_mcp_tools(client)
        print(f"\n🔧 Created {len(tools)} MCP tools:")
        for tool in tools:
            print(f"   - {tool.metadata.name}: {tool.metadata.description[:50]}...")
        
        print("\n✅ MCP client test complete!")
        
    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback
        traceback.print_exc()

