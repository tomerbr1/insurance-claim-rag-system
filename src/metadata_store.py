"""
Metadata Store Module - SQLite database for structured claim metadata.

This module implements the "structured" part of the hybrid architecture:
- SQL database for fast exact-match queries
- Supports filtering, aggregation, and range queries
- Complements RAG for unstructured narrative queries

Why Hybrid (SQL + RAG)?
- Structured queries (filters, aggregations) are faster and more accurate with SQL
- Narrative queries require semantic understanding (RAG)
- Best of both worlds
"""

import sqlite3
import logging
from pathlib import Path
from typing import Dict, List, Optional, Any
from datetime import datetime

import pandas as pd

from src.config import METADATA_DB

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class MetadataStore:
    """
    SQLite-based metadata storage for insurance claims.
    
    Enables fast SQL queries for structured data while RAG handles narratives.
    
    Schema:
        claims (
            claim_id TEXT PRIMARY KEY,
            claim_type TEXT,
            claimant TEXT,
            policy_number TEXT,
            claim_status TEXT,
            total_value REAL,
            incident_date DATE,
            filing_date DATE,
            settlement_date DATE,
            source_file TEXT,
            created_at TIMESTAMP
        )
    """
    
    def __init__(self, db_path: Optional[Path] = None):
        """
        Initialize the metadata store.
        
        Args:
            db_path: Path to SQLite database file (default: from config)
        """
        self.db_path = db_path or METADATA_DB
        self.conn = sqlite3.connect(str(self.db_path))
        self.conn.row_factory = sqlite3.Row  # Return rows as dictionaries
        self._create_tables()
        logger.info(f"MetadataStore initialized: {self.db_path}")
    
    def _create_tables(self):
        """Create the claims table with proper indexes."""
        self.conn.execute("""
            CREATE TABLE IF NOT EXISTS claims (
                claim_id TEXT PRIMARY KEY,
                claim_type TEXT,
                claimant TEXT,
                policy_number TEXT,
                claim_status TEXT,
                total_value REAL,
                incident_date DATE,
                filing_date DATE,
                settlement_date DATE,
                source_file TEXT,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        """)
        
        # Create indexes for common query patterns
        indexes = [
            ("idx_claim_type", "claim_type"),
            ("idx_claim_status", "claim_status"),
            ("idx_total_value", "total_value"),
            ("idx_incident_date", "incident_date"),
            ("idx_claimant", "claimant"),
        ]
        
        for idx_name, column in indexes:
            self.conn.execute(f"""
                CREATE INDEX IF NOT EXISTS {idx_name} ON claims({column})
            """)
        
        self.conn.commit()
        logger.debug("Database tables and indexes created")
    
    def insert_claim(self, metadata: Dict[str, Any]) -> bool:
        """
        Insert or update claim metadata.
        
        Args:
            metadata: Dictionary with claim metadata
        
        Returns:
            True if successful, False otherwise
        """
        try:
            self.conn.execute("""
                INSERT OR REPLACE INTO claims 
                (claim_id, claim_type, claimant, policy_number, claim_status,
                 total_value, incident_date, filing_date, settlement_date, source_file)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                metadata.get('claim_id'),
                metadata.get('claim_type'),
                metadata.get('claimant'),
                metadata.get('policy_number'),
                metadata.get('claim_status'),
                metadata.get('total_value'),
                metadata.get('incident_date'),
                metadata.get('filing_date'),
                metadata.get('settlement_date'),
                metadata.get('source_file')
            ))
            self.conn.commit()
            logger.debug(f"Inserted claim: {metadata.get('claim_id')}")
            return True
        except Exception as e:
            logger.error(f"Error inserting claim: {e}")
            return False
    
    def insert_claims_batch(self, metadata_list: List[Dict[str, Any]]) -> int:
        """
        Insert multiple claims in a batch.
        
        Args:
            metadata_list: List of metadata dictionaries
        
        Returns:
            Number of successfully inserted claims
        """
        count = 0
        for metadata in metadata_list:
            if self.insert_claim(metadata):
                count += 1
        logger.info(f"Inserted {count}/{len(metadata_list)} claims")
        return count
    
    def query(self, sql: str) -> pd.DataFrame:
        """
        Execute a raw SQL query and return results as DataFrame.
        
        Args:
            sql: SQL SELECT query
        
        Returns:
            pandas DataFrame with results
        """
        return pd.read_sql_query(sql, self.conn)
    
    def get_by_id(self, claim_id: str) -> Optional[Dict]:
        """
        Get a single claim by ID.
        
        Args:
            claim_id: The claim ID to look up
        
        Returns:
            Dictionary with claim data, or None if not found
        """
        cursor = self.conn.execute(
            "SELECT * FROM claims WHERE claim_id = ?", 
            (claim_id,)
        )
        row = cursor.fetchone()
        return dict(row) if row else None
    
    def get_all_claims(self) -> List[Dict]:
        """Get all claims as a list of dictionaries."""
        cursor = self.conn.execute("SELECT * FROM claims ORDER BY claim_id")
        return [dict(row) for row in cursor.fetchall()]
    
    def filter_claims(
        self,
        claim_type: Optional[str] = None,
        claim_status: Optional[str] = None,
        min_value: Optional[float] = None,
        max_value: Optional[float] = None,
        date_from: Optional[str] = None,
        date_to: Optional[str] = None,
        claimant_contains: Optional[str] = None
    ) -> pd.DataFrame:
        """
        Filter claims by various criteria.
        
        Args:
            claim_type: Filter by claim type (exact match)
            claim_status: Filter by status (OPEN/CLOSED/SETTLED)
            min_value: Minimum total value
            max_value: Maximum total value
            date_from: Incident date from (YYYY-MM-DD)
            date_to: Incident date to (YYYY-MM-DD)
            claimant_contains: Claimant name contains (case-insensitive)
        
        Returns:
            DataFrame with matching claims
        """
        conditions = []
        params = []
        
        if claim_type:
            conditions.append("claim_type = ?")
            params.append(claim_type)
        
        if claim_status:
            conditions.append("claim_status = ?")
            params.append(claim_status)
        
        if min_value is not None:
            conditions.append("total_value >= ?")
            params.append(min_value)
        
        if max_value is not None:
            conditions.append("total_value <= ?")
            params.append(max_value)
        
        if date_from:
            conditions.append("incident_date >= ?")
            params.append(date_from)
        
        if date_to:
            conditions.append("incident_date <= ?")
            params.append(date_to)
        
        if claimant_contains:
            conditions.append("LOWER(claimant) LIKE ?")
            params.append(f"%{claimant_contains.lower()}%")
        
        where_clause = " AND ".join(conditions) if conditions else "1=1"
        sql = f"SELECT * FROM claims WHERE {where_clause} ORDER BY claim_id"
        
        return pd.read_sql_query(sql, self.conn, params=params)
    
    def get_statistics(self) -> Dict:
        """
        Get aggregate statistics about claims.
        
        Returns:
            Dictionary with statistics
        """
        stats = {}
        
        # Total count
        cursor = self.conn.execute("SELECT COUNT(*) FROM claims")
        stats['total_claims'] = cursor.fetchone()[0]
        
        # By status
        cursor = self.conn.execute("""
            SELECT claim_status, COUNT(*) as count 
            FROM claims 
            GROUP BY claim_status
        """)
        stats['by_status'] = {row[0]: row[1] for row in cursor.fetchall()}
        
        # By type
        cursor = self.conn.execute("""
            SELECT claim_type, COUNT(*) as count 
            FROM claims 
            GROUP BY claim_type
        """)
        stats['by_type'] = {row[0]: row[1] for row in cursor.fetchall()}
        
        # Value statistics
        cursor = self.conn.execute("""
            SELECT 
                SUM(total_value) as total,
                AVG(total_value) as average,
                MIN(total_value) as min,
                MAX(total_value) as max
            FROM claims
        """)
        row = cursor.fetchone()
        stats['value_stats'] = {
            'total': row[0],
            'average': row[1],
            'min': row[2],
            'max': row[3]
        }
        
        return stats
    
    def clear(self):
        """Clear all data from the claims table."""
        self.conn.execute("DELETE FROM claims")
        self.conn.commit()
        logger.info("Cleared all claims from metadata store")
    
    def close(self):
        """Close the database connection."""
        self.conn.close()
        logger.info("MetadataStore connection closed")


# Quick test
if __name__ == "__main__":
    print("Testing MetadataStore...")
    
    # Create store (in memory for testing)
    store = MetadataStore(db_path=Path(":memory:"))
    
    # Insert test data
    test_claims = [
        {
            "claim_id": "CLM-2024-001847",
            "claim_type": "Auto Accident",
            "claimant": "Robert J. Mitchell",
            "policy_number": "POL-882341",
            "claim_status": "SETTLED",
            "total_value": 14050.33,
            "incident_date": "2024-10-15",
            "filing_date": "2024-10-16",
            "settlement_date": "2024-11-30"
        },
        {
            "claim_id": "CLM-2024-003012",
            "claim_type": "Slip and Fall",
            "claimant": "Patricia Vaughn",
            "policy_number": "POL-445566",
            "claim_status": "SETTLED",
            "total_value": 142500.00,
            "incident_date": "2024-09-20",
            "filing_date": "2024-09-21",
            "settlement_date": "2024-12-01"
        }
    ]
    
    store.insert_claims_batch(test_claims)
    
    # Test queries
    print("\n📊 All claims:")
    print(store.query("SELECT claim_id, claim_type, total_value FROM claims"))
    
    print("\n📊 Claims over $50k:")
    print(store.filter_claims(min_value=50000))
    
    print("\n📊 Statistics:")
    stats = store.get_statistics()
    for key, value in stats.items():
        print(f"  {key}: {value}")
    
    print("\n✅ MetadataStore test complete!")
    store.close()

