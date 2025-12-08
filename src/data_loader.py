"""
Data Loader Module - Loads PDF claim documents with LLM-based metadata extraction.

This module handles:
1. Loading PDF documents from the claims directory
2. Extracting structured metadata using LLM (not regex)
3. Enriching documents with metadata for downstream processing

Why LLM for metadata extraction?
- Handles format variations (dates in different formats)
- Understands semantic context (incident date vs filing date)
- More robust than brittle regex patterns
- Aligns with RAG-first philosophy
"""

import json
import logging
from pathlib import Path
from typing import List, Dict, Optional

from llama_index.core import Document, SimpleDirectoryReader
from llama_index.llms.openai import OpenAI
from llama_index.readers.file import PDFReader

from src.config import DATA_DIR, OPENAI_API_KEY, OPENAI_MINI_MODEL

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


# Prompt for LLM-based metadata extraction
METADATA_EXTRACTION_PROMPT = """
Extract the following metadata from this insurance claim document.
Return ONLY valid JSON, no additional text or explanation.

Required fields (use null if not found):
{{
    "claim_id": "string (e.g., CLM-2024-001847)",
    "claim_type": "string (e.g., Auto Accident, Slip and Fall, Water Damage)",
    "claimant": "string (full name of claimant)",
    "policy_number": "string",
    "claim_status": "string (OPEN, CLOSED, or SETTLED)",
    "total_value": "float (dollar amount without $ sign)",
    "incident_date": "string (YYYY-MM-DD format)",
    "filing_date": "string (YYYY-MM-DD format)",
    "settlement_date": "string or null (YYYY-MM-DD format if settled)"
}}

Document text:
{document_text}

JSON Response:
"""


def extract_metadata_with_llm(document_text: str, llm: OpenAI, max_chars: int = 15000) -> Dict:
    """
    Use LLM to extract structured metadata from document text.
    
    Args:
        document_text: The full text of the document
        llm: LlamaIndex OpenAI LLM instance
        max_chars: Maximum characters to send to LLM (default: 15000, ~4 pages)
    
    Returns:
        Dictionary with extracted metadata
    
    Why LLM instead of regex?
    - Format flexibility: Handles "Oct 15, 2024", "2024-10-15", "10/15/2024"
    - Semantic understanding: Knows incident_date != filing_date
    - Robustness: No brittle patterns to maintain
    - Consistency: Aligns with RAG philosophy
    
    Note: We use the full document text (up to max_chars) to ensure we capture
    metadata fields that appear at the end of documents (e.g., claim_status).
    """
    # Use full document text for small documents (typical insurance claims are 1-4 pages)
    # Only truncate for unusually large documents to stay within token limits
    if len(document_text) > max_chars:
        logger.warning(
            f"Document text is {len(document_text)} characters, truncating to {max_chars} characters. "
            f"Metadata extraction may be incomplete."
        )
        text_sample = document_text[:max_chars]
    else:
        text_sample = document_text
    
    prompt = METADATA_EXTRACTION_PROMPT.format(document_text=text_sample)
    
    try:
        response = llm.complete(prompt)
        response_text = response.text.strip()
        
        # Clean up response (remove markdown code blocks if present)
        if response_text.startswith("```"):
            lines = response_text.split("\n")
            response_text = "\n".join(lines[1:-1])
        
        metadata = json.loads(response_text)
        logger.debug(f"Extracted metadata: {metadata}")
        return metadata
        
    except json.JSONDecodeError as e:
        logger.error(f"Failed to parse LLM response as JSON: {e}")
        logger.error(f"Response was: {response_text}")
        return get_default_metadata()
    except Exception as e:
        logger.error(f"Error extracting metadata: {e}")
        return get_default_metadata()


def get_default_metadata() -> Dict:
    """Return default metadata structure when extraction fails."""
    return {
        "claim_id": None,
        "claim_type": None,
        "claimant": None,
        "policy_number": None,
        "claim_status": None,
        "total_value": None,
        "incident_date": None,
        "filing_date": None,
        "settlement_date": None
    }


def load_claim_documents(
    data_dir: Optional[Path] = None,
    extract_metadata: bool = True
) -> List[Document]:
    """
    Load all PDF claim documents from the data directory.
    
    Args:
        data_dir: Path to directory containing PDF files (default: DATA_DIR from config)
        extract_metadata: Whether to use LLM to extract metadata (default: True)
    
    Returns:
        List of LlamaIndex Document objects with metadata
    """
    data_dir = data_dir or DATA_DIR
    
    if not data_dir.exists():
        raise FileNotFoundError(f"Data directory not found: {data_dir}")
    
    logger.info(f"Loading documents from: {data_dir}")
    
    # Count PDF files
    pdf_files = list(data_dir.glob("*.pdf"))
    logger.info(f"Found {len(pdf_files)} PDF files")
    
    if not pdf_files:
        raise ValueError(f"No PDF files found in {data_dir}")
    
    # Load documents using SimpleDirectoryReader
    # Use file_extractor to ensure PDFs are loaded as single documents (not split by page)
    pdf_reader = PDFReader(return_full_document=True)  # Combine all pages into one document
    
    reader = SimpleDirectoryReader(
        input_dir=str(data_dir),
        required_exts=[".pdf"],
        recursive=False,
        file_extractor={".pdf": pdf_reader}  # Use our configured PDF reader
    )
    
    documents = reader.load_data()
    logger.info(f"Loaded {len(documents)} documents")
    
    # Extract metadata if requested
    if extract_metadata:
        logger.info(f"Extracting metadata using LLM ({OPENAI_MINI_MODEL})...")
        llm = OpenAI(model=OPENAI_MINI_MODEL, api_key=OPENAI_API_KEY, temperature=0)
        
        for i, doc in enumerate(documents):
            logger.info(f"Processing document {i+1}/{len(documents)}: {doc.metadata.get('file_name', 'Unknown')}")
            
            # Extract metadata
            extracted = extract_metadata_with_llm(doc.text, llm)
            
            # Update document metadata
            doc.metadata.update(extracted)
            
            # Also store source file info
            if 'file_name' in doc.metadata:
                doc.metadata['source_file'] = doc.metadata['file_name']
            
            logger.info(f"  → Claim ID: {extracted.get('claim_id')}, Type: {extracted.get('claim_type')}")
    
    logger.info(f"✅ Successfully loaded {len(documents)} documents with metadata")
    return documents


def get_documents_summary(documents: List[Document]) -> str:
    """
    Generate a summary of loaded documents for display.
    
    Args:
        documents: List of loaded documents
    
    Returns:
        Formatted string summary
    """
    lines = ["=" * 60, "📄 LOADED DOCUMENTS SUMMARY", "=" * 60]
    
    for doc in documents:
        meta = doc.metadata
        claim_id = meta.get('claim_id', 'Unknown')
        claim_type = meta.get('claim_type', 'Unknown')
        claimant = meta.get('claimant', 'Unknown')
        total = meta.get('total_value', 'N/A')
        status = meta.get('claim_status', 'Unknown')
        
        if isinstance(total, (int, float)):
            total_str = f"${total:,.2f}"
        else:
            total_str = str(total)
        
        lines.append(f"\n{claim_id}")
        lines.append(f"  Type: {claim_type}")
        lines.append(f"  Claimant: {claimant}")
        lines.append(f"  Total: {total_str}")
        lines.append(f"  Status: {status}")
    
    lines.append("\n" + "=" * 60)
    return "\n".join(lines)


# Quick test
if __name__ == "__main__":
    from src.config import validate_config
    
    print("Testing data loader...")
    
    try:
        validate_config()
        
        # Load documents
        documents = load_claim_documents()
        
        # Print summary
        print(get_documents_summary(documents))
        
        # Print first document text sample
        if documents:
            print("\n📝 First document text sample (first 500 chars):")
            print("-" * 40)
            print(documents[0].text[:500])
            print("-" * 40)
        
    except Exception as e:
        print(f"❌ Error: {e}")

