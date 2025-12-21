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
from typing import List, Dict, Optional, Tuple

import pdfplumber
from llama_index.core import Document, SimpleDirectoryReader
from llama_index.llms.openai import OpenAI
from llama_index.readers.file import PDFReader

from src.config import DATA_DIR, OPENAI_API_KEY, OPENAI_MINI_MODEL

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


# =============================================================================
# Table Extraction Functions
# =============================================================================

def extract_tables_from_pdf(pdf_path: Path, min_rows: int = 2, min_cols: int = 2) -> List[Dict]:
    """
    Extract tables from a PDF using pdfplumber.

    Args:
        pdf_path: Path to the PDF file
        min_rows: Minimum rows for a valid table (default: 2)
        min_cols: Minimum columns for a valid table (default: 2)

    Returns:
        List of dicts with 'page', 'table_index', 'table_data' for each table
    """
    tables = []
    try:
        with pdfplumber.open(pdf_path) as pdf:
            for page_num, page in enumerate(pdf.pages):
                page_tables = page.extract_tables()
                for i, table in enumerate(page_tables):
                    # Validate table has minimum dimensions
                    if table and len(table) >= min_rows:
                        if table[0] and len(table[0]) >= min_cols:
                            tables.append({
                                'page': page_num + 1,
                                'table_index': i,
                                'table_data': table
                            })
    except Exception as e:
        logger.warning(f"Error extracting tables from {pdf_path}: {e}")

    return tables


def table_to_markdown(table_data: List[List[str]]) -> str:
    """
    Convert extracted table data to markdown format.

    Args:
        table_data: 2D list of cell values (first row = headers)

    Returns:
        Markdown-formatted table string
    """
    if not table_data or not table_data[0]:
        return ""

    def clean_cell(cell) -> str:
        """Clean a cell value for markdown."""
        if cell is None:
            return ""
        # Replace newlines and normalize whitespace
        return str(cell).strip().replace('\n', ' ').replace('|', '\\|')

    headers = [clean_cell(h) for h in table_data[0]]
    rows = [[clean_cell(c) for c in row] for row in table_data[1:]]

    # Calculate column widths for nice formatting
    widths = []
    for i, h in enumerate(headers):
        col_width = len(h)
        for row in rows:
            if i < len(row):
                col_width = max(col_width, len(row[i]))
        widths.append(max(col_width, 3))  # Minimum width of 3

    lines = []

    # Header row
    header_cells = [h.ljust(w) for h, w in zip(headers, widths)]
    lines.append("| " + " | ".join(header_cells) + " |")

    # Separator row
    sep_cells = ["-" * w for w in widths]
    lines.append("|" + "|".join("-" + s + "-" for s in sep_cells) + "|")

    # Data rows
    for row in rows:
        # Pad row if shorter than headers
        padded_row = row + [""] * (len(headers) - len(row))
        row_cells = [c.ljust(w) for c, w in zip(padded_row, widths)]
        lines.append("| " + " | ".join(row_cells) + " |")

    return "\n".join(lines)


def load_pdf_with_tables(pdf_path: Path) -> Tuple[str, List[Dict]]:
    """
    Load PDF text and extract tables, merging them into the document.

    Strategy:
    1. Extract plain text using pypdf (via PDFReader)
    2. Extract tables using pdfplumber
    3. Convert tables to markdown
    4. Append tables to text with markers

    Args:
        pdf_path: Path to PDF file

    Returns:
        Tuple of (enhanced_text, table_info_list)
    """
    # Step 1: Get base text using existing PDFReader
    try:
        pdf_reader = PDFReader(return_full_document=True)
        docs = pdf_reader.load_data(pdf_path)
        base_text = docs[0].text if docs else ""
    except Exception as e:
        logger.error(f"Error reading PDF text from {pdf_path}: {e}")
        base_text = ""

    # Step 2: Extract tables using pdfplumber
    tables = extract_tables_from_pdf(pdf_path)

    if not tables:
        # No tables found - return original text
        return base_text, []

    # Step 3: Convert tables to markdown and append to text
    table_sections = []
    enhanced_text = base_text

    for table_info in tables:
        markdown_table = table_to_markdown(table_info['table_data'])
        if markdown_table:
            # Create a marked section for the table
            section = f"\n\n[EXTRACTED TABLE - Page {table_info['page']}]\n{markdown_table}\n[END TABLE]\n"
            enhanced_text += section

            table_sections.append({
                'page': table_info['page'],
                'table_index': table_info['table_index'],
                'row_count': len(table_info['table_data']),
                'col_count': len(table_info['table_data'][0]) if table_info['table_data'] else 0,
                'markdown': markdown_table
            })

    return enhanced_text, table_sections


# =============================================================================
# Metadata Extraction
# =============================================================================

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
    extract_metadata: bool = True,
    extract_tables: bool = True
) -> List[Document]:
    """
    Load all PDF claim documents from the data directory.

    Args:
        data_dir: Path to directory containing PDF files (default: DATA_DIR from config)
        extract_metadata: Whether to use LLM to extract metadata (default: True)
        extract_tables: Whether to extract tables from PDFs (default: True)

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

    documents = []

    if extract_tables:
        # Load PDFs with table extraction
        logger.info("Loading PDFs with table extraction enabled...")
        for pdf_path in sorted(pdf_files):
            text, table_info = load_pdf_with_tables(pdf_path)

            doc = Document(
                text=text,
                metadata={
                    'file_name': pdf_path.name,
                    'file_path': str(pdf_path),
                    'has_tables': len(table_info) > 0,
                    'table_count': len(table_info)
                }
            )

            if table_info:
                logger.info(f"  {pdf_path.name}: Extracted {len(table_info)} table(s)")

            documents.append(doc)
    else:
        # Original approach without table extraction
        pdf_reader = PDFReader(return_full_document=True)

        reader = SimpleDirectoryReader(
            input_dir=str(data_dir),
            required_exts=[".pdf"],
            recursive=False,
            file_extractor={".pdf": pdf_reader}
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

