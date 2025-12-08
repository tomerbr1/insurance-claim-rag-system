"""
Configuration module for the Insurance Claims RAG System.

Loads settings from environment variables with sensible defaults.

Model Strategy:
- Main agents & routing: GPT-4 (best quality)
- Metadata extraction: GPT-4o-mini (cost-effective, better than 3.5-turbo)
- Embeddings: text-embedding-3-small (fast, good quality)
- Evaluation judge: Gemini 2.5 Flash (different provider to avoid bias)
"""

import os
from pathlib import Path
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# Base paths
PROJECT_ROOT = Path(__file__).parent.parent
DATA_DIR = PROJECT_ROOT / "insurance_claims_data"
CHROMA_DIR = PROJECT_ROOT / "chroma_db"
METADATA_DB = PROJECT_ROOT / "claims_metadata.db"

# OpenAI Configuration
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
OPENAI_MODEL = os.getenv("OPENAI_MODEL", "gpt-4")
OPENAI_MINI_MODEL = os.getenv("OPENAI_MINI_MODEL", "gpt-4o-mini")  # Cost-effective model
OPENAI_EMBEDDING_MODEL = os.getenv("OPENAI_EMBEDDING_MODEL", "text-embedding-3-small")

# Google/Gemini Configuration (for evaluation)
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
GEMINI_EVAL_MODEL = os.getenv("GEMINI_EVAL_MODEL", "gemini-2.0-flash")

# Chunking Configuration
CHUNK_SIZES = [1536, 512, 128]  # Large, Medium, Small
CHUNK_OVERLAP = 20

# Retrieval Configuration
SIMILARITY_TOP_K = 6
AUTO_MERGE_THRESHOLD = 0.5  # Merge if >50% of children match

# ChromaDB Configuration
CHROMA_COLLECTION_NAME = os.getenv("CHROMA_COLLECTION_NAME", "insurance_claims")

# Logging
LOG_LEVEL = os.getenv("LOG_LEVEL", "INFO")


def validate_config():
    """
    Validate that required configuration is present.
    
    Both OpenAI and Google API keys are REQUIRED:
    - OpenAI: for main agents, routing, and metadata extraction
    - Google: for unbiased evaluation (no fallback to avoid bias)
    """
    errors = []
    
    if not OPENAI_API_KEY:
        errors.append("❌ OPENAI_API_KEY is not set. Please set it in .env file.")
    
    if not GOOGLE_API_KEY:
        errors.append(
            "❌ GOOGLE_API_KEY is not set. This is REQUIRED for unbiased evaluation.\n"
            "   We use Gemini (Google) to evaluate OpenAI outputs to avoid self-evaluation bias.\n"
            "   Get your API key from: https://makersuite.google.com/app/apikey"
        )
    
    if not DATA_DIR.exists():
        errors.append(f"❌ Data directory not found: {DATA_DIR}")
    
    if errors:
        print("\n" + "=" * 70)
        print("❌ CONFIGURATION ERRORS")
        print("=" * 70)
        for error in errors:
            print(f"\n{error}")
        print("\n" + "=" * 70)
        raise ValueError("Configuration validation failed. Please fix the errors above.")
    
    return True


def validate_api_keys():
    """
    Validate that API keys are not just set, but actually valid.
    
    This makes real API calls to verify keys work before starting the system.
    """
    import logging
    logger = logging.getLogger(__name__)
    
    logger.info("Validating API keys...")
    
    # Test OpenAI key
    try:
        from llama_index.llms.openai import OpenAI
        test_llm = OpenAI(model="gpt-3.5-turbo", api_key=OPENAI_API_KEY, temperature=0)
        test_llm.complete("test")
        logger.info("✅ OpenAI API key is valid")
    except Exception as e:
        raise ValueError(f"❌ OpenAI API key validation failed: {str(e)}")
    
    # Test Google/Gemini key
    try:
        import google.generativeai as genai
        genai.configure(api_key=GOOGLE_API_KEY)
        test_model = genai.GenerativeModel(GEMINI_EVAL_MODEL)
        test_model.generate_content("test")
        logger.info("✅ Google API key is valid")
    except Exception as e:
        raise ValueError(
            f"❌ Google API key validation failed: {str(e)}\n"
            f"   Make sure you have a valid API key from: https://makersuite.google.com/app/apikey"
        )
    
    return True


def get_llm_config():
    """Get LLM configuration for LlamaIndex."""
    return {
        "model": OPENAI_MODEL,
        "api_key": OPENAI_API_KEY,
        "temperature": 0,  # Deterministic for consistency
    }


def get_embedding_config():
    """Get embedding configuration for LlamaIndex."""
    return {
        "model": OPENAI_EMBEDDING_MODEL,
        "api_key": OPENAI_API_KEY,
    }


# Print configuration on import (useful for debugging)
if __name__ == "__main__":
    print("=== Insurance Claims RAG System Configuration ===")
    print(f"Project Root: {PROJECT_ROOT}")
    print(f"Data Directory: {DATA_DIR}")
    print(f"ChromaDB Directory: {CHROMA_DIR}")
    print(f"\n📦 Models:")
    print(f"  Main Model: {OPENAI_MODEL}")
    print(f"  Cost-effective Model: {OPENAI_MINI_MODEL}")
    print(f"  Embedding Model: {OPENAI_EMBEDDING_MODEL}")
    print(f"  Evaluation Model: {GEMINI_EVAL_MODEL} (Gemini)")
    print(f"\n⚙️  Settings:")
    print(f"  Chunk Sizes: {CHUNK_SIZES}")
    print(f"\n🔑 API Keys:")
    print(f"  OpenAI: {'✅ Set' if OPENAI_API_KEY else '❌ Not set'}")
    print(f"  Google: {'✅ Set' if GOOGLE_API_KEY else '❌ Not set'}")
    
    try:
        validate_config()
        print("\n✅ Configuration is valid!")
    except ValueError as e:
        print(f"\n❌ Configuration errors:\n{e}")

