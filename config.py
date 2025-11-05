"""
Configuration settings for Pharma Intelligence AI.
Centralizes all API endpoints, rate limits, and application settings.
"""

import os
from dotenv import load_dotenv
from pathlib import Path

# Load environment variables
load_dotenv()

# ========================================
# API KEYS
# ========================================
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "")
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY", "")
NCBI_API_KEY = os.getenv("NCBI_API_KEY", "")

# ========================================
# API ENDPOINTS
# ========================================
CLINICAL_TRIALS_API = "https://clinicaltrials.gov/api/v2/studies"
PUBCHEM_API = "https://pubchem.ncbi.nlm.nih.gov/rest/pug"
PUBMED_API = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils"
FDA_API = "https://api.fda.gov/drug"
USPTO_API = "https://developer.uspto.gov/ibd-api/v1"

# ========================================
# RATE LIMITS (requests per second)
# ========================================
RATE_LIMITS = {
    "pubmed": 3 if not NCBI_API_KEY else 10,
    "clinical_trials": 10,
    "pubchem": 5,
    "fda": 10,
    "openai": 3,
    "gemini": 1
}

# ========================================
# LLM CONFIGURATION
# ========================================
# Default model identifiers for CrewAI provider syntax
DEFAULT_OPENAI_MODEL = "openai/gpt-4o-mini"
DEFAULT_GEMINI_MODEL = "google/gemini-2.0-flash-exp"
LLM_TEMPERATURE = 0.3
MAX_TOKENS = 2048
LLM_TIMEOUT = 30  # seconds

# ========================================
# APPLICATION SETTINGS
# ========================================
APP_TITLE = "Pharma Intelligence AI"
APP_ICON = "🧬"
APP_VERSION = "1.0.0"
APP_ENV = os.getenv("APP_ENV", "development")
DEBUG_MODE = os.getenv("DEBUG_MODE", "true").lower() == "true"

# ========================================
# SEARCH & RESULTS
# ========================================
MAX_SEARCH_RESULTS = int(os.getenv("MAX_SEARCH_RESULTS", "20"))
DEFAULT_RESULTS_PER_SOURCE = 10
MIN_RESULTS_THRESHOLD = 3

# ========================================
# CACHING
# ========================================
ENABLE_CACHING = os.getenv("ENABLE_CACHING", "true").lower() == "true"
CACHE_TTL_HOURS = int(os.getenv("CACHE_TTL_HOURS", "24"))
CACHE_DIR = Path("data/cache")
CACHE_DIR.mkdir(parents=True, exist_ok=True)

# ========================================
# OUTPUT DIRECTORIES
# ========================================
OUTPUT_DIR = Path("outputs")
OUTPUT_DIR.mkdir(exist_ok=True)

DATA_DIR = Path("data")
DATA_DIR.mkdir(exist_ok=True)

MOCK_DATA_DIR = Path("data/mock_data")
MOCK_DATA_DIR.mkdir(parents=True, exist_ok=True)

# ========================================
# LOGGING
# ========================================
LOG_LEVEL = "DEBUG" if DEBUG_MODE else "INFO"
LOG_FORMAT = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
LOG_FILE = "pharma_ai.log"

# ========================================
# AGENT CONFIGURATION
# ========================================
AGENT_CONFIG = {
    "verbose": DEBUG_MODE,
    "allow_delegation": True,
    "max_iterations": 10,
    "timeout": 120  # seconds
}

# ========================================
# STREAMLIT CONFIGURATION
# ========================================
STREAMLIT_CONFIG = {
    "page_title": APP_TITLE,
    "page_icon": APP_ICON,
    "layout": "wide",
    "initial_sidebar_state": "expanded"
}

# ========================================
# VALIDATION
# ========================================
def validate_config():
    """Validate critical configuration settings."""
    errors = []
    
    if not (OPENAI_API_KEY or GOOGLE_API_KEY):
        errors.append("Neither OPENAI_API_KEY nor GOOGLE_API_KEY is set in .env file")
    
    if errors:
        raise ValueError(f"Configuration errors:\n" + "\n".join(f"- {e}" for e in errors))

# Validate on import - only warn, don't fail
# This allows the app to start and show configuration errors to the user
if __name__ != "__main__":
    try:
        validate_config()
    except ValueError as e:
        # Log warning but don't prevent import
        import warnings
        warnings.warn(str(e), UserWarning)