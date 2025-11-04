"""
Central configuration for the Airbnb price prediction project.

This configuration centralizes constants and paths for:
- Database connections
- Data folders
- Model artifacts (model, preprocessor, metadata)

It automatically loads environment variables when available (via .env),
falling back to default local development values.
"""

import os
from pathlib import Path
from dotenv import load_dotenv

# --- Load environment variables from .env if present ---
load_dotenv()

# -------------------
# DATABASE CONFIGURATION
# -------------------
DB_USER = os.getenv("DB_USER", "airbnb")
DB_PASSWORD = os.getenv("DB_PASSWORD", "airbnb")
DB_HOST = os.getenv("DB_HOST", "localhost")
DB_PORT = int(os.getenv("DB_PORT", 5432))
DB_NAME = os.getenv("DB_NAME", "airbnb")

ENGINE_URL = f"postgresql://{DB_USER}:{DB_PASSWORD}@{DB_HOST}:{DB_PORT}/{DB_NAME}"

# -------------------
# PATHS CONFIGURATION
# -------------------
BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "data"
RAW_DATA_DIR = DATA_DIR / "raw"

MODELS_DIR = BASE_DIR / "models"
MODEL_FILE = MODELS_DIR / "model_monotone.joblib"
METADATA_FILE = MODELS_DIR / "model_monotone.json"

# -------------------
# OTHER SETTINGS
# -------------------
DEBUG = os.getenv("DEBUG", "true").lower() == "true"
PORT = int(os.getenv("PORT", 8000))
GCP_PROJECT = os.getenv("GCP_PROJECT", "")
GCP_REGION = os.getenv("GCP_REGION", "europe-west1")

# -------------------
# LOGGING
# -------------------
def print_config_summary():
    """Convenience summary for debugging."""
    print("\n[CONFIGURATION SUMMARY]")
    print(f"Database: {ENGINE_URL}")
    print(f"Model file: {MODEL_FILE}")
    print(f"Metadata file: {METADATA_FILE}")
    print(f"Running on port: {PORT}")
    print(f"Debug mode: {DEBUG}")
    if GCP_PROJECT:
        print(f"GCP project: {GCP_PROJECT} ({GCP_REGION})")
    print("-------------------------\n")

if DEBUG:
    print_config_summary()
