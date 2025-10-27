"""
Central configuration for the project.

This module centralizes environment-independent constants and derived paths
used throughout the codebase (database connection string, data and model paths).

Constants
---------
ENGINE_URL : str
    SQLAlchemy connection URL for the project's Postgres database.
BASE_DIR : str
    Absolute path to the project `src` parent directory.
DATA_DIR, RAW_DATA_DIR, PROCESSED_DATA_DIR : str
    Paths to data folders.
MODELS_DIR, MODEL_FILE, PREPROCESSOR_FILE, METADATA_FILE : str
    Paths to model artifacts and metadata.
"""

import os

DB_USER = "airbnb"
DB_PASSWORD = "airbnb"
DB_HOST = "localhost"
DB_PORT = 5432
DB_NAME = "airbnb"

ENGINE_URL = f"postgresql://{DB_USER}:{DB_PASSWORD}@{DB_HOST}:{DB_PORT}/{DB_NAME}"

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(BASE_DIR, "data")
RAW_DATA_DIR = os.path.join(DATA_DIR, "raw")
PROCESSED_DATA_DIR = os.path.join(DATA_DIR, "processed")

MODELS_DIR = os.path.join(BASE_DIR, "models")
MODEL_FILE = os.path.join(MODELS_DIR, "short_term_only.joblib")
PREPROCESSOR_FILE = os.path.join(MODELS_DIR, "preprocessing_v1.pkl")
METADATA_FILE = os.path.join(MODELS_DIR, "short_term_only.metadata.json")
