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
MODEL_FILE = os.path.join(MODELS_DIR, "xgb_model_v1.pkl")
PREPROCESSOR_FILE = os.path.join(MODELS_DIR, "preprocessing_v1.pkl")
METADATA_FILE = os.path.join(MODELS_DIR, "metadata.json")