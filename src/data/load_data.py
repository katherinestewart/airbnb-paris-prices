import os
import pandas as pd
from sqlalchemy import create_engine
from src.config import ENGINE_URL, PROCESSED_DATA_DIR

CACHE_FILE = os.path.join(PROCESSED_DATA_DIR, "airbnb_clean.parquet")

def load_data(use_cache=True):
    """
    Load Airbnb clean data from Postgres or cached parquet.

    Parameters:
    - use_cache (bool): If True and parquet exists, load from cache.

    Returns:
    - pd.DataFrame
    """
    if use_cache and os.path.exists(CACHE_FILE):
        df = pd.read_parquet(CACHE_FILE)
        return df

    engine = create_engine(ENGINE_URL)
    query = """
    SELECT l.*, r.*
    FROM clean.listings_features AS l
    LEFT JOIN clean.reviews_summary AS r
        ON r.listing_id = l.id;
    """
    df = pd.read_sql(query, engine)

    os.makedirs(PROCESSED_DATA_DIR, exist_ok=True)
    df.to_parquet(CACHE_FILE, index=False)

    return df
