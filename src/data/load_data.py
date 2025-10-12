"""
Load and cache the cleaned Airbnb dataset.

This module provides `load_data()` which reads the joined cleaned tables
`clean.listings_features` and `clean.reviews_summary` from Postgres and caches
the result as a parquet file on disk for faster subsequent access.

Missing values are standardized to np.nan and numeric columns are coerced
to floats to prevent NAType errors downstream.

The cache file path is defined by `CACHE_FILE` and defaults to
`data/processed/airbnb_clean.parquet`.
"""

import os
import pandas as pd
from sqlalchemy import create_engine
from src.config import ENGINE_URL, PROCESSED_DATA_DIR
from src.features.build_features import build_features

CACHE_FILE = os.path.join(PROCESSED_DATA_DIR, "airbnb_clean.parquet")


def load_data(use_cache=True) -> pd.DataFrame:
    """
    Load Airbnb clean data from Postgres or cached parquet. Applies feature engineering
    and ensures numeric columns and missing values are safe for ML pipelines.

    Parameters
    ----------
    use_cache : bool
        If True and parquet exists, load from cache.

    Returns
    -------
    pd.DataFrame
        Cleaned Airbnb dataset with consistent numeric types and missing values.
    """
    if use_cache and os.path.exists(CACHE_FILE):
        df = pd.read_parquet(CACHE_FILE)
    else:
        engine = create_engine(ENGINE_URL)
        query = """
        SELECT l.*, r.*
        FROM clean.listings_features AS l
        LEFT JOIN clean.reviews_summary AS r
            ON r.listing_id = l.id;
        """
        df = pd.read_sql(query, engine)
        df = build_features(df)

        os.makedirs(PROCESSED_DATA_DIR, exist_ok=True)
        df.to_parquet(CACHE_FILE, index=False)

    
    for col in df.columns:
        if pd.api.types.is_numeric_dtype(df[col]) or df[col].dtype.name in ["Int64", "UInt8", "Int32"]:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    df = df.replace({pd.NA: pd.NA, None: pd.NA})
    df = df.fillna(value=pd.NA)

    return df
