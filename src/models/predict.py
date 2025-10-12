"""
Airbnb Paris Prices - Safe Prediction Helper

Provides a single function `predict_price(input_dict)` that:
- Loads the persisted model and metadata.
- Safely computes 'neighbourhood_te' from cached training data if required.
- Aligns input columns, fills missing values with np.nan, and coerces numerics.
- Returns a float nightly price prediction in EUR.
"""

from __future__ import annotations
import json
from pathlib import Path
from typing import Dict, Any

import joblib
import numpy as np
import pandas as pd

from src.config import MODEL_FILE, METADATA_FILE
from src.data.load_data import load_data


try:
    _MODEL = joblib.load(MODEL_FILE)
except Exception:
    _MODEL = None


try:
    _METADATA = json.loads(Path(METADATA_FILE).read_text(encoding="utf-8")) if Path(METADATA_FILE).exists() else {}
except Exception:
    _METADATA = {}


def _compute_neighbourhood_te(input_df: pd.DataFrame) -> pd.DataFrame:
    """
    Computes safe neighbourhood target encoding for the model.
    Uses cached training data mean prices per neighbourhood.
    Missing/unseen neighbourhoods fallback to global mean.
    """
    if 'neighbourhood_te' in (_METADATA.get('features') or []):
        if 'neighbourhood_te' not in input_df.columns and 'neighbourhood_cleansed' in input_df.columns:
            try:
                df_cache = load_data(use_cache=True)
                mapping = df_cache.groupby('neighbourhood_cleansed')['price'].mean().to_dict()
                global_mean = float(df_cache['price'].mean())
            except Exception:
                mapping = {}
                global_mean = 0.0

            input_df['neighbourhood_te'] = input_df['neighbourhood_cleansed'].apply(
                lambda x: float(mapping.get(x, global_mean)) if pd.notna(x) else float(global_mean)
            )
    return input_df


def _align_and_clean(input_df: pd.DataFrame) -> pd.DataFrame:
    """
    Aligns input to model features, fills missing columns with np.nan,
    and coerces numeric-like columns to float to prevent NAType errors.
    """
    features = _METADATA.get("features")
    if not isinstance(features, list):
        return input_df.replace({pd.NA: np.nan, None: np.nan})

    for f in features:
        if f not in input_df.columns:
            input_df[f] = np.nan

    input_df = input_df[features].copy()
    input_df = input_df.replace({pd.NA: np.nan, None: np.nan})

    for col in input_df.columns:
        if pd.api.types.is_object_dtype(input_df[col]):
            coerced = pd.to_numeric(input_df[col], errors="coerce")
            if coerced.notna().sum() > 0:
                input_df[col] = coerced

    return input_df


def predict_price(input_dict: Dict[str, Any]) -> float:
    """
    Predict nightly price for a single Airbnb input dictionary.

    Returns
    -------
    float
        Predicted nightly price in EUR.
    """
    if _MODEL is None:
        raise RuntimeError(
            f"Model file not found at {MODEL_FILE}. Run `make train` first."
        )

    input_df = pd.DataFrame([input_dict])
    input_df = _compute_neighbourhood_te(input_df)
    input_df = _align_and_clean(input_df)

    preds = _MODEL.predict(input_df)

    model_name = (_METADATA.get("model_name") or "").lower()
    val = float(preds[0])


    if "log" in model_name:
        val = float(np.expm1(preds[0]))

    return val


if __name__ == "__main__":
    example = {
        'neighbourhood_cleansed': 'Louvre',
        'property_type': 'Entire rental unit',
        'room_type': 'Entire home/apt',
        'accommodates': 2,
        'bedrooms': 1,
        'beds': 1,
        'bathrooms': 1,
        'avg_comment_length': 100.0,
        'days_since_last_review': 30
    }
    print("Predicted price:", predict_price(example))
