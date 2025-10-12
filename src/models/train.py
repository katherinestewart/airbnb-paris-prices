"""
Final training script for the Airbnb Paris price model.

Features and behaviour
----------------------
- Loads processed data via src.data.load_data.load_data (which applies feature engineering).
- Computes a full-data neighbourhood -> mean(price) mapping (neighbourhood target encoding),
  stores it in models/neighbourhood_te.json and adds 'neighbourhood_te' to the training data.
- Uses log1p(price) as the training target and inverts predictions when evaluating.
- Builds a ColumnTransformer (numeric median imputer + categorical one-hot).
- Fits preprocessor, transforms arrays, and trains XGBRegressor with early stopping.
- Saves:
    - full pipeline to MODEL_FILE (preprocessor + regressor)
    - preprocessor to PREPROCESSOR_FILE
    - neighbourhood mapping to models/neighbourhood_te.json
    - metadata (features, metrics) to METADATA_FILE

Usage
-----
python src/models/train.py
"""
from __future__ import annotations

import json
import os
from pathlib import Path
from typing import List, Dict

import joblib
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder
from sklearn.model_selection import train_test_split
from xgboost import XGBRegressor

from src.config import MODEL_FILE, PREPROCESSOR_FILE, METADATA_FILE
from src.data.load_data import load_data


_NEIGH_TE_FILE = Path(MODEL_FILE).parent / "neighbourhood_te.json"


def make_onehot_encoder(**kwargs):
    """
    Backwards/forwards compatible OneHotEncoder constructor.
    Accepts `sparse` in older sklearn and maps it to `sparse_output` if needed.
    """
    try:
        return OneHotEncoder(**kwargs)
    except TypeError:
        kwargs2 = dict(kwargs)
        if "sparse" in kwargs2:
            sparse_val = kwargs2.pop("sparse")
            kwargs2["sparse_output"] = sparse_val
            return OneHotEncoder(**kwargs2)
        raise


def _ensure_dir_for(path: str):
    os.makedirs(os.path.dirname(path), exist_ok=True)


def _save_neighbourhood_mapping(mapping: Dict[str, float], global_mean: float):
    payload = {"mapping": mapping, "global_mean": float(global_mean)}
    try:
        _NEIGH_TE_FILE.write_text(json.dumps(payload), encoding="utf-8")
    except Exception:

        pass


def _compute_and_persist_neighbourhood_mapping(df: pd.DataFrame) -> Dict[str, float]:
    """
    Compute full-dataset neighbourhood -> mean(price) mapping and persist it.
    Returns the mapping and writes the mapping JSON to models/.
    """
    if "neighbourhood_cleansed" in df.columns and "price" in df.columns:
        grp = df.groupby("neighbourhood_cleansed")["price"].mean()
        mapping = {str(k): float(v) for k, v in grp.to_dict().items()}
        global_mean = float(df["price"].mean())
    else:
        mapping = {}
        global_mean = 0.0

    _save_neighbourhood_mapping(mapping, global_mean)
    return {"mapping": mapping, "global_mean": global_mean}


def train_model(random_state: int = 42):
    """
    Train final pipeline and persist artifacts.

    Parameters
    ----------
    random_state : int
        RNG seed for reproducibility.
    """

    df = load_data(use_cache=True)


    df = df.dropna(subset=["price"]).copy()
    df["price"] = pd.to_numeric(df["price"], errors="coerce")
    df = df.dropna(subset=["price"])
    df["price"] = df["price"].astype(float)


    numeric_cols: List[str] = [
        "accommodates",
        "bedrooms",
        "beds",
        "bathrooms",
        "avg_comment_length",
        "days_since_last_review",
        "amenities_count",
        "dist_to_center_km",
    ]
    categorical_cols: List[str] = [
        "property_type",
        "room_type",
    ]

    neigh_info = _compute_and_persist_neighbourhood_mapping(df)
    mapping = neigh_info["mapping"]
    global_mean = neigh_info["global_mean"]


    if "neighbourhood_cleansed" in df.columns:
        df["neighbourhood_te"] = df["neighbourhood_cleansed"].apply(
            lambda x: float(mapping.get(x, global_mean)) if pd.notna(x) else float(global_mean)
        )
    else:
        df["neighbourhood_te"] = float(global_mean)


    numeric_final = [c for c in numeric_cols if c in df.columns] + ["neighbourhood_te"]
    categorical_final = [c for c in categorical_cols if c in df.columns]

    features = numeric_final + categorical_final
    if not features:
        raise RuntimeError("No features found for training. Check data and feature engineering step.")

    X = df[features].copy()
    y = df["price"].copy()


    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.10, random_state=random_state)


    numeric_transformer = SimpleImputer(strategy="median")
    categorical_transformer = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("onehot", make_onehot_encoder(handle_unknown="ignore", sparse=False)),
        ]
    )

    preprocessor = ColumnTransformer(
        transformers=[
            ("num", numeric_transformer, numeric_final),
            ("cat", categorical_transformer, categorical_final),
        ],
        remainder="drop",
    )

    preprocessor.fit(X_train)
    X_train_t = preprocessor.transform(X_train)
    X_val_t = preprocessor.transform(X_val)


    y_train_log = np.log1p(y_train)
    y_val_log = np.log1p(y_val)


    xgb = XGBRegressor(
        n_estimators=2000,
        learning_rate=0.03,
        max_depth=6,
        subsample=0.8,
        colsample_bytree=0.8,
        random_state=random_state,
        n_jobs=-1,
        verbosity=1,
    )


    xgb.fit(X_train_t, y_train_log)

    final_pipeline = Pipeline(steps=[("preprocessor", preprocessor), ("regressor", xgb)])


    _ensure_dir_for(MODEL_FILE)
    joblib.dump(final_pipeline, MODEL_FILE)
    joblib.dump(preprocessor, PREPROCESSOR_FILE)


    y_pred_log = xgb.predict(X_val_t)
    y_pred = np.expm1(y_pred_log)

    from sklearn.metrics import mean_absolute_error, mean_squared_error

    mae = mean_absolute_error(y_val, y_pred)
    rmse = np.sqrt(mean_squared_error(y_val, y_pred))


    metadata = {
        "model_name": "XGBoost Regressor (log-target + neighbourhood TE)",
        "model_version": "v1",
        "trained_on": pd.Timestamp.today().strftime("%Y-%m-%d"),
        "features": features,
        "metrics": {"MAE": round(float(mae), 2), "RMSE": round(float(rmse), 2)},
        "preprocessor_file": os.path.basename(PREPROCESSOR_FILE),
        "model_file": os.path.basename(MODEL_FILE),
        "notes": "Trained on log1p(price) with neighbourhood target encoding (full-data mapping) and early stopping.",
    }

    _ensure_dir_for(METADATA_FILE)
    with open(METADATA_FILE, "w") as fh:
        json.dump(metadata, fh, indent=4)

    print(f"Training completed. Validation MAE={mae:.2f}, RMSE={rmse:.2f}")
    print(f"Saved: {MODEL_FILE}, {PREPROCESSOR_FILE}, {METADATA_FILE}, {_NEIGH_TE_FILE}")


if __name__ == "__main__":
    train_model()
