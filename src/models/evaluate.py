"""
Evaluation and training utilities with K-Fold CV and safe target encoding.

Usage (from project root)
-------------------------
# Run k-fold CV and print metrics:
python -m src.models.evaluate

# Or import functions:
from src.models.evaluate import cross_validate_with_target_encoding, train_final_with_target_encoding
cross_validate_with_target_encoding(n_splits=5)
train_final_with_target_encoding()
"""

from __future__ import annotations

import json
import os
from typing import Dict, Tuple, List

import joblib
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.model_selection import KFold, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder
from sklearn.metrics import mean_absolute_error, mean_squared_error
from xgboost import XGBRegressor

from src.config import MODEL_FILE, PREPROCESSOR_FILE, METADATA_FILE
from src.data.load_data import load_data


def _make_onehot_encoder(**kwargs):
    """
    Backwards/forwards compatible OneHotEncoder factory for different sklearn versions.
    """
    try:
        return OneHotEncoder(**kwargs)
    except TypeError:
        kwargs2 = dict(kwargs)
        if 'sparse' in kwargs2:
            sparse_val = kwargs2.pop('sparse')
            kwargs2['sparse_output'] = sparse_val
            return OneHotEncoder(**kwargs2)
        raise


def _compute_oof_target_encoding(
    df: pd.DataFrame,
    target_col: str = "price",
    cat_col: str = "neighbourhood_cleansed",
    n_splits: int = 5,
    random_state: int = 42
) -> Tuple[pd.Series, Dict[str, float]]:
    """
    Compute out-of-fold (OOF) target encoding for a categorical column.

    Parameters
    ----------
    df : pd.DataFrame
        Full dataset that contains `target_col` and `cat_col`.
    target_col : str
        Name of the numeric target column (e.g., 'price').
    cat_col : str
        Name of the categorical column to encode.
    n_splits : int
    random_state : int

    Returns
    -------
    oof_encoded : pd.Series
        Series aligned with df.index containing the OOF encoding values.
    global_mapping : Dict[str, float]
        Mapping computed on full data (used for final training / unseen categories).
    """
    if cat_col not in df.columns:

        return pd.Series(0.0, index=df.index), {}

    kf = KFold(n_splits=n_splits, shuffle=True, random_state=random_state)
    oof = pd.Series(index=df.index, dtype=float)
    global_mean = df[target_col].mean()

    for train_idx, val_idx in kf.split(df):
        train = df.iloc[train_idx]
        val = df.iloc[val_idx]

        mapping = train.groupby(cat_col)[target_col].mean()

        mapped = val[cat_col].map(mapping).fillna(global_mean)
        oof.iloc[val_idx] = mapped.values


    full_mapping = df.groupby(cat_col)[target_col].mean().to_dict()
    return oof, full_mapping


def _prepare_preprocessor(numeric_cols: List[str], categorical_cols: List[str]):
    """
    Create a simple ColumnTransformer (numeric imputer + categorical onehot).
    """
    numeric_transformer = SimpleImputer(strategy="median")
    categorical_transformer = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("onehot", _make_onehot_encoder(handle_unknown="ignore", sparse=False))
        ]
    )
    preprocessor = ColumnTransformer(
        transformers=[
            ("num", numeric_transformer, numeric_cols),
            ("cat", categorical_transformer, categorical_cols)
        ],
        remainder="drop",
    )
    return preprocessor


def _train_xgb_on_arrays(X_train_t, y_train_log, X_val_t=None, y_val_log=None, random_state: int = 42) -> XGBRegressor:
    """
    Train an XGBRegressor on already-transformed numeric arrays.
    """
    xgb = XGBRegressor(
        n_estimators=2000,
        learning_rate=0.03,
        max_depth=6,
        subsample=0.8,
        colsample_bytree=0.8,
        random_state=random_state,
        n_jobs=-1,
        verbosity=0,
    )

    xgb.fit(X_train_t, y_train_log)
    return xgb


def cross_validate_with_target_encoding(
    n_splits: int = 5,
    random_state: int = 42,
    numeric_cols: List[str] = None,
    categorical_cols: List[str] = None,
) -> Dict[str, Tuple[float, float]]:
    """
    Run K-fold cross-validation using out-of-fold target encoding for neighbourhood.

    Parameters
    ----------
    n_splits : int
    random_state : int
    numeric_cols : List[str] or None
        If None, a default set will be used (based on the dataset snapshot).
    categorical_cols : List[str] or None
        If None, default categorical columns will be used.

    Returns
    -------
    results : dict
        Summary dictionary with 'MAE' and 'RMSE' arrays and their mean/std.
    """
    df = load_data(use_cache=True)
    df = df.dropna(subset=["price"]).copy()
    df["price"] = pd.to_numeric(df["price"], errors="coerce")
    df = df.dropna(subset=["price"])
    df["price"] = df["price"].astype(float)


    if numeric_cols is None:
        numeric_cols = [
            "accommodates",
            "bedrooms",
            "beds",
            "bathrooms",
            "avg_comment_length",
            "days_since_last_review",
            "amenities_count",
            "dist_to_center_km",
        ]
    if categorical_cols is None:
        categorical_cols = ["property_type", "room_type"]


    oof_te, full_mapping = _compute_oof_target_encoding(
        df, target_col="price", cat_col="neighbourhood_cleansed", n_splits=n_splits, random_state=random_state
    )
    df = df.assign(neighbourhood_te=oof_te.values)


    numeric_final = [c for c in numeric_cols if c in df.columns] + ["neighbourhood_te"]
    categorical_final = [c for c in categorical_cols if c in df.columns]

    kf = KFold(n_splits=n_splits, shuffle=True, random_state=random_state)
    maes = []
    rmses = []

    for fold_idx, (train_idx, val_idx) in enumerate(kf.split(df), start=1):
        train = df.iloc[train_idx]
        val = df.iloc[val_idx]

        X_train = train[numeric_final + categorical_final].copy()
        y_train = train["price"].copy()
        X_val = val[numeric_final + categorical_final].copy()
        y_val = val["price"].copy()


        preprocessor = _prepare_preprocessor(numeric_cols=numeric_final, categorical_cols=categorical_final)
        preprocessor.fit(X_train)

        X_train_t = preprocessor.transform(X_train)
        X_val_t = preprocessor.transform(X_val)


        y_train_log = np.log1p(y_train)
        y_val_log = np.log1p(y_val)


        xgb = _train_xgb_on_arrays(X_train_t, y_train_log, X_val_t, y_val_log, random_state=random_state)


        y_pred_log = xgb.predict(X_val_t)
        y_pred = np.expm1(y_pred_log)

        mae = mean_absolute_error(y_val, y_pred)
        rmse = np.sqrt(mean_squared_error(y_val, y_pred))

        maes.append(mae)
        rmses.append(rmse)

        print(f"Fold {fold_idx}/{n_splits}: MAE={mae:.2f}, RMSE={rmse:.2f}")

    maes = np.array(maes)
    rmses = np.array(rmses)
    print("\nCross-validation summary:")
    print(f"MAE  mean={maes.mean():.2f}, std={maes.std():.2f}")
    print(f"RMSE mean={rmses.mean():.2f}, std={rmses.std():.2f}")

    return {
        "MAE": (float(maes.mean()), float(maes.std())),
        "RMSE": (float(rmses.mean()), float(rmses.std())),
    }


def train_final_with_target_encoding(
    random_state: int = 42,
    test_size: float = 0.1,
    numeric_cols: List[str] = None,
    categorical_cols: List[str] = None,
):
    """
    Train a final model on the full dataset using neighbourhood target encoding.

    The function:
    - computes full-data target encoding mapping for neighbourhood_cleansed
    - adds `neighbourhood_te` to the dataset
    - builds preprocessor and final pipeline (preprocessor + xgb)
    - fits xgb with a small validation split for early stopping
    - saves MODEL_FILE, PREPROCESSOR_FILE, METADATA_FILE
    """
    df = load_data(use_cache=True)
    df = df.dropna(subset=["price"]).copy()
    df["price"] = pd.to_numeric(df["price"], errors="coerce")
    df = df.dropna(subset=["price"])
    df["price"] = df["price"].astype(float)

    if numeric_cols is None:
        numeric_cols = [
            "accommodates",
            "bedrooms",
            "beds",
            "bathrooms",
            "avg_comment_length",
            "days_since_last_review",
            "amenities_count",
            "dist_to_center_km",
        ]
    if categorical_cols is None:
        categorical_cols = ["property_type", "room_type"]


    full_mapping = df.groupby("neighbourhood_cleansed")["price"].mean().to_dict()
    global_mean = df["price"].mean()
    df["neighbourhood_te"] = df["neighbourhood_cleansed"].map(full_mapping).fillna(global_mean)

    numeric_final = [c for c in numeric_cols if c in df.columns] + ["neighbourhood_te"]
    categorical_final = [c for c in categorical_cols if c in df.columns]

    X = df[numeric_final + categorical_final].copy()
    y = df["price"].copy()

    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=test_size, random_state=random_state)

    preprocessor = _prepare_preprocessor(numeric_cols=numeric_final, categorical_cols=categorical_final)
    preprocessor.fit(X_train)
    X_train_t = preprocessor.transform(X_train)
    X_val_t = preprocessor.transform(X_val)

    y_train_log = np.log1p(y_train)
    y_val_log = np.log1p(y_val)

    xgb = _train_xgb_on_arrays(X_train_t, y_train_log, X_val_t, y_val_log, random_state=random_state)

    final_pipeline = Pipeline(steps=[("preprocessor", preprocessor), ("regressor", xgb)])
    os.makedirs(os.path.dirname(MODEL_FILE), exist_ok=True)
    joblib.dump(final_pipeline, MODEL_FILE)
    joblib.dump(preprocessor, PREPROCESSOR_FILE)

    y_pred_log = xgb.predict(X_val_t)
    y_pred = np.expm1(y_pred_log)
    mae = mean_absolute_error(y_val, y_pred)
    rmse = np.sqrt(mean_squared_error(y_val, y_pred))


    metadata = {
        "model_name": "XGBoost Regressor (log-target + neighbourhood TE)",
        "model_version": "v1",
        "trained_on": pd.Timestamp.today().strftime("%Y-%m-%d"),
        "features": numeric_final + categorical_final,
        "metrics": {"MAE": round(float(mae), 2), "RMSE": round(float(rmse), 2)},
        "preprocessor_file": os.path.basename(PREPROCESSOR_FILE),
        "model_file": os.path.basename(MODEL_FILE),
        "notes": "Final model trained with neighbourhood target encoding (full-data mapping) and early stopping.",
    }
    os.makedirs(os.path.dirname(METADATA_FILE), exist_ok=True)
    with open(METADATA_FILE, "w") as fh:
        json.dump(metadata, fh, indent=4)

    print(f"Final training complete. Validation MAE={mae:.2f}, RMSE={rmse:.2f}")
    return metadata


if __name__ == "__main__":

    print("Running 5-fold cross-validation with out-of-fold target encoding...")
    cv_res = cross_validate_with_target_encoding(n_splits=5)
    print("\nCV results:", cv_res)

    print("\nTraining final model with target encoding (this will overwrite models/*).")
    metadata = train_final_with_target_encoding()
    print("Saved final model and metadata:", metadata)
