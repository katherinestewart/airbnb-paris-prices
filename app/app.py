from __future__ import annotations

import json
import logging
import os
from pathlib import Path
from typing import Any, Dict, List, Optional

import joblib
import numpy as np
import pandas as pd
from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from sqlalchemy import create_engine

LOG = logging.getLogger("airbnb_predict_api")
logging.basicConfig(level=logging.INFO)

ARTIFACTS_DIR = Path(os.getenv("ARTIFACTS_DIR", "models"))
MODEL_FILE = Path(os.getenv("MODEL_FILE", ARTIFACTS_DIR / "model_monotone.joblib"))
META_FILE = Path(os.getenv("META_FILE", ARTIFACTS_DIR / "model_monotone.json"))
DB_URL = os.getenv("AIRBNB_DB_URL", "postgresql://airbnb:airbnb@localhost:5432/airbnb")

SQL_BASE = """
SELECT l.*, r.n_reviews, r.first_review, r.last_review,
       r.avg_comment_length, r.days_since_last_review
FROM clean.listings_features l
LEFT JOIN clean.reviews_summary r ON r.listing_id = l.id
"""

BATHROOMS_UPPER_CAP = float(os.getenv("BATHROOMS_UPPER_CAP", 4.0))
BEDROOMS_UPPER_CAP = float(os.getenv("BEDROOMS_UPPER_CAP", 4.0))
BEDS_UPPER_CAP = float(os.getenv("BEDS_UPPER_CAP", 4.0))
ACCOMMODATES_UPPER_CAP = float(os.getenv("ACCOMMODATES_UPPER_CAP", 8.0))

app = FastAPI(title="Airbnb Price Predictor API", version="1.0")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

_model = None
_feature_columns: Optional[List[str]] = None
_meta: Dict[str, Any] = {}

class PredictRequest(BaseModel):
    rows: List[Dict[str, Any]]

class PredictResponse(BaseModel):
    predictions: List[float]
    model_meta: Dict[str, Any]

@app.on_event("startup")
def load_artifacts():
    global _model, _feature_columns, _meta
    LOG.info("Loading model and metadata from %s and %s...", MODEL_FILE, META_FILE)
    try:
        if not MODEL_FILE.exists():
            raise FileNotFoundError(f"Model file not found: {MODEL_FILE}")
        _model = joblib.load(MODEL_FILE)
        LOG.info("Model loaded successfully.")
        if META_FILE.exists():
            with open(META_FILE, "r", encoding="utf-8") as fh:
                _meta = json.load(fh) or {}
        else:
            _meta = {}
            LOG.warning("Metadata file not found: %s. Proceeding without metadata.", META_FILE)

        _feature_columns = _meta.get("feature_columns") or _meta.get("features")
        if _feature_columns is None:
            LOG.warning("No feature column list found in metadata; inference will attempt best-effort.")
    except Exception as exc:
        LOG.exception("Failed to load artifacts: %s", exc)
        _model = None
        _feature_columns = None
        _meta = {}

def apply_engineered_features_to_row(row: Dict[str, Any]) -> Dict[str, Any]:
    out = dict(row)

    try:
        b = out.get("bathrooms", None)
        b = 0.0 if b is None else float(b)
        b = max(0.0, min(b, BATHROOMS_UPPER_CAP))
        out["bathrooms"] = b
    except Exception:
        out["bathrooms"] = 0.0

    try:
        bd = out.get("bedrooms", None)
        bd = 0.0 if bd is None else float(bd)
        bd = max(0.0, min(bd, BEDROOMS_UPPER_CAP))
        out["bedrooms"] = bd
    except Exception:
        out["bedrooms"] = 0.0

    try:
        bs = out.get("beds", None)
        bs = 0.0 if bs is None else float(bs)
        bs = max(0.0, min(bs, BEDS_UPPER_CAP))
        out["beds"] = bs
    except Exception:
        out["beds"] = 0.0

    try:
        ac = out.get("accommodates", None)
        ac = 1.0 if ac is None else float(ac)
        if ac < 1.0:
            ac = 1.0
        ac = min(ac, ACCOMMODATES_UPPER_CAP)
        out["accommodates"] = ac
    except Exception:
        out["accommodates"] = 1.0

    out["bathrooms_per_bedroom"] = float(out["bathrooms"]) / float(out["bedrooms"]) if out["bedrooms"] else 0.0
    out["bathrooms_per_accommodates"] = float(out["bathrooms"]) / float(out["accommodates"]) if out["accommodates"] else 0.0

    return out

def prepare_rows_for_model(rows: List[Dict[str, Any]]) -> pd.DataFrame:
    processed = [apply_engineered_features_to_row(r) for r in rows]
    df = pd.DataFrame(processed)
    if _feature_columns:
        for c in _feature_columns:
            if c not in df.columns:
                df[c] = np.nan
        df = df[_feature_columns]
    return df

def sanitize_and_predict_dataframe(df_in: pd.DataFrame):
    global _model, _feature_columns
    if _model is None:
        raise RuntimeError("Model not loaded")

    df = df_in.copy()
    if _feature_columns:
        for c in _feature_columns:
            if c not in df.columns:
                df[c] = np.nan
        df = df[_feature_columns]

    cat_cols = [c for c in ["neighbourhood_cleansed", "property_type_slim", "room_type"] if c in df.columns]
    for c in cat_cols:
        df[c] = df[c].astype(object).where(df[c].notnull(), None)

    preds = _model.predict(df)
    preds_list = [float(p) for p in preds]
    model_meta = {"saved_at": _meta.get("saved_at"), "cv_mae_mean": _meta.get("cv_mae_mean")}
    return preds_list, model_meta

@app.get("/health")
def health():
    return {"status": "ok", "model_loaded": _model is not None}

@app.get("/metadata")
def metadata():
    return {
        "model_meta": {
            "saved_at": _meta.get("saved_at"),
            "cv_mae_mean": _meta.get("cv_mae_mean"),
            "train_rows": _meta.get("train_rows"),
            "test_rows": _meta.get("test_rows"),
        },
        "feature_columns": _feature_columns,
    }

@app.post("/predict", response_model=PredictResponse)
def predict(req: PredictRequest):
    if not req.rows:
        raise HTTPException(status_code=400, detail="Request must contain at least one row in 'rows'")
    try:
        df = prepare_rows_for_model(req.rows)
        preds, meta = sanitize_and_predict_dataframe(df)
        return PredictResponse(predictions=preds, model_meta=meta)
    except HTTPException:
        raise
    except Exception as exc:
        LOG.exception("Prediction error: %s", exc)
        raise HTTPException(status_code=500, detail=f"Prediction failed: {exc}")

@app.get("/predict_from_db")
def predict_from_db(id: int = Query(..., description="Listing id to fetch & predict")):
    try:
        listing_id = int(id)
    except Exception:
        raise HTTPException(status_code=400, detail="Invalid id parameter")

    engine = create_engine(DB_URL)
    sql = SQL_BASE.rstrip().rstrip(";") + "\nWHERE l.id = %(listing_id)s;"
    try:
        df = pd.read_sql(sql, engine, params={"listing_id": listing_id})
    except Exception as exc:
        LOG.exception("DB read failed: %s", exc)
        raise HTTPException(status_code=500, detail=f"DB read failed: {exc}")

    if df.empty:
        raise HTTPException(status_code=404, detail=f"Listing id {listing_id} not found")

    df = df.apply(lambda r: pd.Series(apply_engineered_features_to_row(r.to_dict())), axis=1)

    try:
        preds, meta = sanitize_and_predict_dataframe(df)
    except Exception as exc:
        LOG.exception("Prediction failed after DB fetch: %s", exc)
        raise HTTPException(status_code=500, detail=f"Prediction failed: {exc}")

    return {"id": listing_id, "predicted_price": float(preds[0]), "model_meta": meta}
