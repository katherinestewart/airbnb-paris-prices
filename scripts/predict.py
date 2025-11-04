from __future__ import annotations

import argparse
import json
import logging
import os
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional

import joblib
import numpy as np
import pandas as pd
from sqlalchemy import create_engine, text

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
LOG = logging.getLogger("predict_clean_db")

ARTIFACTS_DIR = Path(os.getenv("ARTIFACTS_DIR", "models"))
MODEL_FILE = Path(os.getenv("MODEL_FILE", ARTIFACTS_DIR / "model_monotone.joblib"))
META_FILE = Path(os.getenv("META_FILE", ARTIFACTS_DIR / "model_monotone.json"))
DB_URL = os.getenv("AIRBNB_DB_URL", "postgresql://airbnb:airbnb@localhost:5432/airbnb")

BATHROOMS_UPPER_CAP = float(os.getenv("BATHROOMS_UPPER_CAP", 4.0))

SQL_BASE = """
SELECT l.*, r.n_reviews, r.first_review, r.last_review,
       r.avg_comment_length, r.days_since_last_review
FROM clean.listings_features l
LEFT JOIN clean.reviews_summary r ON r.listing_id = l.id
"""

def _find_encoders(obj: Any) -> List[Any]:
    encs: List[Any] = []
    if obj is None:
        return encs

    if hasattr(obj, "transformers_"):
        for _name, transformer, _cols in getattr(obj, "transformers_", []):
            if transformer in ("drop", "passthrough", None):
                continue
            encs.extend(_find_encoders(transformer))
    elif hasattr(obj, "steps"):
        for _name, step in getattr(obj, "steps", []):
            encs.extend(_find_encoders(step))
    else:
        if hasattr(obj, "categories_"):
            encs.append(obj)
    return encs


def patch_encoder_categories(model: Any) -> None:
    """
    Replace pandas.NA/None inside encoder.categories_ with numpy.nan and ensure object dtype.
    This prevents sklearn internals from calling np.isnan on pandas.NA.
    """
    if model is None:
        return

    encs = _find_encoders(model)
    LOG.debug("Patching %d encoder(s)", len(encs))
    for enc in encs:
        try:
            fixed = []
            for cat_arr in enc.categories_:
                seq = list(cat_arr)
                seq_fixed = [np.nan if (v is pd.NA or v is None) else v for v in seq]
                fixed.append(np.array(seq_fixed, dtype=object))
            enc.categories_ = fixed
        except Exception as exc:
            LOG.debug("Skipping patch for encoder %s: %s", getattr(enc, "__class__", enc), exc)


def load_model_and_meta(model_path: Path = MODEL_FILE, meta_path: Path = META_FILE):
    """Load model and metadata JSON. Return (model, feature_columns, meta)."""
    model_path = Path(model_path)
    meta_path = Path(meta_path)

    if not model_path.exists():
        raise FileNotFoundError(f"Model file not found: {model_path}")

    LOG.info("Loading model from: %s", model_path)
    model = joblib.load(model_path)

    try:
        patch_encoder_categories(model)
    except Exception:
        LOG.exception("Patch of encoder categories failed (continuing).")

    LOG.info("Reading metadata from: %s", meta_path)
    meta: Dict[str, Any] = {}
    if meta_path.exists():
        meta = json.loads(meta_path.read_text(encoding="utf-8"))
    else:
        LOG.warning("Metadata file not found: %s. Proceeding without metadata.", meta_path)

    feature_columns = meta.get("feature_columns") or meta.get("features")
    if feature_columns is not None and not isinstance(feature_columns, list):
        raise KeyError("'feature_columns' or 'features' in metadata must be a list")

    return model, (list(feature_columns) if feature_columns is not None else None), meta


def apply_engineered_features_to_row(row: Dict[str, Any]) -> Dict[str, Any]:
    out = dict(row)

    try:
        b = out.get("bathrooms", None)
        if b is None:
            b = 0.0
        else:
            b = float(b)
        b = max(0.0, b)
        b = min(b, BATHROOMS_UPPER_CAP)
        out["bathrooms"] = b
    except Exception:
        out["bathrooms"] = 0.0

    try:
        bd = out.get("bedrooms", None)
        if bd is None:
            bd = 1.0
        else:
            bd = float(bd)
        if bd <= 0:
            bd = 1.0
        out["bedrooms"] = bd
    except Exception:
        out["bedrooms"] = 1.0

    try:
        ac = out.get("accommodates", None)
        if ac is None:
            ac = 1.0
        else:
            ac = float(ac)
        if ac <= 0:
            ac = 1.0
        out["accommodates"] = ac
    except Exception:
        out["accommodates"] = 1.0

    out["bathrooms_per_bedroom"] = float(out["bathrooms"]) / float(out["bedrooms"]) if out["bedrooms"] else 0.0
    out["bathrooms_per_accommodates"] = float(out["bathrooms"]) / float(out["accommodates"]) if out["accommodates"] else 0.0

    return out


def prepare_rows_for_model(rows: List[Dict[str, Any]], feature_columns: Optional[List[str]]):
    processed = [apply_engineered_features_to_row(r) for r in rows]
    df = pd.DataFrame(processed)

    if feature_columns:
        for c in feature_columns:
            if c not in df.columns:
                df[c] = np.nan
        df = df[feature_columns]

    return df


def _coerce_boolean_like_series(s: pd.Series) -> pd.Series:
    if s.dtype != object and not pd.api.types.is_string_dtype(s.dtype):
        return s

    def _norm(v):
        if v is None or (isinstance(v, float) and np.isnan(v)):
            return np.nan
        if isinstance(v, bool):
            return 1 if v else 0
        if isinstance(v, (int, float, np.integer, np.floating)) and not (isinstance(v, float) and np.isnan(v)):
            return v
        try:
            sv = str(v).strip().lower()
        except Exception:
            return np.nan
        if sv in {"t", "true", "1", "y", "yes", "on"}:
            return 1
        if sv in {"f", "false", "0", "n", "no", "off"}:
            return 0
        return None

    mapped = s.map(_norm)
    non_null = s.notna().sum()
    mapped_numeric = mapped.dropna().shape[0]
    if non_null == 0:
        return s
    if mapped_numeric / non_null >= 0.5:
        mapped = mapped.replace({None: np.nan})
        return pd.to_numeric(mapped, errors="coerce")
    return s


def _coerce_boolean_like_frame(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    for col in df.select_dtypes(include=["object", "string"]).columns:
        try:
            coerced = _coerce_boolean_like_series(df[col])
            if pd.api.types.is_numeric_dtype(coerced.dtype):
                df[col] = coerced
            else:
                df[col] = df[col].astype(object).where(df[col].notna(), np.nan)
        except Exception:
            df[col] = df[col].astype(object).where(df[col].notna(), np.nan)
    return df


def sanitize_inputs_for_sklearn(X: pd.DataFrame) -> pd.DataFrame:
    X = X.copy()
    X = X.replace({pd.NA: np.nan, None: np.nan})

    X = _coerce_boolean_like_frame(X)

    for col in X.select_dtypes(include=["string", "object"]):
        X[col] = X[col].astype(object).where(X[col].notna(), np.nan)

    for col in X.columns:
        if X[col].dtype == object:
            coerced = pd.to_numeric(X[col], errors="coerce")
            non_na_before = int(X[col].notna().sum())
            non_na_after = int(coerced.notna().sum())
            if non_na_before == 0 or (non_na_after / max(1, non_na_before)) >= 0.5:
                X[col] = coerced

    for col in X.columns:
        if pd.api.types.is_integer_dtype(X[col].dtype) and X[col].isnull().any():
            X[col] = X[col].astype(float)

    return X


def fetch_from_db(ids: List[int], db_url: str = DB_URL) -> pd.DataFrame:
    if not ids:
        raise ValueError("No listing ids provided for DB fetch")

    ids_int = [int(i) for i in ids]
    engine = create_engine(db_url)

    if len(ids_int) == 1:
        where = "WHERE l.id = %(listing_id)s"
        params = {"listing_id": ids_int[0]}
        sql = SQL_BASE.rstrip().rstrip(";") + "\n" + where + ";"
        LOG.info("Querying clean tables for id %s", ids_int[0])
        return pd.read_sql(sql, engine, params=params)
    else:
        where = "WHERE l.id IN (" + ",".join(map(str, ids_int)) + ")"
        sql = SQL_BASE.rstrip().rstrip(";") + "\n" + where + ";"
        LOG.info("Querying clean tables for ids: %s", ids_int)
        return pd.read_sql(sql, engine)


def read_from_csv(path: Path) -> pd.DataFrame:
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"CSV file not found: {path}")
    LOG.info("Reading input CSV: %s", path)
    return pd.read_csv(path)


def write_output(df: pd.DataFrame, out_path: Optional[Path]):
    if out_path:
        out_path = Path(out_path)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(out_path, index=False)
        LOG.info("Saved predictions to: %s", out_path)
    else:
        sys.stdout.write(df.to_csv(index=False))
        sys.stdout.flush()


def build_arg_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Load trained model and predict prices for listings (uses clean.* tables).")
    mode = p.add_mutually_exclusive_group(required=True)
    mode.add_argument("--from-db", action="store_true", help="Fetch rows from the clean DB tables (requires --id)")
    mode.add_argument("--from-csv", type=Path, help="Path to a CSV file with listing rows (optional)")

    p.add_argument("--id", dest="ids", action="append", help="Listing id to predict. Use multiple --id for multiple ids.")
    p.add_argument("--model", type=Path, default=MODEL_FILE, help="Path to joblib model file")
    p.add_argument("--meta", type=Path, default=META_FILE, help="Path to metadata json")
    p.add_argument("--output", type=Path, help="Output CSV path. If omitted prints to stdout")
    p.add_argument("--no-coerce", action="store_true", help="Disable numeric coercion attempts (keep raw dtypes from DB/CSV)")
    p.add_argument("--verbose", action="store_true", help="Enable verbose logging")
    return p


def main(argv: Optional[List[str]] = None) -> int:
    parser = build_arg_parser()
    args = parser.parse_args(argv)

    if args.verbose:
        LOG.setLevel(logging.DEBUG)

    try:
        model, feature_columns, meta = load_model_and_meta(args.model, args.meta)
    except Exception as exc:
        LOG.exception("Failed to load model or metadata: %s", exc)
        print(f"ERROR: {exc}", file=sys.stderr)
        return 2


    try:
        if args.from_db:
            if not args.ids:
                parser.error("--from-db requires at least one --id")
            df_in = fetch_from_db(args.ids)
        else:
            df_in = read_from_csv(args.from_csv)
    except Exception as exc:
        LOG.exception("Failed to load input rows: %s", exc)
        print(f"ERROR: {exc}", file=sys.stderr)
        return 3

    if df_in.empty:
        LOG.warning("No rows loaded for prediction.")
        print("No rows loaded for prediction.", file=sys.stderr)
        return 4


    try:
        if args.no_coerce:
            if feature_columns:
                for c in feature_columns:
                    if c not in df_in.columns:
                        df_in[c] = np.nan
                X = df_in[feature_columns]
            else:
                X = df_in.copy()
            X = X.replace({pd.NA: np.nan, None: np.nan})
            for c in X.select_dtypes(include=["string", "object"]):
                X[c] = X[c].astype(object).where(X[c].notna(), np.nan)
        else:

            rows = df_in.to_dict(orient="records")
            X = prepare_rows_for_model(rows, feature_columns)
            X = X.replace({pd.NA: np.nan, None: np.nan})
            for c in X.select_dtypes(include=["string", "object"]):
                X[c] = X[c].astype(object).where(X[c].notna(), np.nan)
    except Exception as exc:
        LOG.exception("Failed to prepare features: %s", exc)
        print(f"ERROR preparing features: {exc}", file=sys.stderr)
        return 5

    X = sanitize_inputs_for_sklearn(X)

    bad_cols = []
    for col in X.columns:
        try:
            sample = X[col].dropna().astype(str).head(20).tolist()
            for v in sample:
                try:
                    float(v)
                except Exception:
                    bad_cols.append((col, v))
                    break
        except Exception:
            bad_cols.append((col, "<could not sample>"))
    if bad_cols:
        LOG.debug("Potential non-numeric columns (first non-numeric sample shown): %s", bad_cols)

    try:
        preds = model.predict(X)
    except Exception as exc:
        LOG.exception("Prediction failed: %s", exc)
        print(f"ERROR during prediction: {exc}", file=sys.stderr)
        return 6

    out_df = df_in.copy().reset_index(drop=True)
    try:
        preds_list = [float(x) for x in preds]
    except Exception:
        preds_list = list(preds)
    out_df["predicted_price"] = preds_list

    try:
        write_output(out_df, args.output)
    except Exception as exc:
        LOG.exception("Failed to write output: %s", exc)
        print(f"ERROR writing output: {exc}", file=sys.stderr)
        return 7

    LOG.info("Predicted %d rows. Model saved at: %s", out_df.shape[0], meta.get("saved_at"))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
