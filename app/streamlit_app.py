from __future__ import annotations

import json
import logging
import math
import os
from pathlib import Path
from typing import Any, Dict, List, Optional

import joblib
import numpy as np
import pandas as pd
import requests
import streamlit as st

# ---------- Config ----------
LOG = logging.getLogger("streamlit_airbnb")
logging.basicConfig(level=logging.INFO)

API_URL = os.getenv("API_URL", "http://localhost:8000")
ARTIFACTS_DIR = Path(os.getenv("ARTIFACTS_DIR", "models"))
MODEL_FILE = Path(os.getenv("MODEL_FILE", ARTIFACTS_DIR / "model_monotone.joblib"))
META_FILE = Path(os.getenv("META_FILE", ARTIFACTS_DIR / "model_monotone.json"))
PROPERTY_MAP = Path(os.getenv("PROPERTY_MAP", "docs/property_type_map.json"))
NEIGHBOURHOOD_TE = Path(os.getenv("NEIGHBOURHOOD_TE", "models/neighbourhood_te.json"))

BATHROOMS_UPPER_CAP = float(os.getenv("BATHROOMS_UPPER_CAP", 4.0))
BEDROOMS_UPPER_CAP = float(os.getenv("BEDROOMS_UPPER_CAP", 4.0))
BEDS_UPPER_CAP = float(os.getenv("BEDS_UPPER_CAP", 4.0))
ACCOMMODATES_UPPER_CAP = float(os.getenv("ACCOMMODATES_UPPER_CAP", 8.0))

st.set_page_config(page_title="Airbnb Price Predictor", layout="wide")
st.title("Airbnb Price Predictor")

# ---------- Caching helpers ----------
@st.cache_resource
def load_metadata() -> dict:
    try:
        if META_FILE.exists():
            return json.loads(META_FILE.read_text(encoding="utf-8"))
    except Exception as e:
        LOG.exception("Failed to load metadata: %s", e)
    return {}

@st.cache_resource
def load_property_map() -> dict:
    try:
        if PROPERTY_MAP.exists():
            return json.loads(PROPERTY_MAP.read_text(encoding="utf-8"))
    except Exception as e:
        LOG.exception("Failed to load property map: %s", e)
    return {"kept_categories": [], "mapping": {}}

@st.cache_resource
def load_neighbourhoods() -> List[str]:
    try:
        if NEIGHBOURHOOD_TE.exists():
            payload = json.loads(NEIGHBOURHOOD_TE.read_text(encoding="utf-8"))
            mapping = payload.get("mapping") or {}
            return sorted(list(mapping.keys()))
    except Exception:
        LOG.exception("Failed to load neighbourhood TE; using fallback list")
    return ["Louvre", "Montmartre", "Saint-Germain", "Le Marais"]

metadata = load_metadata()
feature_columns: Optional[List[str]] = None
if isinstance(metadata.get("feature_columns"), list):
    feature_columns = metadata.get("feature_columns")
elif isinstance(metadata.get("features"), list):
    feature_columns = metadata.get("features")

prop_map = load_property_map()
neighbourhood_list = load_neighbourhoods()
default_nb = "Louvre" if "Louvre" in neighbourhood_list else (neighbourhood_list[0] if neighbourhood_list else "")

# ---------- Helpers ----------
def apply_engineered_features_to_row(row: Dict[str, Any]) -> Dict[str, Any]:
    out = dict(row)
    # bathrooms
    try:
        b = out.get("bathrooms", 0.0)
        b = float(b) if b is not None else 0.0
        b = max(0.0, min(b, BATHROOMS_UPPER_CAP))
        out["bathrooms"] = b
    except Exception:
        out["bathrooms"] = 0.0

    # bedrooms
    try:
        bd = out.get("bedrooms", 0.0)
        bd = float(bd) if bd is not None else 0.0
        bd = max(0.0, min(bd, BEDROOMS_UPPER_CAP))
        out["bedrooms"] = bd
    except Exception:
        out["bedrooms"] = 0.0

    # accommodates
    try:
        ac = out.get("accommodates", 1.0)
        ac = float(ac) if ac is not None else 1.0
        if ac < 1.0:
            ac = 1.0
        ac = min(ac, ACCOMMODATES_UPPER_CAP)
        out["accommodates"] = ac
    except Exception:
        out["accommodates"] = 1.0

    # beds
    try:
        bs = out.get("beds", 0.0)
        bs = float(bs) if bs is not None else 0.0
        bs = max(0.0, min(bs, BEDS_UPPER_CAP))
        out["beds"] = bs
    except Exception:
        out["beds"] = 0.0

    out["bathrooms_per_bedroom"] = float(out["bathrooms"]) / float(out["bedrooms"]) if out["bedrooms"] else 0.0
    out["bathrooms_per_accommodates"] = float(out["bathrooms"]) / float(out["accommodates"]) if out["accommodates"] else 0.0
    return out

def prop_group(room_type: str, prop_type: str) -> str:
    room_type_lc = (room_type or "").lower()
    prop_type_lc = (prop_type or "").lower()
    if room_type_lc == "entire home/apt":
        return "Entire home/apt" if "entire" in prop_type_lc else "Other"
    if room_type_lc == "private room":
        return "Private room" if "private" in prop_type_lc else "Other"
    if room_type_lc == "shared room":
        return "Shared room" if "shared" in prop_type_lc else "Other"
    if room_type_lc == "hotel room":
        return "Hotel room" if "hotel" in prop_type_lc or "boutique" in prop_type_lc else "Other"
    return "Other"

def get_property_types_for_room(room_type: str) -> List[str]:
    types_for_room = []
    mapping = prop_map.get("mapping", {}) or {}
    for prop, mapped_group in mapping.items():
        if prop_group(room_type, prop) == room_type:
            types_for_room.append(prop if mapped_group != "Other" else "Other")
    return sorted(list(set(types_for_room)))

# JSON-sanitizer: convert numpy scalars -> native python and NaN/Inf -> None
def _sanitize_value_for_json(value: Any) -> Any:
    # numpy scalar -> python scalar
    if isinstance(value, (np.generic,)):
        try:
            value = value.item()
        except Exception:
            value = float(value)
    # Pandas NA (pd.NA)
    if value is pd.NA:
        return None
    # floats: reject NaN/Inf -> return None
    if isinstance(value, float):
        if not math.isfinite(value):
            return None
        return value
    # ints are fine
    if isinstance(value, (int, str, bool, type(None))):
        return value
    # lists / tuples / arrays: sanitize elements
    if isinstance(value, (list, tuple, np.ndarray)):
        return [_sanitize_value_for_json(v) for v in value]
    # fallback: try convert to native python (e.g., Decimal -> float/string)
    try:
        if hasattr(value, "__float__"):
            f = float(value)
            if math.isfinite(f):
                return f
            return None
    except Exception:
        pass
    try:
        return str(value)
    except Exception:
        return None

def sanitize_row_for_api(row: Dict[str, Any]) -> Dict[str, Any]:
    return {k: _sanitize_value_for_json(v) for k, v in row.items()}

# ---------- Sidebar inputs ----------
st.sidebar.header("Property Details")

neighbourhood = st.sidebar.selectbox(
    "Neighbourhood",
    neighbourhood_list,
    index=neighbourhood_list.index(default_nb) if default_nb in neighbourhood_list else 0,
)

room_type = st.sidebar.selectbox("Room Type", ["Entire home/apt", "Private room", "Shared room", "Hotel room"])
property_type_slim = st.sidebar.selectbox("Property Type (grouped)", get_property_types_for_room(room_type))

accommodates = int(st.sidebar.number_input("Accommodates", 1, int(ACCOMMODATES_UPPER_CAP), 2))
bedrooms = int(st.sidebar.number_input("Bedrooms", 0, int(BEDROOMS_UPPER_CAP), 1))
beds = int(st.sidebar.number_input("Beds", 0, int(BEDS_UPPER_CAP), 1))
bathrooms = float(st.sidebar.number_input("Bathrooms", 0.0, float(BATHROOMS_UPPER_CAP), 1.0, step=0.5))

user_row = {
    "neighbourhood_cleansed": neighbourhood,
    "property_type_slim": property_type_slim,
    "room_type": room_type,
    "accommodates": accommodates,
    "bedrooms": bedrooms,
    "beds": beds,
    "bathrooms": bathrooms,
}

# ---------- Main UI ----------
col_main, _ = st.columns([2, 1])
with col_main:
    if st.button("Predict Price"):
        # engineered features (base)
        processed_row = apply_engineered_features_to_row(user_row)

        # Build a DataFrame copy for local model use WITHOUT mutating processed_row
        if feature_columns:
            row_for_df = dict(processed_row)  # copy
            for c in feature_columns:
                if c not in row_for_df:
                    row_for_df[c] = np.nan
            df = pd.DataFrame([row_for_df])[feature_columns]
        else:
            df = pd.DataFrame([processed_row])

        # Prepare API payload: sanitize and do NOT include np.nan or numpy types
        api_payload_row = sanitize_row_for_api(processed_row)
        api_payload = {"rows": [api_payload_row]}

        # --- Try API first (robust handling) ---
        api_used = False
        api_err = None
        try:
            resp = requests.post(f"{API_URL.rstrip('/')}/predict", json=api_payload, timeout=10)
            resp.raise_for_status()
            data = resp.json()
            preds = data.get("predictions")
            if isinstance(preds, list) and len(preds) > 0:
                val = float(preds[0])
                st.markdown(f"<h1 style='margin:0'>€{val:,.2f}</h1>", unsafe_allow_html=True)
                st.caption("Predicted price per night (via API)")
                api_used = True
            else:
                st.error("API returned unexpected payload. See logs for details.")
                LOG.warning("API returned unexpected payload: %s", data)
        except Exception as e:
            api_err = e
            LOG.exception("API call failed: %s", e)
            st.info("API call failed; attempting to use local model as fallback.")

        # --- Fallback to local model if API did not provide a valid result ---
        if not api_used:
            if MODEL_FILE.exists() and META_FILE.exists():
                try:
                    model = joblib.load(MODEL_FILE)
                    # Use df (which may contain np.nan) for local model
                    if feature_columns and set(feature_columns).issubset(set(df.columns)):
                        df_use = df[feature_columns]
                    else:
                        df_use = df
                    preds = model.predict(df_use)
                    val = float(preds[0])
                    st.markdown(f"<h1 style='margin:0'>€{val:,.2f}</h1>", unsafe_allow_html=True)
                    caption = "Predicted price per night (local model)"
                    if api_err:
                        caption += " — (API failed; showing local fallback)"
                    st.caption(caption)
                    if api_err:
                        st.write(f"API error (for diagnostics): {api_err}")
                except Exception as exc:
                    LOG.exception("Local prediction failed: %s", exc)
                    st.error(f"Local prediction failed: {exc}")
                    if api_err:
                        st.error(f"API error: {api_err}")
            else:
                st.error("No API reachable and local model artifact missing.")
                if api_err:
                    st.write(f"API error: {api_err}")
