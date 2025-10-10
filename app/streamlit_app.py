import json
from pathlib import Path
from typing import List, Dict, Tuple, Optional

import joblib
import pandas as pd
import streamlit as st

from src.config import MODEL_FILE, METADATA_FILE
from src.data.load_data import load_data

st.set_page_config(page_title="Airbnb Paris Price Predictor", layout="wide")
st.title("Airbnb Paris Price Predictor")

@st.cache_resource
def load_model():
    p = Path(MODEL_FILE)
    if not p.exists():
        return None
    return joblib.load(p)

@st.cache_resource
def load_metadata() -> Dict:
    p = Path(METADATA_FILE)
    if not p.exists():
        return {}
    try:
        return json.loads(p.read_text(encoding="utf-8"))
    except Exception:
        return {}

@st.cache_resource
def load_df() -> pd.DataFrame:
    return load_data(use_cache=True)

@st.cache_data
def get_category_options(df: pd.DataFrame, column: str) -> List[str]:
    if column not in df.columns:
        return []
    vals = df[column].dropna().astype(str).unique().tolist()
    return sorted(vals)

@st.cache_data
def get_numeric_bounds(df: pd.DataFrame, column: str) -> Optional[Tuple[float, float, float]]:
    if column not in df.columns:
        return None
    ser = pd.to_numeric(df[column], errors="coerce").dropna()
    if ser.empty:
        return None
    return float(ser.min()), float(ser.max()), float(ser.median())

model = load_model()
metadata = load_metadata()
expected_features = metadata.get("features") if isinstance(metadata.get("features"), list) else None

df = load_df()

st.sidebar.header("Property Details")

neighbourhood_options = get_category_options(df, "neighbourhood_cleansed")
if neighbourhood_options:
    neighbourhood = st.sidebar.selectbox("Neighbourhood", neighbourhood_options, index=0)
else:
    neighbourhood = st.sidebar.text_input("Neighbourhood", value="Louvre")

property_type_options = get_category_options(df, "property_type")
if property_type_options:
    property_type = st.sidebar.selectbox("Property Type", property_type_options, index=0)
else:
    property_type = st.sidebar.text_input("Property Type", value="Apartment")

room_type_options = get_category_options(df, "room_type")
if room_type_options:
    room_type = st.sidebar.selectbox("Room Type", room_type_options, index=0)
else:
    room_type = st.sidebar.text_input("Room Type", value="Entire home/apt")

def number_input_with_bounds(label: str, column: str, step=1, is_int=True, min_override=None, max_override=None, default_override=None):
    bounds = get_numeric_bounds(df, column)
    if bounds:
        min_v, max_v, median_v = bounds
        if min_override is not None:
            min_v = min_override
        if max_override is not None:
            max_v = max_override
        if default_override is not None:
            default_v = default_override
        else:
            default_v = median_v

        if is_int:
            min_val = int(min_v)
            max_val = int(max_v)
            default_val = int(default_v)
            step_val = int(step)
            return st.sidebar.number_input(label, min_value=min_val, max_value=max_val, value=default_val, step=step_val)
        else:
            min_val = float(min_v)
            max_val = float(max_v)
            default_val = float(default_v)
            step_val = float(step) if isinstance(step, (int, float)) else 0.1
            return st.sidebar.number_input(label, min_value=min_val, max_value=max_val, value=default_val, step=step_val)
    else:
        if is_int:
            return st.sidebar.number_input(label, min_value=0, max_value=20, value=1, step=1)
        else:
            return st.sidebar.number_input(label, min_value=0.0, max_value=10.0, value=1.0, step=0.1)

accommodates = number_input_with_bounds("Accommodates", "accommodates", step=1, is_int=True, default_override=2)
bedrooms = number_input_with_bounds("Bedrooms", "bedrooms", step=1, is_int=True, default_override=1)
beds = number_input_with_bounds("Beds", "beds", step=1, is_int=True, default_override=1)
bathrooms = number_input_with_bounds("Bathrooms", "bathrooms", step=1, is_int=True, default_override=1)
avg_comment_length = number_input_with_bounds("Avg Comment Length", "avg_comment_length", step=1, is_int=False, default_override=100.0)
days_since_last_review = number_input_with_bounds("Days Since Last Review", "days_since_last_review", step=1, is_int=True, default_override=30)

input_dict = {
    "neighbourhood_cleansed": neighbourhood,
    "property_type": property_type,
    "room_type": room_type,
    "accommodates": int(accommodates),
    "bedrooms": int(bedrooms),
    "beds": int(beds),
    "bathrooms": int(bathrooms),
    "avg_comment_length": float(avg_comment_length),
    "days_since_last_review": int(days_since_last_review),
}

def make_input_df(inp: Dict, feature_order: List[str] = None) -> pd.DataFrame:
    df_in = pd.DataFrame([inp])
    if feature_order:
        for f in feature_order:
            if f not in df_in.columns:
                df_in[f] = pd.NA
        df_in = df_in[feature_order]
    return df_in

def predict(input_df: pd.DataFrame):
    if model is None:
        st.error("Model file not found in models/ — run `make train` first.")
        return None
    try:
        preds = model.predict(input_df)
        return float(pd.Series(preds).iloc[0])
    except Exception as e:
        st.error(f"Prediction failed: {e}")
        return None

col1, col2 = st.columns([2, 1])

with col1:
    if st.button("Predict Price"):
        input_df = make_input_df(input_dict, feature_order=expected_features)
        price = predict(input_df)
        if price is not None:
            st.markdown(f"<h1 style='margin:0'>€{price:,.2f}</h1>", unsafe_allow_html=True)
            st.caption("Predicted price per night")

st.markdown("---")
