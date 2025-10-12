import json
from pathlib import Path
from typing import List, Dict

import pandas as pd
import streamlit as st

from src.models.predict import predict_price
from src.data.load_data import load_data
from src.config import METADATA_FILE

st.set_page_config(page_title="Airbnb Paris Price Predictor", layout="wide")
st.title("Airbnb Paris Price Predictor")

@st.cache_resource
def load_metadata() -> Dict:
    p = Path(METADATA_FILE)
    if not p.exists():
        return {}
    try:
        return json.loads(p.read_text(encoding="utf-8"))
    except Exception:
        return {}

@st.cache_data
def load_df() -> pd.DataFrame:
    return load_data(use_cache=True)

@st.cache_data
def get_category_options(df: pd.DataFrame, column: str) -> List[str]:
    if column not in df.columns:
        return []
    vals = df[column].dropna().astype(str).unique().tolist()
    return sorted(vals)

@st.cache_data
def get_numeric_bounds(df: pd.DataFrame, column: str):
    if column not in df.columns:
        return None
    ser = pd.to_numeric(df[column], errors="coerce").dropna()
    if ser.empty:
        return None
    return float(ser.min()), float(ser.max()), float(ser.median())

metadata = load_metadata()
expected_features = metadata.get("features") if isinstance(metadata.get("features"), list) else None
df = load_df()

st.sidebar.header("Property Details")

# Dropdowns
def select_or_text(df_col: str, label: str, default: str):
    options = get_category_options(df, df_col)
    if options:
        return st.sidebar.selectbox(label, options, index=0)
    return st.sidebar.text_input(label, value=default)

neighbourhood = select_or_text("neighbourhood_cleansed", "Neighbourhood", "Louvre")
property_type = select_or_text("property_type", "Property Type", "Apartment")
room_type = select_or_text("room_type", "Room Type", "Entire home/apt")

# Number inputs
def number_input_with_bounds(label: str, column: str, step=1, is_int=True, default_override=None):
    bounds = get_numeric_bounds(df, column)
    if bounds:
        min_v, max_v, median_v = bounds
        default_val = default_override if default_override is not None else median_v
        if is_int:
            return st.sidebar.number_input(label, int(min_v), int(max_v), int(default_val), step=int(step))
        else:
            return st.sidebar.number_input(label, float(min_v), float(max_v), float(default_val), step=float(step))
    else:
        return st.sidebar.number_input(label, 0 if is_int else 0.0, 20 if is_int else 10.0,
                                       int(default_override) if default_override else 1,
                                       step=int(step) if is_int else float(step))

accommodates = number_input_with_bounds("Accommodates", "accommodates", is_int=True, default_override=2)
bedrooms = number_input_with_bounds("Bedrooms", "bedrooms", is_int=True, default_override=1)
beds = number_input_with_bounds("Beds", "beds", is_int=True, default_override=1)
bathrooms = number_input_with_bounds("Bathrooms", "bathrooms", is_int=True, default_override=1)
avg_comment_length = number_input_with_bounds("Avg Comment Length", "avg_comment_length", is_int=False, default_override=100.0)
days_since_last_review = number_input_with_bounds("Days Since Last Review", "days_since_last_review", is_int=True, default_override=30)

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

col1, col2 = st.columns([2, 1])

with col1:
    if st.button("Predict Price"):
        try:
            price = predict_price(input_dict)
            st.markdown(f"<h1 style='margin:0'>€{price:,.2f}</h1>", unsafe_allow_html=True)
            st.caption("Predicted price per night")
        except Exception as e:
            st.error(f"Prediction failed: {e}")

st.markdown("---")
