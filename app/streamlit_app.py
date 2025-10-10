import streamlit as st
import joblib
import pandas as pd


model = joblib.load('models/xgb_model_v1.pkl')

st.set_page_config(page_title="Airbnb Paris Price Predictor", layout="wide")
st.title("Airbnb Paris Price Predictor")

st.sidebar.header("Property Details")


neighbourhood = st.sidebar.selectbox("Neighbourhood", ["Paris 1er", "Paris 2e", "Paris 3e"])
property_type = st.sidebar.selectbox("Property Type", ["Apartment", "Loft", "House"])
room_type = st.sidebar.selectbox("Room Type", ["Entire home/apt", "Private room", "Shared room"])
accommodates = st.sidebar.number_input("Accommodates", min_value=1, max_value=10, value=2)
bedrooms = st.sidebar.number_input("Bedrooms", min_value=0, max_value=10, value=1)
beds = st.sidebar.number_input("Beds", min_value=0, max_value=10, value=1)
bathrooms = st.sidebar.number_input("Bathrooms", min_value=0, max_value=10, value=1)
avg_comment_length = st.sidebar.number_input("Avg Comment Length", min_value=0, value=100)
days_since_last_review = st.sidebar.number_input("Days Since Last Review", min_value=0, value=30)


input_df = pd.DataFrame([{
    'neighbourhood_cleansed': neighbourhood,
    'property_type': property_type,
    'room_type': room_type,
    'accommodates': accommodates,
    'bedrooms': bedrooms,
    'beds': beds,
    'bathrooms': bathrooms,
    'avg_comment_length': avg_comment_length,
    'days_since_last_review': days_since_last_review
}])


if st.button("Predict Price"):
    prediction = model.predict(input_df)[0]
    st.success(f"Predicted price per night: €{prediction:.2f}")
