import joblib
import pandas as pd
from src.config import MODEL_FILE

model = joblib.load(MODEL_FILE)

def predict_price(input_dict):
    """
    Predict Airbnb price from input dictionary of features.
    """
    input_df = pd.DataFrame([input_dict])
    prediction = model.predict(input_df)[0]
    return prediction

if __name__ == "__main__":
    
    sample_input = {
        'neighbourhood_cleansed': 'Paris 1er',
        'property_type': 'Apartment',
        'room_type': 'Entire home/apt',
        'accommodates': 2,
        'bedrooms': 1,
        'beds': 1,
        'bathrooms': 1,
        'avg_comment_length': 100,
        'days_since_last_review': 30
    }
    price = predict_price(sample_input)
    print(f"Predicted price: €{price:.2f}")
