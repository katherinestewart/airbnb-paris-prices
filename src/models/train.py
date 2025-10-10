import os
import joblib
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from xgboost import XGBRegressor
from src.data.load_data import load_data
from src.config import MODEL_FILE, PREPROCESSOR_FILE, METADATA_FILE

def train_model():
    df = load_data()

    df = df.dropna(subset=['price'])
    df['price'] = df['price'].astype(float)

    numeric_cols = ['accommodates','bedrooms','beds','bathrooms','avg_comment_length','days_since_last_review']
    categorical_cols = ['neighbourhood_cleansed','property_type','room_type']

    X = df[numeric_cols + categorical_cols]
    y = df['price']

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    numeric_transformer = SimpleImputer(strategy='median')
    categorical_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='most_frequent')),
        ('onehot', OneHotEncoder(handle_unknown='ignore'))
    ])

    preprocessor = ColumnTransformer(
        transformers=[
            ('num', numeric_transformer, numeric_cols),
            ('cat', categorical_transformer, categorical_cols)
        ])

    model = Pipeline(steps=[
        ('preprocessor', preprocessor),
        ('regressor', XGBRegressor(
            n_estimators=100,
            max_depth=6,
            learning_rate=0.1,
            random_state=42,
            n_jobs=-1
        ))
    ])

    model.fit(X_train, y_train)


    os.makedirs(os.path.dirname(MODEL_FILE), exist_ok=True)
    joblib.dump(model, MODEL_FILE)
    joblib.dump(preprocessor, PREPROCESSOR_FILE)


    from sklearn.metrics import mean_absolute_error, mean_squared_error
    import numpy as np
    y_pred = model.predict(X_test)
    mae = mean_absolute_error(y_test, y_pred)
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))


    import json
    metadata = {
        "model_name": "XGBoost Regressor",
        "model_version": "v1",
        "trained_on": pd.Timestamp.today().strftime("%Y-%m-%d"),
        "features": numeric_cols + categorical_cols,
        "metrics": {
            "MAE": round(mae, 2),
            "RMSE": round(rmse, 2)
        },
        "preprocessor_file": os.path.basename(PREPROCESSOR_FILE),
        "model_file": os.path.basename(MODEL_FILE),
        "notes": "Trained on Airbnb Paris cleaned dataset using numeric and categorical features."
    }
    with open(METADATA_FILE, 'w') as f:
        json.dump(metadata, f, indent=4)

    print(f"Training completed. MAE={mae:.2f}, RMSE={rmse:.2f}")

if __name__ == "__main__":
    train_model()
