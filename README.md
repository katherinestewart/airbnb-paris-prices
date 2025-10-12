#                       Airbnb Paris Prices Predictor

This repository provides an end-to-end solution for predicting nightly Airbnb prices in Paris using machine learning. The workflow encompasses data ingestion, cleaning, feature engineering, model training, evaluation, and an interactive web application for predictions.

-----------------------------------------------------------------------------

-Features-

. Data Handling

    . Loads raw Airbnb listings and reviews from a PostgreSQL database.
    . Cleans, joins, and caches the data for fast repeated access.
    . Provides commands to verify and reset processed data.

. Feature Engineering

    . Numeric and categorical features, including amenities counts and property  characteristics.
    . Target encoding of neighbourhood_cleansed to capture local price patterns.
    . Computes derived features like dist_to_center_km and avg_comment_length.

. Modeling

    . XGBoost Regressor trained on log-transformed nightly price (log1p(price)).
    . Handles missing values via median (numeric) and most-frequent (categorical) imputers.
    . Categorical variables are one-hot encoded.
    . Early stopping is used during training for optimal performance.
    . K-Fold cross-validation with out-of-fold target encoding for robust MAE/RMSE estimates.

. Prediction and Deployment

    . Streamlit web application for interactive price prediction.
    . Accepts user input for property features and outputs predicted nightly price.
    . Handles unseen or missing categories safely.
    . Provides caching for model, metadata, and data for faster response.

-----------------------------------------------------------------------------

-Requirements-

. Python 3.10+
. PostgreSQL (for raw Airbnb data)
. Docker & Docker Compose (optional, for environment reproducibility)

. Install Python dependencies:
  pip install -r requirements.txt

-----------------------------------------------------------------------------

-Makefile Commands-

The repository uses a Makefile to streamline workflow:

Command	Description:

. make db
    #################################

. make clean
    #################################

. make verify_clean
    #################################

. make cache
    Generate cached parquet dataset for faster access.

. make train
    Train the final model using the latest processed dataset. Saves the pipeline, preprocessor, and metadata.

. make evaluate
    Run K-Fold cross-validation with out-of-fold target encoding for neighbourhood, and optionally train the final model.

 .make app
    Launch the Streamlit web application.

-----------------------------------------------------------------------------

-Directory Structure-
```text
airbnb-paris-prices/
│
├─ app/
│   └─ streamlit_app.py                 # Streamlit web application entrypoint
│
├─ data/
│   ├─ raw/                             # Raw unprocessed Airbnb data
│   └─ processed/                       # Processed & cached dataset (parquet)
│
├─ docs/
│   ├─ feature_dictionary.md            # Description of features
│   ├─ model_card.md                    # Model summary and details
│   ├─ model_metrics.csv                # MAE/RMSE and other metrics
│   └─ top_features.csv                 # Feature importance or rankings
│
├─ models/
│   ├─ metadata.json                    # Metadata for model and preprocessing
│   ├─ neighbourhood_te.json            # Full-data target encoding mapping
│   ├─ preprocessing_v1.pkl             # Preprocessor pipeline
│   └─ xgb_model_v1.pkl                 # Trained XGBoost model pipeline
│
├─ scripts/
│   ├─ clean_listings.py                # Scripts for cleaning Airbnb listings
│   └─ clean_reviews.py                 # Scripts for cleaning review summaries
│
├─ sql/
│   ├─ indexes.sql                      # Database index creation
│   ├─ load.sql                         # Loading raw data into database
│   └─ schema.sql                       # Database schema definition
│
├─ src/
│   ├─ __pycache__/                     # Compiled Python bytecode
│   │   ├─ __init__.cpython-310.pyc
│   │   └─ config.cpython-310.pyc
│   │
│   ├─ __init__.py
│   ├─ config.py                        # Global paths and constants
│   │
│   ├─ data/
│   │   ├─ __pycache__/
│   │   ├─ __init__.py
│   │   └─ load_data.py                 # Load and cache processed data
│   │
│   ├─ features/
│   │   ├─ __pycache__/
│   │   └─ build_features.py            # Feature engineering functions
│   │
│   └─ models/
│       ├─ __pycache__/
│       ├─ __init__.py
│       ├─ evaluate.py                  # K-Fold CV + target encoding
│       ├─ predict.py                   # Prediction helper module
│       └─ train.py                     # Final training script
│
├─ .gitignore
├─ docker-compose.yml
├─ Makefile
├─ README.md
└─ requirements.txt
```

-----------------------------------------------------------------------------

-Usage-

.1  Setup Database and Data

    make db
    make clean

.2  Generate Cached Dataset

    make cache

.3  Train the Model

    make train

      (This will train XGBoost on log-transformed nightly price.
       Saves pipeline to models/xgb_model_v1.pkl and metadata to
       models/metadata.json.)

.4  Evaluate Model Performance

    make evaluate

      (Runs 5-fold cross-validation using out-of-fold target encoding.
       Produces stable MAE and RMSE estimates.
       Optionally retrains final model on full dataset.)

.5  Run Streamlit App

    make app

      (Interactive interface for predicting nightly price.
      Input fields for neighbourhood, property_type, room_type,
      accommodates, bedrooms, beds, bathrooms, avg_comment_length, days_since_last_review. Predicted price displayed instantly.)

-----------------------------------------------------------------------------

-Notes-

. Target Encoding: neighbourhood_cleansed is encoded using mean price.
  During prediction, unseen or missing neighbourhoods default to the global mean.

. Log Transformation: price is trained with log1p(price) to stabilize
  variance; predictions are inverted using expm1.

. Caching: Both Streamlit and training scripts leverage caching to avoid
  recomputation and improve responsiveness.

. Validation Metrics: MAE and RMSE are reported on a validation split for
  train.py and across folds for evaluate.py.

-----------------------------------------------------------------------------

-License-

This repository is released under the MIT License.
