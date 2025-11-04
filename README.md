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

. Modeling

    . XGBoost Regressor trained on log-transformed nightly price (log1p(price)).
    . Handles missing values via median (numeric) and most-frequent (categorical) imputers.
    . Categorical variables are one-hot encoded.

. Prediction and Deployment

    . Streamlit web application for interactive price prediction.
    . Accepts user input for property features and outputs predicted nightly price.
    . Handles unseen or missing categories safely.
    . Provides caching for model, metadata, and data for faster response.

-----------------------------------------------------------------------------

-Requirements-

- Python 3.10+ (use the version pinned in`requirements.txt` for reproducibility)
- Docker & Docker Compose (recommended for local end-to-end)
- PostgreSQL (the Makefile brings a containerized instance)
- Recommended: `git`, `gcloud` (only if deploying to GCP)

Install Python dependencies:

```bash
pip install -r requirements.txt

-----------------------------------------------------------------------------

-Makefile Commands-

The repository uses a Makefile to streamline workflow:

Command	Description:

. make up
    Start Docker Compose services (detached).

. make schema
    Ensure the DB service is running, then apply `sql/schema.sql` inside the `db` container to create the `raw` schema and tables.

. make load
    Load raw CSV data into the `raw` schema by executing `sql/load.sql` inside the `db` container.

. make indexes
    Create database indexes by executing `sql/indexes.sql` inside the `db` container.

. make db
    Convenience target that runs `up`, `schema`, `load`, and `indexes` in sequence to prepare the local database.

. make verify
    Run a set of `psql` checks in the `db` container to list `raw` tables, show counts for `raw.listings` and `raw.reviews`, and display table schemas.

. make down
    Stop and remove Docker Compose containers.

. make reset
    Tear down containers and volumes, bring the stack back up and re-run `schema`, `load`, and `indexes` to recreate and reload the database.

. make count_rows
    Quick SQL count that reports the number of rows in `raw.listings` and `raw.reviews`.

. make clean
    Run the cleaning pipeline scripts: `python scripts/clean_listings.py` and `python scripts/clean_reviews.py` to populate the `clean` schema.

. make verify_clean
    Run a diagnostic query that reports table names, column counts, row counts, and min/max prices for `clean` tables (`listings_features`, `reviews`, `reviews_summary`).

. make train
    Run the legacy training script: `python scripts/model.py`.

. make train_monotone
    Run the monotone-constrained training pipeline: `python scripts/model_monotone.py`.

. make start_api
    Start the FastAPI application locally with Uvicorn on port `8000` (reload enabled).

. make start_streamlit
    Start the Streamlit UI locally (points to `http://localhost:8000` by default) on port `8501`.

. make local_compose
    Start the `api` and `streamlit` services via Docker Compose (built if necessary).

. make stop_all
    Stop and remove the `api` and `streamlit` compose services (leaves `db` if desired).

. make up_all
    Build and bring up the full stack (`db`, `api`, `streamlit`) via Docker Compose.

. make test_api_health
    Curl the `/health` endpoint of the running API for a quick smoke test.

. make test_api_metadata
    Curl the `/metadata` endpoint of the running API.

. make test_api_predict
    POST an inline single-row JSON sample to `/predict` and pretty-print the response (useful for basic functional testing).

. make test_api_predict_file
    POST a JSON file (set via `JSON=path/to/file.json`) to `/predict` for integration testing of batched inputs.

. make build_streamlit_image
    Build the Streamlit Docker image locally using `Dockerfile.streamlit`.

. make build_api_image
    Wrapper target to build the API Docker image (delegates to the project’s docker build workflow).

. make deploy_api
    Composite target that orchestrates the API build → push → deploy workflow (wraps `docker_allow`, `docker_build`, `docker_push`, `docker_deploy` where configured).

. make deploy_all
    High-level target that runs `deploy_api` (add Streamlit deployment steps if you choose to host Streamlit in the cloud).

-----------------------------------------------------------------------------

-Directory Structure-
```text
airbnb-paris-prices/
│
├─ .devcontainer/
│   └─ devcontainer.json
│
├─ .streamlit/
│   ├─ config.toml
│   └─ secrets.toml
│
├─ app/
│   ├─ app.py
│   ├─ config.py
│   ├─ paris_bnb_map.py
│   └─ streamlit_app.py
│
├─ data/
│   └─ raw/
│
├─ docs/
│   ├─ feature_dictionary.md
│   └─ property_type_map.json
│
├─ models/
│   ├─ model_monotone.joblib
│   └─ model_monotone.json
│
├─ notebooks/
│
├─ scripts/
│   ├─ clean_listings.py
│   ├─ clean_reviews.py
│   ├─ model_monotone.py
│   └─ predict.py
│
├─ sql/
│   ├─ indexes.sql
│   ├─ load.sql
│   └─ schema.sql
│
├─ .envrc
├─ .env
├─ .env.yaml
├─ .gcloudignore
├─ .gitignore
├─ cloudbuild.yaml
├─ docker-compose.yml
├─ Dockerfile.api
├─ Dockerfile.streamlit
├─ Makefile
├─ README.md
├─ requirements.txt
└─ setup.py

```

-----------------------------------------------------------------------------

-Usage-

.1  Setup Database and Data

    make db
    make clean

.2  Train the Model

    make train_monotone

      (Monotone-constrained training.
      Generate model_monotone.joblib and
      model monotone.json)

.3  Start services for local testing

    make run_api

      (Start FastAPI locally (development server))


.4  Run Streamlit App

    make run_streamlit

      (Interactive interface for predicting nightly price.
      Input fields for neighbourhood, property_type, room_type,
      accommodates, bedrooms, beds and bathrooms.
      Predicted price displayed instantly.)

-----------------------------------------------------------------------------


## Highlights

- Clean & reproducible data pipeline that ingests raw Airbnb CSVs into a local PostgreSQL instance.
- Feature engineering: numeric/categorical processing, amenities features, derived ratios.
- Modeling: XGBoost regressor trained on log-transformed nightly price with a reproducible pipeline and metadata.
- Web app: Streamlit UI for interactive predictions.
- Docker + Docker Compose for local reproducible environments, plus `cloudbuild.yaml` for GCP Cloud Run deployments.

-----------------------------------------------------------------------------

-License-

This repository is released under the MIT License.
