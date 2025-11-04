from pathlib import Path
import os
import json
import time
import platform
import logging

import numpy as np
import pandas as pd
import joblib
import xgboost as xgb

from sqlalchemy import create_engine
from sklearn.model_selection import train_test_split, KFold, RandomizedSearchCV
from sklearn.compose import ColumnTransformer, TransformedTargetRegressor
from sklearn.preprocessing import OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.metrics import mean_absolute_error
import sklearn  # for version pin in metadata

LOG = logging.getLogger("model_monotone")
logging.basicConfig(level=logging.INFO)

ARTIFACTS_DIR = Path("models")

DB_URL = os.getenv("AIRBNB_DB_URL", "postgresql://airbnb:airbnb@localhost:5432/airbnb")

SQL = """
SELECT l.*, r.n_reviews, r.first_review, r.last_review,
       r.avg_comment_length, r.days_since_last_review
FROM clean.listings_features l
LEFT JOIN clean.reviews_summary r ON r.listing_id = l.id;
"""

RANDOM_SEED = 42
N_SPLITS = 5
N_ITER = 40  # RandomizedSearchCV iterations


def load_df():
    eng = create_engine(DB_URL)
    return pd.read_sql(SQL, eng)


def build_preproc(df: pd.DataFrame) -> ColumnTransformer:
    # Keep same categorical and numeric split as original
    cat = ["neighbourhood_cleansed", "property_type_slim", "room_type"]
    num = [c for c in df.columns
           if c not in cat + ["price", "id", "first_review", "last_review", "property_type"]]
    return ColumnTransformer([
        ("cat", OneHotEncoder(handle_unknown="infrequent_if_exist", min_frequency=0.01), cat),
        ("num", SimpleImputer(strategy="median"), num),
    ])


def main():
    LOG.info("Loading data...")
    df = load_df()

    y = df["price"]
    X = df.drop(columns=["price", "id", "first_review", "last_review", "property_type"])

    # Stratify on binned log-price; fallback if qcut fails
    y_log = np.log(y.clip(lower=1e-6))
    try:
        strat = pd.qcut(y_log, q=10, labels=False, duplicates="drop")
    except Exception:
        LOG.warning("pd.qcut failed for stratification; falling back to no stratify.")
        strat = None

    try:
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=RANDOM_SEED, stratify=strat
        )
    except ValueError:
        LOG.warning("Stratified split failed; falling back to random split without stratify.")
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=RANDOM_SEED, stratify=None
        )

    LOG.info("Building preprocessor and computing transformed feature names...")
    pre = build_preproc(df)

    # Fit preprocessor on the training set to capture categories and ensure consistent output columns
    pre.fit(X_train)

    # Try to get transformed feature names in output order. Fallback to original X_train columns.
    try:
        transformed_feature_names = list(pre.get_feature_names_out())
    except Exception as exc:
        LOG.warning("Could not get transformed feature names via get_feature_names_out(): %s", exc)
        LOG.warning("Falling back to original training feature names (may reduce granularity of monotone mapping).")
        transformed_feature_names = list(X_train.columns)

    LOG.info("Transformed feature count: %d", len(transformed_feature_names))

    # Define original numeric features we want to constrain to be monotone increasing (+1).
    # Change this dict to adjust which features are constrained and their sign.
    constrained_original = {
        "bathrooms": 1,
        "bedrooms": 1,
        "beds": 1,
        "accommodates": 1,
    }

    # Build monotone map for each transformed feature: if the original name appears in the
    # transformed feature name, assign the same constraint sign. Otherwise 0.
    monotone_list = []
    for tf in transformed_feature_names:
        assigned = 0
        for orig_name, sign in constrained_original.items():
            # simple substring match works for common get_feature_names_out outputs
            if orig_name in tf:
                assigned = int(sign)
                break
        monotone_list.append(assigned)

    LOG.info("Monotone constraint vector length: %d (sum=%d)", len(monotone_list), sum(abs(np.array(monotone_list))))
    # XGBoost expects a sequence (tuple/list) in the same order as input features; use tuple
    monotone_constraints_param = tuple(int(x) for x in monotone_list)

    # Build the XGBRegressor with monotone constraints
    LOG.info("Creating XGBRegressor with monotone_constraints (len=%d)", len(monotone_constraints_param))
    xgb_reg = xgb.XGBRegressor(
        tree_method="hist",
        n_estimators=1000,
        learning_rate=0.05,
        random_state=RANDOM_SEED,
        n_jobs=-1,
        monotone_constraints=monotone_constraints_param,
        max_bin=256,  # slightly increase max_bin for hist + monotone constraints
    )

    pipe = Pipeline([("pre", pre), ("model", xgb_reg)])

    model = TransformedTargetRegressor(
        regressor=pipe, func=np.log, inverse_func=np.exp
    )

    param_dist = {
        "regressor__model__max_depth": [4, 6, 8, 10],
        "regressor__model__min_child_weight": [1, 3, 5, 7],
        "regressor__model__subsample": [0.6, 0.8, 1.0],
        "regressor__model__colsample_bytree": [0.6, 0.8, 1.0],
        "regressor__model__gamma": [0, 0.5, 1.0],
        "regressor__model__reg_alpha": [0, 0.001, 0.01, 0.1],
        "regressor__model__reg_lambda": [0.1, 1.0, 5.0, 10.0],
        "regressor__model__learning_rate": [0.03, 0.05, 0.08],
        "regressor__model__n_estimators": [400, 800, 1200],
        "regressor__model__objective": ["reg:squarederror", "reg:absoluteerror"],
    }

    cv = KFold(n_splits=N_SPLITS, shuffle=True, random_state=RANDOM_SEED)

    search = RandomizedSearchCV(
        estimator=model,
        param_distributions=param_dist,
        n_iter=N_ITER,
        scoring="neg_mean_absolute_error",
        cv=cv,
        n_jobs=-1,
        verbose=1,
        random_state=RANDOM_SEED,
    )

    LOG.info("Starting RandomizedSearchCV...")
    search.fit(X_train, y_train)

    best = search.best_estimator_
    y_pred = best.predict(X_test)  # back on price scale thanks to TTR

    # Metrics
    mae_overall = mean_absolute_error(y_test, y_pred)
    p90 = y_test.quantile(0.90)
    hi = y_test >= p90
    mae_hi = mean_absolute_error(y_test[hi], y_pred[hi]) if hi.any() else float("nan")
    mae_lo = mean_absolute_error(y_test[~hi], y_pred[~hi]) if (~hi).any() else float("nan")

    baseline_pred = np.full_like(y_test, float(y_train.median()), dtype=float)
    baseline_mae = mean_absolute_error(y_test, baseline_pred)

    # Ensure artifacts dir exists
    ARTIFACTS_DIR.mkdir(parents=True, exist_ok=True)

    model_path = ARTIFACTS_DIR / "model_monotone.joblib"
    joblib.dump(best, model_path)

    meta = {
        "saved_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        "python": platform.python_version(),
        "numpy": np.__version__,
        "pandas": pd.__version__,
        "sklearn": sklearn.__version__,
        "xgboost": xgb.__version__,
        "target": "price (log-trained via TTR, predictions on price scale)",
        "train_rows": int(len(X_train)),
        "test_rows": int(len(X_test)),
        "features": list(X.columns),
        "feature_columns": list(X.columns),
        "transformed_feature_count": len(transformed_feature_names),
        "monotone_constrained_original": constrained_original,
        "monotone_constraints_vector_length": len(monotone_constraints_param),
        "cv_mae_mean": float(-search.best_score_),
        "mae_overall": float(mae_overall),
        "mae_p90plus": float(mae_hi) if not pd.isna(mae_hi) else None,
        "mae_le_p90": float(mae_lo) if not pd.isna(mae_lo) else None,
        "baseline_mae": float(baseline_mae),
        "best_params": search.best_params_,
    }
    meta_path = ARTIFACTS_DIR / "model_monotone.json"
    meta_path.write_text(json.dumps(meta, indent=2))

    LOG.info("Saved → %s", model_path)
    LOG.info("Saved → %s", meta_path)
    print(json.dumps(meta, indent=2))


if __name__ == "__main__":
    main()
