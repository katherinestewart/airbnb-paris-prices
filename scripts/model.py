from pathlib import Path
import os
import json
import time
import platform
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

# --- Data ---
def load_df():
    eng = create_engine(DB_URL)
    return pd.read_sql(SQL, eng)

def build_preproc(df):
    cat = ["neighbourhood_cleansed", "property_type_slim", "room_type"]
    num = [c for c in df.columns
           if c not in cat + ["price","id","first_review","last_review","property_type"]]
    return ColumnTransformer([
        ("cat", OneHotEncoder(handle_unknown="infrequent_if_exist", min_frequency=0.01), cat),
        ("num", SimpleImputer(strategy="median"), num),
    ])

# --- Train ---
def main():
    df = load_df()

    y = df["price"]
    X = df.drop(columns=["price","id","first_review","last_review","property_type"])

    # Stratify on binned log-price
    y_log = np.log(y)
    strat = pd.qcut(y_log, q=10, labels=False, duplicates="drop")

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=RANDOM_SEED, stratify=strat
    )

    pre = build_preproc(df)

    xgb_reg = xgb.XGBRegressor(
        tree_method="hist",
        n_estimators=1000,
        learning_rate=0.05,
        random_state=RANDOM_SEED,
        n_jobs=-1,
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

    search.fit(X_train, y_train)

    best = search.best_estimator_
    y_pred = best.predict(X_test)  # back on price scale thanks to TTR

    # Metrics
    mae_overall = mean_absolute_error(y_test, y_pred)
    p90 = y_test.quantile(0.90)
    hi = y_test >= p90
    mae_hi = mean_absolute_error(y_test[hi], y_pred[hi])
    mae_lo = mean_absolute_error(y_test[~hi], y_pred[~hi])

    baseline_pred = np.full_like(y_test, float(y_train.median()), dtype=float)
    baseline_mae = mean_absolute_error(y_test, baseline_pred)

    model_path = ARTIFACTS_DIR / "model.joblib"
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
        "cv_mae_mean": float(-search.best_score_),
        "mae_overall": float(mae_overall),
        "mae_p90plus": float(mae_hi),
        "mae_le_p90": float(mae_lo),
        "baseline_mae": float(baseline_mae),
        "best_params": search.best_params_,
    }
    meta_path = ARTIFACTS_DIR / "model.metadata.json"
    meta_path.write_text(json.dumps(meta, indent=2))

    print(f"\nSaved → {model_path}")
    print(f"Saved → {meta_path}")
    print(json.dumps(meta, indent=2))


if __name__ == "__main__":
    main()
