from pathlib import Path
import joblib
import json
import numpy as np
import pandas as pd
from sqlalchemy import create_engine
from sklearn.model_selection import train_test_split, KFold, RandomizedSearchCV
from sklearn.compose import ColumnTransformer, TransformedTargetRegressor
from sklearn.preprocessing import OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.metrics import mean_absolute_error
import xgboost as xgb

ARTIFACT_DIR = Path("models"); ARTIFACT_DIR.mkdir(parents=True, exist_ok=True)
DB_URL = "postgresql://airbnb:airbnb@localhost:5432/airbnb"

SQL = """
SELECT l.*, r.n_reviews, r.first_review, r.last_review,
       r.avg_comment_length, r.days_since_last_review
FROM clean.listings_features l
LEFT JOIN clean.reviews_summary r ON r.listing_id = l.id;
"""

def load_df():
    eng = create_engine(DB_URL)
    return pd.read_sql(SQL, eng)

def build_preproc(df):
    cat = ["neighbourhood_cleansed", "property_type_slim", "room_type"]
    num = [c for c in df.columns
           if c not in cat + ["price","id","first_review","last_review","property_type"]]
    return ColumnTransformer([
        ("cat", OneHotEncoder(
            handle_unknown="infrequent_if_exist",
            min_frequency=0.01
        ), cat),
        ("num", SimpleImputer(strategy="median"), num),
    ])

def main(seed=42):
    df = load_df()

    y = df["price"]
    X = df.drop(columns=["price","id","first_review","last_review","property_type"])

    # stratify by binned log-price so the tail is represented
    strat = pd.qcut(np.log(y), q=10, labels=False, duplicates="drop")
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=seed, stratify=strat)

    pre = build_preproc(df)
    xgb_reg = xgb.XGBRegressor(tree_method="hist", n_estimators=1000, learning_rate=0.05, random_state=seed)
    pipe = Pipeline([("pre", pre), ("model", xgb_reg)])
    model = TransformedTargetRegressor(regressor=pipe, func=np.log, inverse_func=np.exp)

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
    cv = KFold(n_splits=5, shuffle=True, random_state=seed)
    search = RandomizedSearchCV(
        estimator=model, param_distributions=param_dist, n_iter=30,
        scoring="neg_mean_absolute_error", cv=cv, n_jobs=-1, random_state=seed, verbose=1
    )
    search.fit(X_train, y_train)
    best = search.best_estimator_

    y_pred = best.predict(X_test)
    mae_overall = mean_absolute_error(y_test, y_pred)
    p90 = y_test.quantile(0.90)
    hi = y_test >= p90
    metrics = {
        "cv_mae_mean": float(-search.best_score_),
        "mae_overall": float(mae_overall),
        "mae_p90plus": float(mean_absolute_error(y_test[hi], y_pred[hi])),
        "mae_le_p90": float(mean_absolute_error(y_test[~hi], y_pred[~hi])),
        "best_params": search.best_params_,
        "features": list(X.columns),
    }

    joblib.dump(best, ARTIFACT_DIR / "price_xgb_ttr_v2.joblib")
    (ARTIFACT_DIR / "price_xgb_ttr_v2.metadata.json").write_text(json.dumps(metrics, indent=2))


    print("Saved models/retrained_model.joblib")
    print(json.dumps(metrics, indent=2))

if __name__ == "__main__":
    main()
