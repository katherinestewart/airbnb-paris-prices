import ast
import json
import pandas as pd
from pathlib import Path
from collections import Counter
from sqlalchemy import create_engine, text, BigInteger

MAP_PATH = Path("docs/property_type_map.json")


def load_raw_listings():
    print("Loading raw listings from Postgres...")
    engine = create_engine("postgresql://airbnb:airbnb@localhost:5432/airbnb")
    query = "SELECT * FROM raw.listings"
    return pd.read_sql(query, engine)


def select_columns(df):
    keep_cols = [
        "id", "neighbourhood_cleansed", "latitude", "longitude",
        "property_type", "room_type", "accommodates", "bedrooms",
        "beds", "bathrooms", "amenities", "price", "minimum_nights",
        "maximum_nights", "host_is_superhost", "review_scores_rating",
        "review_scores_accuracy", "review_scores_cleanliness",
        "review_scores_checkin", "review_scores_communication",
        "review_scores_location", "review_scores_value", "reviews_per_month"
    ]
    return df[keep_cols].copy()


def convert_numeric(df):
    df["id"] = pd.to_numeric(df["id"], errors="coerce")

    assert df["id"].notna().all(), "Found non-numeric id after coercion"
    df["id"] = df["id"].astype("int64")

    numeric_cols = [
        "latitude","longitude","accommodates","bedrooms","beds","bathrooms",
        "minimum_nights","maximum_nights",
        "review_scores_rating","review_scores_accuracy","review_scores_cleanliness",
        "review_scores_checkin","review_scores_communication","review_scores_location",
        "review_scores_value","reviews_per_month"
    ]
    for col in numeric_cols:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    df["price"] = df["price"].str.replace(r"[^\d.]", "", regex=True).astype(float)
    return df


def handle_missing(df):
    df = df.dropna(subset=["price"])
    for col in ["beds", "bathrooms", "bedrooms"]:
        df[col] = df[col].fillna(df[col].median())
    df["host_is_superhost"] = (
        df["host_is_superhost"]
        .map({"t": True, "f": False})
        .fillna(False)
        .astype(bool)
    )
    return df


def engineer_features(df):
    df["amenities_count"] = df["amenities"].fillna("[]").str.count(",") + 1
    df["amenities_list"] = df["amenities"].apply(
        lambda x: ast.literal_eval(x) if isinstance(x, str) else []
    )
    top_amenities = [
        a for a, _ in Counter(
            [am for sublist in df["amenities_list"] for am in sublist]
        ).most_common(10)
    ]
    base = df["amenities"].fillna("[]")
    for amenity in top_amenities:
        col_name = f"amenity_{amenity.replace(' ', '_').lower()}"
        df[col_name] = base.str.contains(amenity, case=False, regex=False).fillna(False).astype(bool)
    df.drop(columns=["amenities", "amenities_list"], inplace=True)
    return df


def apply_outlier_filters(df):
    # chosen from cleaning notebook
    PRICE_MIN = 10.0
    PRICE_MAX = 2000.0
    PPG_MAX   = 500.0  # price per guest

    ppg = df["price"] / df["accommodates"].clip(lower=1)
    mask = df["price"].between(PRICE_MIN, PRICE_MAX) & (ppg <= PPG_MAX)
    return df.loc[mask].copy()


def filter_short_stays(df, max_min_nights=27):
    """Keep listings with minimum_nights <= max_min_nights (default: <28)."""
    return df[df["minimum_nights"] <= max_min_nights].copy()


def add_property_type_slim(df):
    """
    Add df['property_type_slim'] using a persisted JSON mapping.
    Expects docs/property_type_map.json to exist.
    Any categories not present in the mapping are set to 'Other'.
    """
    if not MAP_PATH.exists():
        raise FileNotFoundError(
            f"Property-type mapping not found at {MAP_PATH}. "
            "Please commit docs/property_type_map.json."
        )

    with open(MAP_PATH, "r") as f:
        payload = json.load(f)

    prop_map = payload.get("mapping", payload)

    out = df.copy()
    out["property_type_slim"] = out["property_type"].map(prop_map).fillna("Other")
    return out


def save_to_postgres(df):
    print("Saving to Postgres (clean.listings_features)...")
    engine = create_engine("postgresql://airbnb:airbnb@localhost:5432/airbnb")
    with engine.begin() as conn:
        conn.execute(text("CREATE SCHEMA IF NOT EXISTS clean"))
    df.to_sql(
        "listings_features",
        engine,
        schema="clean",
        if_exists="replace",
        index=False,
        dtype={"id": BigInteger()},
    )
    print("✅ Done.")


def main():
    df = load_raw_listings()
    df = select_columns(df)
    df = convert_numeric(df)
    df = handle_missing(df)
    df = engineer_features(df)
    df = apply_outlier_filters(df)
    df = filter_short_stays(df, max_min_nights=27)
    df = add_property_type_slim(df)
    save_to_postgres(df)


if __name__ == "__main__":
    main()
