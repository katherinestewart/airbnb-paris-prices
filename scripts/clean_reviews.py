import pandas as pd
from sqlalchemy import create_engine, text

ENGINE_URL = "postgresql://airbnb:airbnb@localhost:5432/airbnb"

def load_raw_reviews():
    print("📥 Loading raw.reviews from Postgres...")
    engine = create_engine(ENGINE_URL)
    return pd.read_sql("SELECT * FROM raw.reviews", engine)

def select_columns(df):
    keep = ["listing_id", "id", "date", "reviewer_id", "reviewer_name", "comments"]
    return df[keep].copy()

def convert_types(df):
    df["listing_id"]  = pd.to_numeric(df["listing_id"], errors="coerce")
    df["id"]          = pd.to_numeric(df["id"], errors="coerce")
    df["reviewer_id"] = pd.to_numeric(df["reviewer_id"], errors="coerce")
    df["date"]        = pd.to_datetime(df["date"], errors="coerce")
    return df

def handle_missing(df):
    df = df.dropna(subset=["listing_id", "id", "date"])
    df["reviewer_name"] = df["reviewer_name"].fillna("Unknown")
    df["comments"]      = df["comments"].fillna("")
    return df

def save_clean_reviews(df):
    print("💾 Saving clean.reviews ...")
    engine = create_engine(ENGINE_URL)
    with engine.begin() as conn:
        conn.execute(text("CREATE SCHEMA IF NOT EXISTS clean"))
    df.to_sql("reviews", engine, schema="clean", if_exists="replace", index=False)

def build_and_save_summary(df):
    print("🧮 Building clean.reviews_summary ...")
    summary = (
        df.groupby("listing_id")
          .agg(
              n_reviews=("id", "count"),
              first_review=("date", "min"),
              last_review=("date", "max"),
              # robust to any non-string
              avg_comment_length=("comments", lambda x: x.astype(str).str.len().mean()),
          )
          .reset_index()
    )
    latest_date = df["date"].max()
    summary["days_since_last_review"] = (latest_date - summary["last_review"]).dt.days

    engine = create_engine(ENGINE_URL)
    with engine.begin() as conn:
        conn.execute(text("CREATE SCHEMA IF NOT EXISTS clean"))
    summary.to_sql("reviews_summary", engine, schema="clean", if_exists="replace", index=False)

def main():
    df = load_raw_reviews()
    df = select_columns(df)
    df = convert_types(df)
    df = handle_missing(df)
    save_clean_reviews(df)
    build_and_save_summary(df)
    print("✅ clean.reviews + clean.reviews_summary ready.")

if __name__ == "__main__":
    main()
