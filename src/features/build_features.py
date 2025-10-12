"""
Feature engineering utilities for the Airbnb Paris dataset.

"""
from math import radians, cos, sin, asin, sqrt
import pandas as pd
import numpy as np
from typing import List


def haversine_km(lon1, lat1, lon2, lat2):
    lon1, lat1, lon2, lat2 = map(radians, [lon1, lat1, lon2, lat2])
    dlon = lon2 - lon1
    dlat = lat2 - lat1
    a = sin(dlat / 2.0) ** 2 + cos(lat1) * cos(lat2) * sin(dlon / 2.0) ** 2
    c = 2 * asin(sqrt(a))
    R = 6371.0
    return R * c


def add_geofeatures(df: pd.DataFrame, center_lat: float = 48.8566, center_lon: float = 2.3522) -> pd.DataFrame:
    """
    Add distance to Paris centre (dist_to_center_km).
    """
    df = df.copy()
    if 'latitude' in df.columns and 'longitude' in df.columns:
        df['dist_to_center_km'] = df.apply(
            lambda r: haversine_km(r['longitude'], r['latitude'], center_lon, center_lat)
            if pd.notna(r['latitude']) and pd.notna(r['longitude']) else np.nan,
            axis=1
        )
    else:
        df['dist_to_center_km'] = np.nan
    return df


def add_amenity_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Ensure amenity booleans are numeric and compute amenities_count.
    """
    df = df.copy()
    amen_cols = [c for c in df.columns if c.startswith('amenity_')]
    for c in amen_cols:
        df[c] = df[c].apply(lambda v: 1 if (v is True or str(v).strip().lower() in ['true', '1', 't', 'yes', 'y']) else 0)
    if 'amenities_count' in df.columns:
        df['amenities_count'] = pd.to_numeric(df['amenities_count'], errors='coerce').fillna(df[amen_cols].sum(axis=1))
    else:
        df['amenities_count'] = df[amen_cols].sum(axis=1) if amen_cols else 0
    return df


def add_review_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Normalize review-related numeric fields and fill sensible defaults.
    """
    df = df.copy()
    review_cols = [
        'review_scores_rating', 'review_scores_accuracy', 'review_scores_cleanliness',
        'review_scores_checkin', 'review_scores_communication', 'review_scores_location',
        'review_scores_value', 'reviews_per_month', 'n_reviews', 'avg_comment_length', 'days_since_last_review'
    ]
    for c in review_cols:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors='coerce')
    if 'review_scores_rating' in df.columns:
        df['review_scores_rating'] = df['review_scores_rating'].fillna(df['review_scores_rating'].median())
    if 'reviews_per_month' in df.columns:
        df['reviews_per_month'] = df['reviews_per_month'].fillna(0)
    if 'avg_comment_length' in df.columns:
        df['avg_comment_length'] = df['avg_comment_length'].fillna(0)
    if 'days_since_last_review' in df.columns:
        df['days_since_last_review'] = df['days_since_last_review'].fillna(df['days_since_last_review'].max())
    return df


def build_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Apply all feature transforms and return augmented DataFrame.
    """
    df = df.copy()
    df = add_amenity_features(df)
    df = add_review_features(df)
    df = add_geofeatures(df)
    if 'price' in df.columns and 'accommodates' in df.columns:
        df['price_per_person'] = df['price'] / df['accommodates'].replace({0: 1})
    if 'host_is_superhost' in df.columns:
        df['host_is_superhost'] = df['host_is_superhost'].apply(
            lambda v: 1 if str(v).strip().lower() in ['t', 'true', 'yes', 'y', '1'] else 0
        )
    return df
