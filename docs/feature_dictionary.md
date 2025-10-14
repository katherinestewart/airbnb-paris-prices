# Airbnb Paris – Cleaned Data Dictionary

## How to regenerate
```bash
make db        # run from root with Docker running. raw csv -> Postgres
make clean     # raw schema -> clean schema
make verify_clean
```

## `make verify_clean` Output

| table             | cols |  rows  | min_price | max_price |
|:------------------|-----:|-------:|----------:|----------:|
| listings_features |   34 |  53326 |        15 |      2000 |
| reviews           |    6 | 2173219 |           |           |
| reviews_summary   |    6 |   64498 |           |           |


## Clean Listings Steps

**Type casting:** numerics from text (latitude, longitude, accommodates, bedrooms, beds, bathrooms, minimum_nights, maximum_nights, review scores, reviews_per_month).

**Price parsing:** stripped currency/symbols and converted to float.

**Missing values:**
- Drop rows with missing price.
- Median impute: beds, bathrooms, bedrooms.
- host_is_superhost: Converted t to True, f to False and assumed missing = False.

**Amenities:**

- Added columns for top 10 amenities, one-hot encoded (Boolean)
- Added column `amenities_count`: Count of number of amenities listed for property

**Price Outliers:**
- Dropped any rows with prices not lying between 10 and 2000
- Dropped any rows with price per guest > 500
- Caps used: PRICE_MIN=10, PRICE_MAX=2000, PPG_MAX=500

**Location:** verified all rows within Paris, no filter applied

**Nights sanity checks:**
- minimum_nights > 0
- maximum_nights ≥ minimum_nights

**Review scores:** may be NULL for listings with no reviews.

## `clean.listings_features` Columns

id _BIGINT_\
neighbourhood_cleansed _TEXT_\
latitude _DOUBLE PRECISION_\
longitude _DOUBLE PRECISION_\
property_type _TEXT_\
room_type _TEXT_\
accommodates _BIGINT_\
bedrooms _DOUBLE PRECISION_\
beds _DOUBLE PRECISION_\
bathrooms _DOUBLE PRECISION_\
price _DOUBLE PRECISION_\
minimum_nights _BIGINT_\
maximum_nights _BIGINT_\
host_is_superhost _BOOLEAN_\
review_scores_rating _DOUBLE PRECISION_\
review_scores_accuracy _DOUBLE PRECISION_\
review_scores_cleanliness _DOUBLE PRECISION_\
review_scores_checkin _DOUBLE PRECISION_\
review_scores_communication _DOUBLE PRECISION_\
review_scores_location _DOUBLE PRECISION_\
review_scores_value _DOUBLE PRECISION_\
reviews_per_month _DOUBLE PRECISION_\
amenities_count _BIGINT_\
amenity_kitchen _BOOLEAN_\
amenity_wifi _BOOLEAN_\
amenity_hot_water _BOOLEAN_\
amenity_smoke_alarm _BOOLEAN_\
amenity_hair_dryer _BOOLEAN_\
amenity_dishes_and_silverware _BOOLEAN_\
amenity_bed_linens _BOOLEAN_\
amenity_cooking_basics _BOOLEAN_\
amenity_essentials _BOOLEAN_\
amenity_iron _BOOLEAN_\
property_type_slim _TEXT_

## Clean Reviews Steps

**Type casting:** date -> datetime; ids stay numeric.

**Missing values:** fill reviewer_name="Unknown", comments="".

**Row filter:** drop rows with missing id, listing_id or date.

**Outputs:**
- write clean.reviews (cleaned rows)
- write clean.reviews_summary

## `clean.reviews` Columns

listing_id _BIGINT_\
id _BIGINT_\
date _TIMESTAMP WITHOUT TIME ZONE_\
reviewer_id _BIGINT_\
reviewer_name _TEXT_\
comments _TEXT_

## `clean.reviews_summary` Columns

listing_id _BIGINT_\
n_reviews _BIGINT_\
first_review _TIMESTAMP WITHOUT TIME ZONE_\
last_review _TIMESTAMP WITHOUT TIME ZONE_\
avg_comment_length _DOUBLE PRECISION_\
days_since_last_review _BIGINT_

**Join key:** clean.listings_features.id <-> clean.reviews_summary.listing_id
