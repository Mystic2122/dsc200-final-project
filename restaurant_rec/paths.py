from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
RAW_DIR = ROOT / "data" / "raw"
PROCESSED_DIR = ROOT / "data" / "processed"

BUSINESS_JSON = RAW_DIR / "yelp_academic_dataset_business.json"
REVIEW_JSON = RAW_DIR / "yelp_academic_dataset_review.json"

BUSINESSES_PARQUET = PROCESSED_DIR / "restaurants.parquet"
REVIEWS_PARQUET = PROCESSED_DIR / "reviews.parquet"
TEXT_INDEX = PROCESSED_DIR / "text_index.joblib"
