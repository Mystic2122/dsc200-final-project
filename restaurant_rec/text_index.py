"""Step 2: build a TF-IDF "document" per restaurant from its reviews.

    python -m restaurant_rec.text_index

Each restaurant's document is its name + categories + its most useful reviews.
Used for "describe what you're craving" search and "more like these" matching.
"""

import time

import duckdb
import joblib
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer

from .paths import BUSINESSES_PARQUET, REVIEWS_PARQUET, TEXT_INDEX

REVIEWS_PER_RESTAURANT = 25


def build_documents():
    con = duckdb.connect()
    docs = con.execute(f"""
        WITH top_reviews AS (
            SELECT business_id, text
            FROM read_parquet('{REVIEWS_PARQUET.as_posix()}')
            WHERE length(trim(text)) >= 20
            QUALIFY row_number() OVER (PARTITION BY business_id
                                       ORDER BY useful DESC, date DESC) <= {REVIEWS_PER_RESTAURANT}
        )
        SELECT business_id, string_agg(text, ' ') AS review_text
        FROM top_reviews
        GROUP BY business_id
    """).df()
    biz = pd.read_parquet(BUSINESSES_PARQUET, columns=["business_id", "name", "categories"])
    biz = biz.merge(docs, on="business_id", how="left")
    # Repeat name/categories so they carry weight next to long review text.
    biz["document"] = ((biz["name"] + " ") * 2 + (biz["categories"].fillna("") + " ") * 3
                       + biz["review_text"].fillna(""))
    return biz[["business_id", "document"]]


def main():
    t0 = time.time()
    print("Building restaurant documents...")
    docs = build_documents()
    print(f"  {len(docs):,} documents ({time.time() - t0:.0f}s)")

    print("Fitting TF-IDF...")
    vectorizer = TfidfVectorizer(stop_words="english", ngram_range=(1, 2), min_df=5,
                                 max_df=0.5, max_features=100_000, sublinear_tf=True,
                                 dtype=np.float32)
    matrix = vectorizer.fit_transform(docs["document"])
    joblib.dump({"business_ids": docs["business_id"].tolist(),
                 "vectorizer": vectorizer,
                 "matrix": matrix}, TEXT_INDEX)
    print(f"  matrix {matrix.shape}, saved to {TEXT_INDEX} ({time.time() - t0:.0f}s)")


if __name__ == "__main__":
    main()
