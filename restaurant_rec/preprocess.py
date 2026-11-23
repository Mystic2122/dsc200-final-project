"""Step 1: raw Yelp JSON -> cleaned, restaurant-only Parquet files.

    python -m restaurant_rec.preprocess

Outputs (in data/processed/):
    restaurants.parquet  one row per restaurant, attributes flattened, metro area assigned
    reviews.parquet      one review per (user, restaurant) for those restaurants
"""

import json
import time

import duckdb
import pandas as pd

from . import cleaning
from .attributes import flatten_attributes
from .paths import (BUSINESS_JSON, BUSINESSES_PARQUET, PROCESSED_DIR,
                    REVIEW_JSON, REVIEWS_PARQUET)


def load_restaurants():
    # The business file is small (~120 MB), so pandas handles it fine.
    all_biz = pd.read_json(BUSINESS_JSON, lines=True)
    biz = all_biz[all_biz["categories"].fillna("").str.contains(r"\bRestaurants\b", regex=True)].copy()
    log = {"tagged Restaurants": len(biz)}

    biz["name"] = cleaning.clean_names(biz["name"])
    biz["city"] = cleaning.clean_cities(biz["city"])

    venue = cleaning.is_non_restaurant_venue(biz, cleaning.strong_food_categories(all_biz))
    biz = biz[~venue]
    log["dropped non-restaurant venues"] = int(venue.sum())

    n = len(biz)
    biz = cleaning.assign_metro(biz)
    log["dropped outside metros"] = n - len(biz)

    biz, id_map = cleaning.dedupe(biz)
    log["merged duplicate listings"] = len(id_map)

    flat = pd.DataFrame([flatten_attributes(a) for a in biz["attributes"]], index=biz.index)
    hours = biz["hours"].apply(lambda h: h if isinstance(h, dict) else {})
    hours_cols = pd.DataFrame({f"hours_{d}": hours.apply(lambda h, d=d: h.get(d)) for d in cleaning.DAYS})

    keep = ["business_id", "name", "address", "city", "state", "metro", "postal_code",
            "latitude", "longitude", "stars", "review_count", "is_open", "categories"]
    out = pd.concat([biz[keep], flat, hours_cols], axis=1).reset_index(drop=True)
    out = cleaning.clean_hours(out)
    out["hours_json"] = out[[f"hours_{d}" for d in cleaning.DAYS]].apply(
        lambda r: json.dumps({d: r[f"hours_{d}"] for d in cleaning.DAYS if pd.notna(r[f"hours_{d}"])}), axis=1)
    return out, id_map, log


def write_reviews(con, id_map):
    """Stream the ~5 GB review file through DuckDB.

    - reviews of merged duplicate listings are moved to the kept listing
    - one review per (user, restaurant): the latest (3.5% of rows are repeat reviews)
    - negative `useful` counts clipped to 0
    """
    con.register("id_map", id_map)
    con.execute(f"""
        COPY (
            WITH raw AS (
                SELECT r.review_id, r.user_id,
                       coalesce(m.new_id, r.business_id) AS business_id,
                       r.stars::TINYINT AS stars, greatest(r.useful, 0) AS useful,
                       r.date::TIMESTAMP AS date, r.text
                FROM read_json('{REVIEW_JSON.as_posix()}',
                               format = 'newline_delimited',
                               columns = {{review_id: 'VARCHAR', user_id: 'VARCHAR',
                                          business_id: 'VARCHAR', stars: 'DOUBLE',
                                          useful: 'INTEGER', funny: 'INTEGER', cool: 'INTEGER',
                                          text: 'VARCHAR', date: 'VARCHAR'}}) r
                LEFT JOIN id_map m ON r.business_id = m.old_id
            )
            SELECT * FROM raw
            SEMI JOIN read_parquet('{BUSINESSES_PARQUET.as_posix()}') b USING (business_id)
            QUALIFY row_number() OVER (PARTITION BY user_id, business_id
                                       ORDER BY date DESC, review_id) = 1
        ) TO '{REVIEWS_PARQUET.as_posix()}' (FORMAT parquet, COMPRESSION zstd)
    """)


def add_review_stats(con):
    # Yelp's `stars` is rounded to the nearest half star; recompute the exact mean.
    stats = con.execute(f"""
        SELECT business_id, AVG(stars) AS avg_stars, COUNT(*) AS n_reviews
        FROM read_parquet('{REVIEWS_PARQUET.as_posix()}')
        GROUP BY business_id
    """).df()
    biz = pd.read_parquet(BUSINESSES_PARQUET)
    biz = biz.merge(stats, on="business_id", how="left")
    biz["n_reviews"] = biz["n_reviews"].fillna(0).astype(int)
    biz["avg_stars"] = biz["avg_stars"].fillna(biz["stars"])
    biz.to_parquet(BUSINESSES_PARQUET, index=False)
    return biz


def main():
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    t0 = time.time()

    print("Loading and cleaning businesses...")
    restaurants, id_map, log = load_restaurants()
    restaurants.to_parquet(BUSINESSES_PARQUET, index=False)
    for step, n in log.items():
        print(f"  {step}: {n:,}")
    print(f"  -> {len(restaurants):,} restaurants ({time.time() - t0:.0f}s)")

    print("Filtering reviews...")
    con = duckdb.connect()
    write_reviews(con, id_map)
    restaurants = add_review_stats(con)
    n_reviews = con.execute(f"SELECT COUNT(*) FROM read_parquet('{REVIEWS_PARQUET.as_posix()}')").fetchone()[0]
    print(f"  -> {n_reviews:,} reviews ({time.time() - t0:.0f}s)")
    print("By metro:")
    print(restaurants["metro"].value_counts().sort_index().to_string())


if __name__ == "__main__":
    main()
