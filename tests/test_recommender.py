"""Tests for the restaurant recommender.

Unit tests (UT-*) use small hand-made inputs, so they run without the Yelp data.
Functional tests (FT-*) run the full recommender on the processed Yelp data and are skipped if
data/processed/ has not been built (see README).

Run:  python -m pytest -v
"""

import time

import numpy as np
import pandas as pd
import pytest

from restaurant_rec.attributes import flatten_attributes, parse_value
from restaurant_rec.cleaning import clean_cities, clean_hours, dedupe
from restaurant_rec.paths import BUSINESSES_PARQUET, TEXT_INDEX
from restaurant_rec.recommender import (Preferences, Recommender, active_weights,
                                        is_open_at)

DAYS = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]


# ---------------------------------------------------------------- unit tests

def test_ut1_parse_attributes():
    """UT-1: Yelp's string-encoded attributes are parsed into typed columns."""
    assert parse_value("u'free'") == "free"
    assert parse_value("None") is None
    row = flatten_attributes({
        "RestaurantsPriceRange2": "2",
        "RestaurantsTakeOut": "True",
        "WiFi": "u'free'",
        "BusinessParking": "{'garage': False, 'street': True, 'lot': None}",
    })
    assert row["price_range"] == 2
    assert row["takeout"] is True
    assert row["wifi"] == "free"
    assert row["parking_street"] is True and row["parking_garage"] is False
    assert row["parking_any"] is True
    assert row["delivery"] is None  # missing attribute -> unknown, not False


def test_ut2_open_hours_past_midnight():
    """UT-2: is_open_at handles normal hours, past-midnight spans, closed days and 24h."""
    bar = {f"hours_{d}": None for d in DAYS}
    bar["hours_Friday"] = "17:0-2:0"           # Friday 5 pm to Saturday 2 am
    assert is_open_at(bar, "Friday", 23 * 60)       # Fri 11 pm
    assert is_open_at(bar, "Saturday", 60)          # Sat 1 am (spill-over)
    assert not is_open_at(bar, "Saturday", 3 * 60)  # Sat 3 am
    assert not is_open_at(bar, "Friday", 12 * 60)   # Fri noon
    assert not is_open_at(bar, "Monday", 20 * 60)   # no hours Monday = closed
    diner = {f"hours_{d}": "0:0-0:0" for d in DAYS}
    assert is_open_at(diner, "Wednesday", 4 * 60)   # open 24h


def test_ut3_clean_hours_zero_means_closed():
    """UT-3: "0:0-0:0" means closed that day unless every listed day is "0:0-0:0" (open 24h)."""
    cols = [f"hours_{d}" for d in DAYS]
    biz = pd.DataFrame([["0:0-0:0"] + ["11:0-21:0"] * 6, ["0:0-0:0"] * 7], columns=cols)
    out = clean_hours(biz)
    assert pd.isna(out.loc[0, "hours_Monday"])          # closed Monday
    assert out.loc[0, "hours_Tuesday"] == "11:0-21:0"
    assert (out.loc[1, cols] == "0:0-0:0").all()        # 24h kept


def test_ut4_dedupe_keeps_open_listing():
    """UT-4: duplicate listings (same name + address) merge into the open one with most reviews."""
    biz = pd.DataFrame({
        "business_id": ["old", "new", "other"],
        "name": ["Joe's Pizza", "Joe's Pizza", "Joe's Pizza"],
        "address": ["1 Main St", "1 main st ", "9 Elm St"],
        "is_open": [0, 1, 1],
        "review_count": [300, 40, 10],
    })
    kept, id_map = dedupe(biz)
    assert set(kept["business_id"]) == {"new", "other"}
    assert id_map.to_dict("records") == [{"old_id": "old", "new_id": "new"}]


def test_ut5_clean_cities():
    """UT-5: city spelling variants map to one canonical name."""
    raw = pd.Series(["Philadelphia", "PHILA", "Philadelphia ", "St.Louis", "Saint Louis", "Tampa", "Tampa, FL"])
    assert list(clean_cities(raw)) == ["Philadelphia"] * 3 + ["St. Louis"] * 2 + ["Tampa"] * 2


def _toy_recommender():
    biz = pd.DataFrame({
        "business_id": list("ABCD"),
        "name": ["A", "B", "C", "D"],
        "state": ["PA", "PA", "PA", "FL"],
        "metro": ["Philadelphia, PA"] * 3 + ["Tampa, FL"],
        "city": ["Philadelphia"] * 3 + ["Tampa"],
        "is_open": [1, 1, 0, 1],
        "price_range": [1.0, 4.0, 2.0, np.nan],
        "takeout": [True, None, True, True],
        "avg_stars": [4.5, 3.0, 5.0, 4.0],
        "n_reviews": [500, 20, 100, 50],
        "latitude": [39.95] * 3 + [27.95],
        "longitude": [-75.16] * 3 + [-82.46],
        **{f"hours_{d}": "11:0-21:0" for d in DAYS},
    })
    return Recommender(biz)


def test_ut6_hard_filters():
    """UT-6: filter() drops closed, out-of-area, over-budget and missing-attribute restaurants."""
    rec = _toy_recommender()
    p = Preferences(state="PA", max_price=2, required=["takeout"])
    assert list(rec.filter(p)["business_id"]) == ["A"]  # B too pricey/no takeout, C closed, D in FL


def test_ut7_weighted_score_range_and_off():
    """UT-7: scores are in [0, 1]; Off features and features with no input get weight 0."""
    rec = _toy_recommender()
    p = Preferences(state="PA", weights={"quality": 2, "popularity": 0, "price": 1,
                                         "distance": 1, "text": 1})
    w = active_weights(p)
    assert w["distance"] == 0 and w["text"] == 0 and w["popularity"] == 0
    scored = rec.score(rec.filter(p), p)
    assert scored["score"].between(0, 1).all()
    expected = (2 * scored["f_quality"] + 1 * scored["f_price"]) / 3
    assert np.allclose(scored["score"], expected)


# ---------------------------------------------------------- functional tests

needs_data = pytest.mark.skipif(not BUSINESSES_PARQUET.exists() or not TEXT_INDEX.exists(),
                                reason="processed Yelp data not built")


@pytest.fixture(scope="module")
def rec():
    return Recommender.load()


@needs_data
def test_ft1_filters_respected(rec):
    """FT-1 (FR-1): every result satisfies the user's filters."""
    p = Preferences(state="PA", metro="Philadelphia, PA", min_price=1, max_price=2,
                    required=["parking_any", "takeout"], min_reviews=50)
    out = rec.recommend(p, k=50)
    assert len(out) == 50
    assert (out["state"] == "PA").all() and (out["metro"] == "Philadelphia, PA").all()
    assert (out["price_range"].isna() | (out["price_range"] <= 2)).all()
    assert out["parking_any"].eq(True).all() and out["takeout"].eq(True).all()
    assert (out["n_reviews"] >= 50).all() and (out["is_open"] == 1).all()


@needs_data
def test_ft2_ranking_follows_weights(rec):
    """FT-2 (FR-2): results are sorted by score; raising the price weight makes results cheaper."""
    base = dict(state="PA", metro="Philadelphia, PA", min_reviews=10)
    low = rec.recommend(Preferences(**base, weights={"quality": 1, "popularity": 0.5, "price": 0,
                                                     "distance": 0, "text": 0}), k=25)
    high = rec.recommend(Preferences(**base, weights={"quality": 1, "popularity": 0.5, "price": 2,
                                                      "distance": 0, "text": 0}), k=25)
    assert low["score"].is_monotonic_decreasing and high["score"].between(0, 1).all()
    assert high["price_range"].mean() < low["price_range"].mean()


@needs_data
def test_ft3_result_count_and_empty(rec):
    """FT-3 (FR-3): at most k results; impossible filters return an empty result."""
    for k in (5, 10, 25, 50):
        assert len(rec.recommend(Preferences(state="PA", metro="Philadelphia, PA"), k=k)) == k
    impossible = Preferences(state="PA", metro="Philadelphia, PA", min_reviews=10**6)
    assert rec.recommend(impossible, k=10).empty


@needs_data
def test_ft4_craving_text(rec):
    """FT-4 (FR-4): a "ramen" craving with High text weight returns ramen places."""
    p = Preferences(state="PA", metro="Philadelphia, PA", query="ramen",
                    weights={"quality": 1, "popularity": 0.5, "price": 0, "distance": 0, "text": 2})
    out = rec.recommend(p, k=10)
    is_ramen = out["categories"].str.contains("Ramen", na=False) | out["name"].str.contains("ramen", case=False)
    assert is_ramen.sum() >= 7


@needs_data
def test_ft5_responsiveness(rec):
    """FT-5 (NFR-2): recommend() finishes in under 2 s for the largest metro, with text and open-at."""
    p = Preferences(state="PA", metro="Philadelphia, PA", query="spicy noodles",
                    open_day="Friday", open_minutes=19 * 60)
    start = time.perf_counter()
    rec.recommend(p, k=50)
    assert time.perf_counter() - start < 2.0
