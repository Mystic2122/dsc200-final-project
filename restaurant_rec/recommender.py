"""Hybrid restaurant recommender.

1. Hard filters   – location, price, required attributes, open hours, min reviews.
2. Weighted score – Bayesian-adjusted rating, popularity, low price, distance,
                    and TF-IDF text similarity to a query / liked restaurants.
   score = sum(w_i * feature_i) / sum(w_i), every feature scaled to [0, 1].
3. Group mode     – score per person, then combine with mean or least misery (min).
"""

from dataclasses import dataclass, field

import joblib
import numpy as np
import pandas as pd

from .paths import BUSINESSES_PARQUET, TEXT_INDEX

BAYES_PRIOR_REVIEWS = 20  # how many "average" reviews every restaurant starts with

DEFAULT_WEIGHTS = {"quality": 1.0, "popularity": 0.5, "price": 0.0,
                   "distance": 1.0, "text": 1.0}


@dataclass
class Preferences:
    state: str | None = None            # e.g. "PA"
    city: str | None = None
    metro: str | None = None            # e.g. "Philadelphia, PA"
    min_price: int | None = None        # 1-4 ($ to $$$$); restaurants with unknown price are kept
    max_price: int | None = None
    required: list[str] = field(default_factory=list)  # boolean columns that must be True
    open_day: str | None = None         # "Monday" ... "Sunday"
    open_minutes: int | None = None     # minutes after midnight
    location: tuple[float, float] | None = None  # (lat, lon)
    radius_km: float | None = None
    min_reviews: int = 0
    include_closed: bool = False        # include businesses Yelp marks permanently closed
    query: str = ""
    liked_ids: list[str] = field(default_factory=list)
    weights: dict[str, float] = field(default_factory=lambda: dict(DEFAULT_WEIGHTS))


# ---------- opening hours ----------

DAYS = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]


def _parse_span(span):
    """'11:0-21:30' -> (660, 1290) in minutes after midnight."""
    def to_min(t):
        h, m = t.split(":")
        return int(h) * 60 + int(m)
    start, end = span.split("-")
    return to_min(start), to_min(end)


def is_open_at(row, day, minutes):
    today = row.get(f"hours_{day}")
    if isinstance(today, str):
        start, end = _parse_span(today)
        if start == end:  # 0:0-0:0 = open 24h (cleaning removed the "closed" meaning)
            return True
        if start < end and start <= minutes < end:
            return True
        if end < start and minutes >= start:  # e.g. 17:00-02:00, before midnight
            return True
    yesterday = row.get(f"hours_{DAYS[DAYS.index(day) - 1]}")
    if isinstance(yesterday, str):
        start, end = _parse_span(yesterday)
        if end < start and minutes < end:  # spill-over past midnight
            return True
    return False


def haversine_km(lat1, lon1, lat2, lon2):
    lat1, lon1, lat2, lon2 = map(np.radians, (lat1, lon1, lat2, lon2))
    a = (np.sin((lat2 - lat1) / 2) ** 2
         + np.cos(lat1) * np.cos(lat2) * np.sin((lon2 - lon1) / 2) ** 2)
    return 6371 * 2 * np.arcsin(np.sqrt(a))


def active_weights(p: Preferences) -> dict[str, float]:
    """The user's weights, minus features that have no input (no location, no query...)."""
    weights = dict(p.weights)
    if p.location is None:
        weights["distance"] = 0
    if not p.query.strip() and not p.liked_ids:
        weights["text"] = 0
    return weights


# ---------- recommender ----------

class Recommender:
    def __init__(self, businesses: pd.DataFrame, text_index: dict | None = None):
        self.biz = businesses.reset_index(drop=True)
        self.global_mean = np.average(self.biz["avg_stars"],
                                      weights=self.biz["n_reviews"].clip(lower=1))
        n = self.biz["n_reviews"]
        self.biz["bayes_rating"] = ((n * self.biz["avg_stars"] + BAYES_PRIOR_REVIEWS * self.global_mean)
                                    / (n + BAYES_PRIOR_REVIEWS))

        self.index = text_index
        if text_index is not None:
            row_of = {bid: i for i, bid in enumerate(text_index["business_ids"])}
            self.text_rows = self.biz["business_id"].map(row_of).to_numpy()

    @classmethod
    def load(cls):
        biz = pd.read_parquet(BUSINESSES_PARQUET)
        index = joblib.load(TEXT_INDEX) if TEXT_INDEX.exists() else None
        return cls(biz, index)

    # --- step 1: hard filters ---
    def filter(self, p: Preferences) -> pd.DataFrame:
        df = self.biz
        mask = pd.Series(True, index=df.index)
        if not p.include_closed:
            mask &= df["is_open"] == 1
        if p.state:
            mask &= df["state"] == p.state
        if p.metro:
            mask &= df["metro"] == p.metro
        if p.city:
            mask &= df["city"] == p.city
        price = df["price_range"]
        if p.min_price:
            mask &= price.isna() | (price >= p.min_price)
        if p.max_price:
            mask &= price.isna() | (price <= p.max_price)
        for col in p.required:
            mask &= df[col].eq(True)
        if p.min_reviews:
            mask &= df["n_reviews"] >= p.min_reviews
        df = df[mask].copy()

        if p.location is not None:
            df["distance_km"] = haversine_km(p.location[0], p.location[1],
                                             df["latitude"], df["longitude"])
            if p.radius_km:
                df = df[df["distance_km"] <= p.radius_km]

        if p.open_day and p.open_minutes is not None and len(df):
            open_mask = df.apply(lambda r: is_open_at(r, p.open_day, p.open_minutes), axis=1)
            df = df[open_mask]
        return df

    # --- step 2: features + weighted score ---
    def text_similarity(self, df: pd.DataFrame, p: Preferences) -> np.ndarray:
        if self.index is None or (not p.query.strip() and not p.liked_ids):
            return np.zeros(len(df))
        matrix = self.index["matrix"]
        # User profile = query vector + mean vector of liked restaurants.
        profile = np.zeros(matrix.shape[1], dtype=np.float32)
        if p.query.strip():
            profile += self.index["vectorizer"].transform([p.query]).toarray().ravel()
        liked = self.text_rows[self.biz.index[self.biz["business_id"].isin(p.liked_ids)]]
        liked = liked[~np.isnan(liked)].astype(int)
        if len(liked):
            profile += np.asarray(matrix[liked].mean(axis=0)).ravel()

        rows = self.text_rows[df.index]
        sims = np.zeros(len(df))
        has_doc = ~np.isnan(rows)
        if has_doc.any():
            sims[has_doc] = matrix[rows[has_doc].astype(int)] @ profile
        if sims.max() > 0:
            sims = sims / sims.max()
        return sims

    def score(self, df: pd.DataFrame, p: Preferences) -> pd.DataFrame:
        df = df.copy()
        f = pd.DataFrame(index=df.index)
        f["quality"] = (df["bayes_rating"] - 1) / 4
        log_n = np.log1p(df["n_reviews"])
        f["popularity"] = log_n / log_n.max() if len(df) and log_n.max() > 0 else 0.0
        f["price"] = (1 - (df["price_range"] - 1) / 3).fillna(0.5)  # $ = 1, $$$$ = 0
        if "distance_km" in df and p.location is not None:
            scale = p.radius_km or max(df["distance_km"].max(), 1e-9)
            f["distance"] = (1 - df["distance_km"] / scale).clip(0, 1)
        else:
            f["distance"] = 0.0
        f["text"] = self.text_similarity(df, p)

        weights = active_weights(p)
        total = sum(weights.values()) or 1.0
        df["score"] = sum(weights[k] * f[k] for k in weights) / total
        for k in f:
            df[f"f_{k}"] = f[k]
        return df

    def recommend(self, p: Preferences, k: int = 10) -> pd.DataFrame:
        candidates = self.filter(p)
        if candidates.empty:
            return candidates
        scored = self.score(candidates, p)
        scored = scored[~scored["business_id"].isin(p.liked_ids)]
        return scored.sort_values("score", ascending=False).head(k)

    # --- step 3: group mode ---
    def recommend_group(self, people: list[Preferences], k: int = 10,
                        method: str = "least_misery") -> pd.DataFrame:
        """Restaurants must pass everyone's filters; scores combined by mean or min."""
        candidate_sets = [self.filter(p) for p in people]
        common = candidate_sets[0].index
        for c in candidate_sets[1:]:
            common = common.intersection(c.index)
        if len(common) == 0:
            return self.biz.iloc[0:0]
        per_person = [self.score(c.loc[common], p)["score"] for c, p in zip(candidate_sets, people)]
        scores = pd.concat(per_person, axis=1)
        out = candidate_sets[0].loc[common].copy()
        out["score"] = scores.min(axis=1) if method == "least_misery" else scores.mean(axis=1)
        return out.sort_values("score", ascending=False).head(k)
