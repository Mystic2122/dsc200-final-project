"""Offline accuracy evaluation: can the recommender predict where real Yelp users went next?

    python -m restaurant_rec.evaluate               # 2000 users, k = 10 (~1-2 min)
    python -m restaurant_rec.evaluate --users 300 --k 5

Protocol (temporal hold-out, per user):
  * Sample users with >= MIN_REVIEWS reviews in one metro (their "home" metro); only those reviews are used.
  * Sort them by date. The oldest 80% are the user's past; the newest 20% are hidden (the test set).
  * "Liked" = past restaurants rated >= 4 stars. They are passed to the recommender as `liked_ids`,
    exactly like the "Places you already like" box in the UI.
  * "Relevant" = hidden restaurants rated >= 4 stars. Users with no liked or no relevant ones are skipped.
  * Candidates = every restaurant in the home metro that was operating at the split date (first review
    before it, last review after it), minus places the user already reviewed in the past.
  * Each scoring setup ranks the candidates; we check how many relevant restaurants land in the top k.
  * The user's own hidden ratings are removed from avg_stars / n_reviews before scoring (no leakage
    through the rating or popularity features).

Metrics (averaged over users):
  hit rate@k  = share of users with at least one relevant restaurant in their top k
  precision@k = relevant restaurants in the top k / k
  recall@k    = relevant restaurants in the top k / all relevant restaurants of that user
"""

import argparse
import time

import duckdb
import numpy as np
import pandas as pd

from .paths import BUSINESSES_PARQUET, REVIEWS_PARQUET, ROOT
from .recommender import BAYES_PRIOR_REVIEWS, Preferences, Recommender

MIN_REVIEWS = 10      # reviews a user needs in their home metro
TEST_FRACTION = 0.2   # newest share of each user's reviews that is hidden
LIKED_STARS = 4       # a rating >= this counts as "liked" / relevant
RESULTS_CSV = ROOT / "results" / "evaluation.csv"

FEATURES = ["quality", "popularity", "price", "text"]
# Scoring setups to compare (weights on the recommender's features; None = random ranking).
SETUPS = {
    "Random (baseline)":              None,
    "Popularity only":                {"popularity": 1},
    "Rating only":                    {"quality": 1},
    "Liked-places text only":         {"text": 1},
    "Rating + popularity":            {"quality": 1, "popularity": 1},
    "Hybrid (UI defaults)":           {"quality": 2, "popularity": 0.5, "text": 2},  # High / Low / High
    "Hybrid (popularity + text)":     {"popularity": 1, "text": 1},
    "Hybrid (equal weights)":         {"quality": 1, "popularity": 1, "text": 1},
}


def load_data(n_users: int, seed: int):
    """Sampled users (user_id, metro), their home-metro reviews, and each restaurant's active period."""
    con = duckdb.connect()
    con.execute(f"""CREATE VIEW r AS SELECT user_id, business_id, stars, date
                    FROM read_parquet('{REVIEWS_PARQUET.as_posix()}')""")
    con.execute(f"""CREATE VIEW b AS SELECT business_id, metro
                    FROM read_parquet('{BUSINESSES_PARQUET.as_posix()}')""")
    users = con.execute(f"""
        WITH per_metro AS (SELECT user_id, metro, count(*) AS n FROM r JOIN b USING (business_id) GROUP BY ALL)
        SELECT user_id, metro FROM per_metro
        WHERE n >= {MIN_REVIEWS}
        QUALIFY row_number() OVER (PARTITION BY user_id ORDER BY n DESC, metro) = 1
        ORDER BY user_id
    """).df()
    n_eligible = len(users)
    users = users.sample(min(n_users, n_eligible), random_state=seed)
    con.register("sampled", users)
    reviews = con.execute("""
        SELECT r.* FROM r JOIN b USING (business_id) JOIN sampled s ON s.user_id = r.user_id AND s.metro = b.metro
        ORDER BY r.user_id, r.date
    """).df()
    active = con.execute("SELECT business_id, min(date) AS first, max(date) AS last FROM r GROUP BY 1").df()
    return users, reviews, active.set_index("business_id"), n_eligible


def remove_own_ratings(cands: pd.DataFrame, own: pd.Series, global_mean: float) -> pd.DataFrame:
    """Recompute avg_stars / n_reviews / bayes_rating without the user's hidden ratings."""
    stars = cands["business_id"].map(own)
    has = stars.notna()
    n = cands["n_reviews"] - has
    avg = (cands["avg_stars"] * cands["n_reviews"] - stars.fillna(0)) / n.clip(lower=1)
    cands = cands.assign(n_reviews=n, avg_stars=avg)
    cands["bayes_rating"] = (n * avg + BAYES_PRIOR_REVIEWS * global_mean) / (n + BAYES_PRIOR_REVIEWS)
    return cands


def evaluate(n_users=2000, k=10, seed=42):
    t0 = time.time()
    rec = Recommender.load()
    if rec.index is None:
        raise SystemExit("No text index found. Run `python -m restaurant_rec.text_index` first.")
    users, reviews, active, n_eligible = load_data(n_users, seed)
    print(f"{n_eligible:,} users have >= {MIN_REVIEWS} reviews in one metro; sampled {len(users):,} "
          f"({time.time() - t0:.0f}s)")

    all_on = {name: 1.0 for name in FEATURES}
    home_metro = dict(zip(users["user_id"], users["metro"]))
    rows, skipped = [], 0
    for user_id, user_reviews in reviews.groupby("user_id", sort=False):
        metro = home_metro[user_id]
        n_test = max(1, round(len(user_reviews) * TEST_FRACTION))
        past, test = user_reviews.iloc[:-n_test], user_reviews.iloc[-n_test:]
        liked = past.loc[past["stars"] >= LIKED_STARS, "business_id"].tolist()
        relevant = set(test.loc[test["stars"] >= LIKED_STARS, "business_id"])
        if not liked or not relevant:
            skipped += 1
            continue

        split = test["date"].min()
        p = Preferences(metro=metro, include_closed=True, liked_ids=liked, weights=all_on)
        cands = rec.filter(p)
        span = active.loc[cands["business_id"]]
        operating = (span["first"].to_numpy() < split) & (span["last"].to_numpy() >= split)
        cands = cands[operating & ~cands["business_id"].isin(past["business_id"]).to_numpy()]
        cands = remove_own_ratings(cands, test.set_index("business_id")["stars"], rec.global_mean)

        # Score once with every feature on, then re-weight: same formula as Recommender.score.
        f = rec.score(cands, p)[[f"f_{name}" for name in FEATURES]].to_numpy()
        ids = cands["business_id"].to_numpy()
        is_rel = np.isin(ids, list(relevant))
        info = {"user_id": user_id, "metro": metro, "n_candidates": len(ids), "n_liked": len(liked),
                "n_relevant": len(relevant)}
        for setup, weights in SETUPS.items():
            if weights is None:
                # Random ranking: use the exact expected values instead of one noisy draw
                # (with ~3,000 candidates a single random top 10 almost never hits).
                n, r = len(ids), is_rel.sum()
                p_miss = np.prod([(n - r - i) / (n - i) for i in range(min(k, n))]) if n - r >= k else 0.0
                rows.append({"setup": setup, **info, "hit": 1 - p_miss,
                             "precision": r / n, "recall": min(k, n) * r / n / len(relevant)})
                continue
            w = np.array([weights.get(name, 0.0) for name in FEATURES])
            top = np.argsort(-(f @ w / w.sum()), kind="stable")[:k]
            hits = is_rel[top].sum()
            rows.append({"setup": setup, **info, "hit": float(hits > 0),
                         "precision": hits / k, "recall": hits / len(relevant)})

    per_user = pd.DataFrame(rows)
    n_eval = per_user["user_id"].nunique()
    print(f"Evaluated {n_eval:,} users ({skipped} skipped: nothing liked in the past or nothing liked in the "
          f"hidden part); median {per_user['n_candidates'].median():,.0f} candidates and "
          f"{per_user['n_relevant'].median():.0f} relevant restaurants per user ({time.time() - t0:.0f}s)\n")

    summary = per_user.groupby("setup", sort=False).agg(
        hit_rate=("hit", "mean"), hit_rate_se=("hit", lambda h: h.std() / np.sqrt(len(h))),
        precision=("precision", "mean"), recall=("recall", "mean"))
    summary["lift_vs_random"] = summary["hit_rate"] / summary.loc["Random (baseline)", "hit_rate"]
    summary.columns = [f"hit_rate@{k}", "hit_rate_se", f"precision@{k}", f"recall@{k}", "lift_vs_random"]

    with pd.option_context("display.float_format", "{:.3f}".format, "display.width", 140):
        print(summary.to_string())
        best = summary.drop("Random (baseline)")[f"hit_rate@{k}"].idxmax()
        by_metro = (per_user[per_user["setup"] == best].groupby("metro")
                    .agg(users=("hit", "size"), hit_rate=("hit", "mean")).sort_values("users", ascending=False))
        print(f"\n{best}, hit rate@{k} by metro:")
        print(by_metro.to_string())

    RESULTS_CSV.parent.mkdir(exist_ok=True)
    summary.round(4).to_csv(RESULTS_CSV)
    print(f"\nSaved summary to {RESULTS_CSV.relative_to(ROOT)}  (metrics only, no Yelp data)")
    return summary, per_user


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--users", type=int, default=2000, help="number of users to sample")
    parser.add_argument("--k", type=int, default=10, help="length of the recommendation list")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    evaluate(args.users, args.k, args.seed)


if __name__ == "__main__":
    main()
