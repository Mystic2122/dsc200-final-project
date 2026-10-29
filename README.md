# Personalized Restaurant Recommendation Using Yelp Data

DSC 200 final project. See [project-proposal.md](project-proposal.md).

## Setup

```powershell
py -3.13 -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
```

## Data

Download the **JSON** dataset from the [Yelp Open Dataset](https://business.yelp.com/data/resources/open-dataset/)
(you agree to Yelp's Dataset Terms of Use, which allow non-commercial educational use only) and extract it into `data/raw/`:

```
data/raw/yelp_academic_dataset_business.json
data/raw/yelp_academic_dataset_review.json
```

`data/` is gitignored. **Never commit or share the raw or processed data.**

## Pipeline

```powershell
python -m restaurant_rec.preprocess      # JSON -> cleaned data/processed/*.parquet (restaurants only), ~30 s
python -m restaurant_rec.text_index      # TF-IDF index over each restaurant's reviews, ~2 min
streamlit run app.py                     # open the UI
```

The dataset covers 12 metro areas (Philadelphia, Tampa, Nashville, Indianapolis, St. Louis, New Orleans,
Tucson, Reno, Edmonton, Santa Barbara, Boise, Wilmington), not the whole US.

## Data cleaning

- Drops ~480 non-restaurants that carry a "Restaurants" tag (hotels, casinos, museums, malls, tours)
- Assigns each restaurant to its metro area by location (also fixes mistyped state codes)
- Normalizes city spellings (`PHILA`, `Philly`, `Philadelphia ` -> `Philadelphia`)
- Treats Yelp's `0:0-0:0` hours as *closed* when the restaurant has real hours on other days
- Merges duplicate listings (same name + address) and their reviews
- Keeps only each user's latest review of a restaurant

## How recommendations work (hybrid)

1. **Hard filters**: metro area/town, price range, required attributes (parking, takeout, ...), open at a given time, minimum review count, distance from a ZIP code.
2. **Weighted score**: each feature is scaled to [0, 1] and combined with user-set weights:
   - *quality*: Bayesian-adjusted rating, which pulls restaurants with few reviews toward the global mean
   - *popularity*: log review count
   - *price*: cheaper scores higher
   - *distance*: closeness to the chosen ZIP code
   - *text*: TF-IDF cosine similarity between the user's "craving" text and/or liked restaurants and each restaurant's reviews
3. **Group mode** (`Recommender.recommend_group`): everyone's filters must pass; scores are combined by mean or least misery (min).

## Layout

| Path | Purpose |
|---|---|
| `restaurant_rec/attributes.py` | Parses Yelp's string-encoded attributes (`"{'garage': False, ...}"`, `"u'free'"`) |
| `restaurant_rec/cleaning.py` | Data-cleaning rules (venues, metros, cities, hours, duplicates) |
| `restaurant_rec/preprocess.py` | Filters to restaurants; DuckDB streams the 5 GB review file to Parquet |
| `restaurant_rec/text_index.py` | Builds the per-restaurant TF-IDF matrix |
| `restaurant_rec/recommender.py` | Filtering, scoring, group mode |
| `app.py` | Streamlit UI |
