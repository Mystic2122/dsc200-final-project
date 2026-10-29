"""Streamlit UI.   Run with:  streamlit run app.py"""

from datetime import datetime
from urllib.parse import quote_plus

import streamlit as st

from restaurant_rec.paths import BUSINESSES_PARQUET
from restaurant_rec.recommender import DAYS, Preferences, Recommender, active_weights

st.set_page_config(page_title="Restaurant Recommender", page_icon="🍽️", layout="wide",
                   initial_sidebar_state="expanded")

# Fixed-width sidebar (no drag-to-resize) and sliders without min/max tick labels.
st.markdown("""<style>
[data-testid="stSidebar"][aria-expanded="true"], [data-testid="stSidebar"][data-collapsed="false"] {
    width: 340px !important; min-width: 340px !important; max-width: 340px !important; }
[data-testid="stSidebarResizeHandle"] { display: none !important; pointer-events: none !important; }
[data-testid="stSliderTickBar"] { display: none !important; }
</style>""", unsafe_allow_html=True)

if not BUSINESSES_PARQUET.exists():
    st.error("No processed data found. Run `python -m restaurant_rec.preprocess` first.")
    st.stop()


@st.cache_resource(show_spinner="Loading restaurants...")
def get_recommender():
    return Recommender.load()


rec = get_recommender()
biz = rec.biz

FEATURES = {
    "Parking": "parking_any", "Takeout": "takeout", "Delivery": "delivery",
    "Reservations": "reservations", "Outdoor seating": "outdoor_seating",
    "Kid friendly": "good_for_kids", "Groups": "good_for_groups",
    "Wheelchair accessible": "wheelchair_accessible", "Dogs allowed": "dogs_allowed",
}
FACTORS = {  # weight name -> label shown to the user
    "quality": "Rating", "popularity": "Popularity", "price": "Low price",
    "distance": "Close to ZIP", "text": "Matches craving",
}
IMPORTANCE = {"Off": 0.0, "Low": 0.5, "Med": 1.0, "High": 2.0}
DEFAULT_IMPORTANCE = {"quality": "High", "popularity": "Low", "price": "Off",
                      "distance": "Med", "text": "High"}
PRICES = ["$", "$$", "$$$", "$$$$"]
MIN_REVIEWS = {"Any": 0, "10+": 10, "50+": 50, "200+": 200}

# ---------- sidebar: filters ----------
with st.sidebar:
    st.header("Location", anchor=False)
    metro = st.selectbox("City", sorted(biz["metro"].unique()),
                         index=sorted(biz["metro"].unique()).index("Philadelphia, PA"))
    in_metro = biz[biz["metro"] == metro]
    town = st.selectbox("Town", ["All towns"] + sorted(in_metro["city"].unique()))
    zip_code = st.text_input("Near ZIP / postal code", placeholder="optional, e.g. 19103")
    radius = st.select_slider("Within", options=[1, 2, 5, 10, 25, 50], value=10,
                              format_func=lambda km: f"{km} km", disabled=not zip_code.strip())

    st.header("Filters", anchor=False)
    k = st.segmented_control("Results to show", [5, 10, 25, 50], default=10, required=True)
    price_lo, price_hi = st.select_slider("Price", options=PRICES, value=("$", "$$$$"))
    required = st.pills("Must have", list(FEATURES), selection_mode="multi")
    min_reviews = st.segmented_control("Minimum reviews", list(MIN_REVIEWS), default="10+", required=True)
    open_filter = st.toggle("Open at a specific time")
    if open_filter:
        now = datetime.now()
        day_col, time_col = st.columns(2)
        day = day_col.selectbox("Day", DAYS, index=now.weekday())
        when = time_col.time_input("Time", now.time().replace(second=0, microsecond=0), step=1800)

    st.header("What matters to you", anchor=False)
    weights = {name: IMPORTANCE[st.segmented_control(label, list(IMPORTANCE), default=DEFAULT_IMPORTANCE[name],
                                                     required=True, key=f"w_{name}")]
               for name, label in FACTORS.items()}

# ---------- main: personal taste ----------
st.title("Restaurant Recommender", anchor=False)
query = st.text_input("What are you craving?", placeholder="e.g. spicy ramen, cozy brunch, cheap tacos")

local = in_metro if town == "All towns" else in_metro[in_metro["city"] == town]
local = local.sort_values("n_reviews", ascending=False)
labels = dict(zip(local["business_id"], local["name"] + " — " + local["address"].fillna("") + ", " + local["city"]))
liked = st.multiselect("Places you already like (optional)", list(labels), format_func=labels.get,
                       max_selections=10, placeholder="Search restaurants...")

location = None
if zip_code.strip():
    location = rec.zip_location(zip_code)
    if location is None:
        st.warning(f"No restaurants found with ZIP / postal code {zip_code.strip()}, so it is ignored.")

prefs = Preferences(
    metro=metro,
    city=None if town == "All towns" else town,
    min_price=PRICES.index(price_lo) + 1,
    max_price=PRICES.index(price_hi) + 1,
    required=[FEATURES[f] for f in required],
    open_day=day if open_filter else None,
    open_minutes=when.hour * 60 + when.minute if open_filter else None,
    location=location,
    radius_km=radius if location else None,
    min_reviews=MIN_REVIEWS[min_reviews],
    query=query,
    liked_ids=liked,
    weights=weights,
)

results = rec.recommend(prefs, k)

if results.empty:
    st.warning("No restaurants match those filters. Try loosening them.")
    st.stop()

used = {name: w for name, w in active_weights(prefs).items() if w > 0}
st.caption(f"Top {len(results)} matches · scored on " + ", ".join(FACTORS[n].lower() for n in used))
st.map(results.rename(columns={"latitude": "lat", "longitude": "lon"})[["lat", "lon"]])

for rank, (_, r) in enumerate(results.iterrows(), start=1):
    with st.container(border=True):
        left, right = st.columns([4, 1])
        price = PRICES[int(r["price_range"]) - 1] if r["price_range"] == r["price_range"] else "price n/a"
        maps = "https://www.google.com/maps/search/?api=1&query=" + quote_plus(
            f"{r['name']} {r['address']} {r['city']} {r['state']}")
        left.subheader(f"{rank}. {r['name']}", anchor=False)
        left.write(f"⭐ {r['avg_stars']:.1f} ({r['n_reviews']:,} reviews) · {price} · {r['categories']}")
        distance = f" · {r['distance_km']:.1f} km away" if "distance_km" in r else ""
        left.markdown(f"{r['address']}, {r['city']}, {r['state']}{distance} · [Open in Google Maps]({maps})")
        right.metric("Match score", f"{r['score'] * 100:.0f}")
        right.caption(" · ".join(f"{FACTORS[n]} {r['f_' + n]:.2f}" for n in used))
