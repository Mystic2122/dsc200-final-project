"""Streamlit UI.   Run with:  streamlit run app.py"""

from datetime import datetime
from urllib.parse import quote_plus

import pydeck as pdk
import streamlit as st

from restaurant_rec.paths import BUSINESSES_PARQUET
from restaurant_rec.recommender import DAYS, Preferences, Recommender, active_weights

st.set_page_config(page_title="Restaurant Recommender", layout="wide", initial_sidebar_state="expanded")

# Fixed-width, compact sidebar (no drag-to-resize), less top padding, sliders without min/max labels.
st.markdown("""<style>
[data-testid="stSidebar"][aria-expanded="true"], [data-testid="stSidebar"][data-collapsed="false"] {
    width: 340px !important; min-width: 340px !important; max-width: 340px !important; }
[data-testid="stSidebarResizeHandle"] { display: none !important; pointer-events: none !important; }
[data-testid="stSidebarHeader"] { height: 2.5rem !important; min-height: 0 !important; padding: 0.5rem 1.5rem 0 !important; }
[data-testid="stSidebarUserContent"] { padding-top: 0 !important; padding-bottom: 1rem !important; }
[data-testid="stSidebar"] [data-testid="stVerticalBlock"] { gap: 0.75rem !important; }
[data-testid="stMainBlockContainer"] { padding-top: 2.5rem !important; }
[data-testid="stSliderTickBar"] { display: none !important; }
.st-key-selected_card { border: 2px solid #e63946 !important; }
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
    "quality": "Rating", "popularity": "Popularity", "price": "Low price", "text": "Matches craving",
}
IMPORTANCE = {"Off": 0.0, "Low": 0.5, "Med": 1.0, "High": 2.0}
DEFAULT_IMPORTANCE = {"quality": "High", "popularity": "Low", "price": "Off", "text": "High"}
PRICES = ["$", "$$", "$$$", "$$$$"]
STATE_NAMES = {
    "AB": "Alberta", "AZ": "Arizona", "CA": "California", "DE": "Delaware", "FL": "Florida",
    "ID": "Idaho", "IL": "Illinois", "IN": "Indiana", "LA": "Louisiana", "MO": "Missouri",
    "NJ": "New Jersey", "NV": "Nevada", "PA": "Pennsylvania", "TN": "Tennessee",
}
MIN_REVIEWS = {"Any": 0, "10+": 10, "50+": 50, "200+": 200}


def md_escape(text):
    """Streamlit labels are Markdown, where "$$...$$" is LaTeX math; escape the dollar signs."""
    return text.replace("$", r"\$")


def short_categories(categories, n=4):
    cats = [c for c in categories.split(", ") if c not in ("Restaurants", "Food")]
    return ", ".join(cats[:n]) + (", ..." if len(cats) > n else "")

# ---------- sidebar: filters ----------
with st.sidebar:
    st.subheader("Location", anchor=False)
    states = sorted(biz["state"].unique(), key=STATE_NAMES.get)
    state = st.selectbox("State", states, index=states.index("PA"), format_func=STATE_NAMES.get)
    in_state = biz[biz["state"] == state]
    city_counts = in_state["city"].value_counts()  # biggest city first, the rest alphabetical
    cities = [city_counts.index[0]] + sorted(city_counts.index[1:])
    city = st.selectbox("City", ["All cities"] + cities, index=1)

    st.subheader("Filters", anchor=False)
    k = st.segmented_control("Results to show", [5, 10, 25, 50], default=10, required=True)
    price_lo, price_hi = st.select_slider("Price", options=PRICES, value=("$", "$$$$"), format_func=md_escape)
    required = st.multiselect("Must have", list(FEATURES), placeholder="Anything")
    min_reviews = st.segmented_control("Minimum reviews", list(MIN_REVIEWS), default="10+", required=True)
    open_filter = st.toggle("Open at a specific time")
    if open_filter:
        now = datetime.now()
        day_col, time_col = st.columns(2)
        day = day_col.selectbox("Day", DAYS, index=now.weekday())
        when = time_col.time_input("Time", now.time().replace(second=0, microsecond=0), step=1800)

    st.subheader("What matters to you", anchor=False)
    weights = {name: IMPORTANCE[st.segmented_control(label, list(IMPORTANCE), default=DEFAULT_IMPORTANCE[name],
                                                     required=True, key=f"w_{name}")]
               for name, label in FACTORS.items()}

# ---------- main: personal taste ----------
st.title("Restaurant Recommender", anchor=False)
query = st.text_input("What are you craving?", placeholder="e.g. spicy ramen, cozy brunch, cheap tacos")

local = in_state if city == "All cities" else in_state[in_state["city"] == city]
local = local.sort_values("n_reviews", ascending=False)
labels = dict(zip(local["business_id"], local["name"] + " — " + local["address"].fillna("") + ", " + local["city"]))
liked = st.multiselect("Places you already like (optional)", list(labels), format_func=labels.get,
                       max_selections=10, placeholder="Search restaurants...")

prefs = Preferences(
    state=state,
    city=None if city == "All cities" else city,
    min_price=PRICES.index(price_lo) + 1,
    max_price=PRICES.index(price_hi) + 1,
    required=[FEATURES[f] for f in required],
    open_day=day if open_filter else None,
    open_minutes=when.hour * 60 + when.minute if open_filter else None,
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

results = results.assign(rank=range(1, len(results) + 1))
results["price"] = [PRICES[int(p) - 1] if p == p else "price n/a" for p in results["price_range"]]

# ---------- map <-> list selection ----------
# The selected restaurant is kept in session state; it can be set by clicking a dot on the map
# or a card's "Show on map" button.
if st.session_state.get("selected_id") not in set(results["business_id"]):
    st.session_state["selected_id"] = None


def select_from_map():
    objects = st.session_state["map"].selection["objects"].get("restaurants", [])
    st.session_state["selected_id"] = objects[0]["business_id"] if objects else None


def select_from_list(business_id):
    st.session_state["selected_id"] = business_id


selected_id = st.session_state["selected_id"]


def build_map(results, selected_id):
    is_sel = results["business_id"] == selected_id
    points = results.assign(
        lat=results["latitude"], lon=results["longitude"], label=results["rank"].astype(str),
        stars=results["avg_stars"].round(1), reviews=results["n_reviews"].map("{:,}".format),
        color=[[230, 57, 70, 255] if s else [30, 110, 200, 230] for s in is_sel],
        radius=[16 if s else 11 for s in is_sel],
        is_sel=is_sel,
    )[["business_id", "lat", "lon", "label", "rank", "name", "stars", "reviews", "price", "color", "radius",
       "is_sel"]]
    # later rows are drawn on top: selected restaurant last, then #1, #2, ...
    points = points.sort_values(["is_sel", "rank"], ascending=[True, False]).drop(columns="is_sel")

    if selected_id is not None:
        row = points[points["business_id"] == selected_id].iloc[0]
        view = pdk.ViewState(latitude=row["lat"], longitude=row["lon"], zoom=15)
    else:
        view = pdk.data_utils.compute_view(points[["lon", "lat"]].values.tolist(), view_proportion=0.95)
        view.zoom = min(view.zoom, 15)

    # Plain string props must be quoted ('"pixels"'), or pydeck turns them into data expressions.
    dots = pdk.Layer("ScatterplotLayer", id="restaurants", data=points, get_position=["lon", "lat"],
                     get_fill_color="color", get_radius="radius", radius_units='"pixels"',
                     stroked=True, get_line_color=[255, 255, 255], line_width_min_pixels=1.5,
                     pickable=True, auto_highlight=True)
    labels = pdk.Layer("TextLayer", id="labels", data=points, get_position=["lon", "lat"],
                       get_text="label", get_size=12, get_color=[255, 255, 255],
                       font_weight=700, pickable=False)
    tooltip = {"html": "<b>{rank}. {name}</b><br/>⭐ {stars} ({reviews} reviews) · {price}",
               "style": {"fontSize": "13px"}}
    return pdk.Deck(layers=[dots, labels], initial_view_state=view, tooltip=tooltip, map_style=None)


def render_card(r, show_button=True):
    selected = r["business_id"] == selected_id
    with st.container(border=True, key="selected_card" if selected and show_button else None):
        left, right = st.columns([3, 1])
        maps = "https://www.google.com/maps/search/?api=1&query=" + quote_plus(
            f"{r['name']} {r['address']} {r['city']} {r['state']}")
        left.subheader(f"{r['rank']}. {r['name']}", anchor=False)
        left.write(f"⭐ {r['avg_stars']:.1f} ({r['n_reviews']:,} reviews) · {md_escape(r['price'])} · "
                   f"{short_categories(r['categories'])}")
        left.markdown(f"{r['address']}, {r['city']}, {r['state']} · [Open in Google Maps]({maps})")
        right.metric("Match score", f"{r['score'] * 100:.0f}")
        right.caption(" · ".join(f"{FACTORS[n]} {r['f_' + n]:.2f}" for n in used))
        if show_button:
            left.button("Selected on map" if selected else "Show on map", key=f"show_{r['business_id']}",
                        on_click=select_from_list, args=(r["business_id"],), disabled=selected,
                        icon=":material/location_on:", type="tertiary")


# Map in the middle, scrollable list of recommendations on the right.
map_col, list_col = st.columns(2, gap="medium")
with map_col:
    st.pydeck_chart(build_map(results, selected_id), height=600, key="map",
                    on_select=select_from_map, selection_mode="single-object")
    if selected_id is not None:
        render_card(results[results["business_id"] == selected_id].iloc[0], show_button=False)
        st.button("Show all results", on_click=select_from_list, args=(None,), type="tertiary",
                  icon=":material/zoom_out_map:")
    else:
        st.caption("Hover over a dot to see the restaurant, click it for details, "
                   "or use “Show on map” on a result.")

with list_col.container(height=650, border=False):
    for _, r in results.iterrows():
        render_card(r)
