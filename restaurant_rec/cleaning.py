"""Cleaning rules for the Yelp business data (see CLAUDE.md for the audit behind them)."""

import re

import numpy as np
import pandas as pd

# ---------- non-restaurant venues ----------
# Businesses tagged "Restaurants" that are really hotels, casinos, museums, malls, etc.

VENUE_CATEGORIES = {
    "Hotels", "Hotels & Travel", "Resorts", "Casinos", "Bed & Breakfast", "Guest Houses", "Hostels",
    "Museums", "Art Museums", "Botanical Gardens", "Parks", "Zoos", "Aquariums",
    "Landmarks & Historical Buildings", "Religious Organizations", "Churches", "Performing Arts",
    "Stadiums & Arenas", "Cinema", "Amusement Parks", "Arcades", "Bowling", "Golf", "Country Clubs",
    "Ski Resorts", "Skating Rinks", "Trampoline Parks", "Recreation Centers", "Swimming Pools",
    "Fitness & Instruction", "Gyms", "Amateur Sports Teams", "Airports", "Hospitals", "Medical Centers",
    "Colleges & Universities", "Department Stores", "Shopping Centers", "Outlet Stores", "Gas Stations",
    "Apartments", "Real Estate", "Convention Centers",
}
VENUE_NAME = re.compile(r"hotel|resort|casino|museum|auditorium|historic site|marriott|hilton|hyatt"
                        r"|sheraton|ritz|westin|suites|mall\b|airport|hospital", re.I)
FOOD_NAME = re.compile(r"restaurant|grill|caf[eé]|diner|kitchen|bistro|steakhouse|pizz|tavern|eatery|bbq"
                       r"|sushi|taqueria|bakery|\bdeli\b|\bbar\b|\bpub\b|saloon|brasserie|trattoria|cantina",
                       re.I)
FOOD_ROOTS = {"Restaurants", "Food", "Bars", "Nightlife"}


def strong_food_categories(all_businesses: pd.DataFrame) -> set[str]:
    """Categories where >= 90% of ALL Yelp businesses carrying them are food/drink places."""
    cats = all_businesses["categories"].fillna("").str.split(", ")
    foodish = cats.apply(lambda c: bool(FOOD_ROOTS & set(c)))
    share = pd.DataFrame({"cat": cats, "foodish": foodish}).explode("cat").groupby("cat")["foodish"].agg(["mean", "size"])
    return set(share[(share["mean"] >= 0.9) & (share["size"] >= 20)].index) - FOOD_ROOTS


def is_non_restaurant_venue(biz: pd.DataFrame, strong: set[str]) -> pd.Series:
    """True for venues where food is incidental, e.g. 'Grounds For Sculpture' or 'Harrah's Reno'.

    A business with a venue category is dropped if it has < 2 strong food categories,
    or >= 3 venue categories, or a venue-like name, unless its name says it is a restaurant.
    """
    cats = biz["categories"].str.split(", ").apply(set)
    venue = cats.apply(lambda c: VENUE_CATEGORIES & c)
    n_venue = venue.apply(lambda v: len(v - {"Hotels & Travel"}))
    n_strong = cats.apply(lambda c: len(strong & c))
    venue_name = biz["name"].str.contains(VENUE_NAME)
    food_name = biz["name"].str.contains(FOOD_NAME) & ~venue_name
    return (venue.str.len() > 0) & ~food_name & ((n_strong < 2) | (n_venue >= 3) | venue_name)


# ---------- metro areas ----------
# The dataset covers 12 metro areas; every restaurant is within 60 km of one of these hubs.

METROS = {  # label: (lat, lon, states that belong to it)
    "Boise, ID": (43.6133, -116.2198, {"ID"}),
    "Edmonton, AB": (53.5402, -113.4999, {"AB"}),
    "Indianapolis, IN": (39.7893, -86.1452, {"IN"}),
    "Nashville, TN": (36.1512, -86.7788, {"TN"}),
    "New Orleans, LA": (29.9536, -90.0719, {"LA"}),
    "Philadelphia, PA": (39.9545, -75.1604, {"PA", "NJ", "DE"}),
    "Reno, NV": (39.5142, -119.8056, {"NV", "CA"}),
    "Santa Barbara, CA": (34.4216, -119.6996, {"CA"}),
    "St. Louis, MO": (38.6247, -90.2811, {"MO", "IL"}),
    "Tampa, FL": (27.9864, -82.4834, {"FL"}),
    "Tucson, AZ": (32.2351, -110.9459, {"AZ"}),
    "Wilmington, DE": (39.7490, -75.5547, {"DE", "PA", "NJ"}),
}
MAX_METRO_KM = 80


def assign_metro(biz: pd.DataFrame) -> pd.DataFrame:
    """Add `metro`; drop rows far from every hub; fix state typos (e.g. 'XMS', 'HI') from location."""
    from .recommender import haversine_km
    labels = list(METROS)
    dist = np.column_stack([haversine_km(lat, lon, biz["latitude"].to_numpy(), biz["longitude"].to_numpy())
                            for lat, lon, _ in METROS.values()])
    biz = biz.assign(metro=np.array(labels)[dist.argmin(axis=1)])
    biz = biz[dist.min(axis=1) <= MAX_METRO_KM].copy()
    wrong_state = [s not in METROS[m][2] for s, m in zip(biz["state"], biz["metro"])]
    biz.loc[wrong_state, "state"] = biz.loc[wrong_state, "metro"].str[-2:]
    return biz


# ---------- names and cities ----------

CITY_ALIASES = {  # applied to normalized keys
    "phila": "philadelphia", "philly": "philadelphia", "philadephia": "philadelphia",
    "southwest philadelphia": "philadelphia",
    "nashville-davidson metropolitan government": "nashville",
    "st louis downtown": "st louis", "st louis county": "st louis",
}


def _city_key(city: str) -> str:
    c = city.strip().strip(",").strip().lower()
    c = re.sub(r",\s*[a-z]{2}$", "", c)          # "tampa,fl" -> "tampa"
    c = re.sub(r"\s*\(.*\)$", "", c)              # "largo (walsingham)" -> "largo"
    c = c.split("/")[0]                           # "clearwater/ countryside" -> "clearwater"
    c = re.sub(r"[.']", " ", c)                   # "st.louis", "land o'lakes"
    c = re.sub(r"\bsaint\b", "st", c)
    c = re.sub(r"^mt\b", "mount", c)
    c = re.sub(r"\s+(twp|township)$", "", c)
    c = re.sub(r"\s+", " ", c).strip()
    return CITY_ALIASES.get(c, c)


def _nice_case(city: str) -> str:
    return city.title() if city.isupper() or city.islower() else city


def clean_cities(cities: pd.Series) -> pd.Series:
    """Map spelling variants ('PHILA', 'Philadelphia ', 'Land o lakes') to one display name."""
    keys = cities.fillna("").map(_city_key)
    stripped = cities.fillna("").str.strip().str.rstrip(",").str.strip()
    # display name = most common properly-cased spelling within each key
    tidy = stripped.where(~(stripped.str.isupper() | stripped.str.islower()), None)
    best = (pd.DataFrame({"key": keys, "name": tidy.fillna(stripped.map(_nice_case))})
            .groupby("key")["name"].agg(lambda s: s.value_counts().index[0]))
    best["st louis"] = "St. Louis"
    best["philadelphia"] = "Philadelphia"
    return keys.map(best)


def clean_names(names: pd.Series) -> pd.Series:
    return names.str.replace(r"\s+", " ", regex=True).str.strip()


# ---------- hours ----------

DAYS = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]


def clean_hours(biz: pd.DataFrame) -> pd.DataFrame:
    """Yelp writes "0:0-0:0" both for "open 24h" and for "closed that day".

    If a restaurant has real hours on other days, "0:0-0:0" means closed (8k restaurants, mostly
    Mondays). Only when every listed day is "0:0-0:0" do we keep it as open 24 hours.
    """
    cols = [f"hours_{d}" for d in DAYS]
    hours = biz[cols]
    zero = hours.eq("0:0-0:0")
    has_real = (hours.notna() & ~zero).any(axis=1)
    biz = biz.copy()
    biz[cols] = hours.mask(zero & has_real.to_numpy()[:, None])
    return biz


# ---------- duplicates ----------

def dedupe(biz: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Same name + address = same restaurant (often an old closed listing and a new one).

    Keeps the open listing with the most reviews; returns (deduped, id_map) where id_map maps
    every dropped business_id to the kept one so its reviews can be merged.
    """
    key = biz["name"].str.lower() + "|" + biz["address"].fillna("").str.lower().str.strip()
    has_addr = biz["address"].fillna("").str.strip() != ""
    order = biz.assign(_key=key).sort_values(["is_open", "review_count"], ascending=False)
    keeper = order[has_addr.loc[order.index]].groupby("_key")["business_id"].first()
    mapped = key.map(keeper).where(has_addr)
    id_map = pd.DataFrame({"old_id": biz["business_id"], "new_id": mapped})
    id_map = id_map[id_map["new_id"].notna() & (id_map["old_id"] != id_map["new_id"])]
    return biz[~biz["business_id"].isin(id_map["old_id"])], id_map
