"""Flatten Yelp's messy `attributes` field into plain columns.

Yelp stores most attribute values as *strings of Python literals*, e.g.
    "True", "None", "u'free'", "'full_bar'", "2",
    "{'garage': False, 'street': True, 'lot': None}"
so we run each through ast.literal_eval and then pick out the fields we use.
"""

import ast

# Yelp attribute name -> our column name (boolean attributes)
BOOL_ATTRS = {
    "RestaurantsTakeOut": "takeout",
    "RestaurantsDelivery": "delivery",
    "RestaurantsReservations": "reservations",
    "OutdoorSeating": "outdoor_seating",
    "GoodForKids": "good_for_kids",
    "RestaurantsGoodForGroups": "good_for_groups",
    "WheelchairAccessible": "wheelchair_accessible",
    "DogsAllowed": "dogs_allowed",
    "HasTV": "has_tv",
}

# Yelp attribute name -> our column name (categorical string attributes)
STR_ATTRS = {
    "WiFi": "wifi",            # free / no / paid
    "Alcohol": "alcohol",      # none / beer_and_wine / full_bar
    "NoiseLevel": "noise_level",  # quiet / average / loud / very_loud
    "RestaurantsAttire": "attire",
}

PARKING_KINDS = ["garage", "street", "validated", "lot", "valet"]
AMBIENCE_KINDS = ["romantic", "intimate", "classy", "hipster", "divey",
                  "touristy", "trendy", "upscale", "casual"]
MEAL_KINDS = ["breakfast", "brunch", "lunch", "dinner", "latenight", "dessert"]


def parse_value(raw):
    """Turn one raw attribute string into a Python value (bool/str/int/dict/None)."""
    if raw is None:
        return None
    if not isinstance(raw, str):
        return raw
    try:
        value = ast.literal_eval(raw)
    except (ValueError, SyntaxError):
        return raw
    if isinstance(value, str):
        value = value.strip().lower()
        if value in ("none", ""):
            return None
    return value


def _as_bool(value):
    return value if isinstance(value, bool) else None


def _sub_dict(value):
    return value if isinstance(value, dict) else {}


def flatten_attributes(attrs):
    """Return a flat dict of our columns for one business's `attributes` dict."""
    attrs = attrs or {}
    parsed = {k: parse_value(v) for k, v in attrs.items()}
    row = {}

    price = parsed.get("RestaurantsPriceRange2")
    row["price_range"] = int(price) if isinstance(price, (int, str)) and str(price).isdigit() else None

    for yelp_name, col in BOOL_ATTRS.items():
        row[col] = _as_bool(parsed.get(yelp_name))
    for yelp_name, col in STR_ATTRS.items():
        value = parsed.get(yelp_name)
        row[col] = value if isinstance(value, str) else None

    parking = _sub_dict(parsed.get("BusinessParking"))
    for kind in PARKING_KINDS:
        row[f"parking_{kind}"] = _as_bool(parking.get(kind))
    row["parking_any"] = any(parking.get(k) is True for k in PARKING_KINDS) if parking else None
    row["bike_parking"] = _as_bool(parsed.get("BikeParking"))

    ambience = _sub_dict(parsed.get("Ambience"))
    row["ambience"] = ", ".join(k for k in AMBIENCE_KINDS if ambience.get(k) is True)

    meals = _sub_dict(parsed.get("GoodForMeal"))
    row["good_for_meal"] = ", ".join(k for k in MEAL_KINDS if meals.get(k) is True)

    return row
