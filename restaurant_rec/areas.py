"""Group small towns into areas for the City dropdown.

Many towns have only a handful of open restaurants, which gives meaningless rankings. Within each
(state, metro), the smallest area with fewer than MIN_AREA_SIZE open restaurants is merged into the
geographically nearest other area (by restaurant centroid), and this repeats until every area is big
enough. A small town therefore joins its nearest bigger town, or several nearby small towns combine.
Each area is named after its biggest town.
"""

import pandas as pd

from .recommender import haversine_km

MIN_AREA_SIZE = 10  # open restaurants an area (or metro option) needs to be offered


def build_areas(biz: pd.DataFrame, min_size: int = MIN_AREA_SIZE) -> pd.DataFrame:
    """One row per town: state, metro, city, area (dropdown label), area_size (open restaurants).

    Only open restaurants count. Towns whose whole (state, metro) has fewer than `min_size`
    open restaurants are left out.
    """
    # A town can straddle two metros (e.g. Pottstown, PA); it belongs to the metro of most of its restaurants.
    towns = (biz[biz["is_open"] == 1].groupby(["state", "city"])
             .agg(n=("business_id", "size"), lat=("latitude", "mean"), lon=("longitude", "mean"),
                  metro=("metro", lambda m: m.mode().iloc[0]))
             .reset_index())
    out = []
    for (state, metro), group in towns.groupby(["state", "metro"]):
        # area name -> [member towns, n, lat, lon]; the name is the biggest member town
        areas = {r.city: [[r.city], r.n, r.lat, r.lon] for r in group.itertuples()}
        while len(areas) > 1:
            name, (members, n, lat, lon) = min(areas.items(), key=lambda kv: (kv[1][1], kv[0]))
            if n >= min_size:
                break
            nearest = min((other for other in areas if other != name),
                          key=lambda o: haversine_km(lat, lon, areas[o][2], areas[o][3]))
            o_members, o_n, o_lat, o_lon = areas.pop(nearest)
            del areas[name]
            total = n + o_n
            merged = [members + o_members, total,
                      (lat * n + o_lat * o_n) / total, (lon * n + o_lon * o_n) / total]
            biggest = nearest if (o_n, nearest) > (n, name) else name
            areas[biggest] = merged
        for name, (members, n, _, _) in areas.items():
            if n >= min_size:
                out += [(state, metro, city, name, n) for city in members]
    return pd.DataFrame(out, columns=["state", "metro", "city", "area", "area_size"])
