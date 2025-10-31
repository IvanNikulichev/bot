import sys, math, json, argparse
import pandas as pd
import numpy as np

PARK_STAY_MIN = 30.0
DEFAULT_STAY_MIN = 3.0

NUM = [
    "distance_km",
    "inv_distance",
    "log_distance",
    "rank_by_distance",
    "hours",
    "poi_pop_local",
    "eta_walk_min",
    "stay_min",
    "semantic_cos",
    "token_overlap",
    "interest_match_count",
]
BIN = ["kind_match", "view_hint", "eta_fit"]
HB = ["hb_le1", "hb_1_2", "hb_2_3", "hb_gt3", "hb_unknown"]
KIND = [
    "kind_cafe",
    "kind_dessert",
    "kind_museum",
    "kind_viewpoint",
    "kind_street_art",
    "kind_historic",
    "kind_architecture",
    "kind_park",
]
COLS = NUM + BIN + HB + KIND


def haversine_km(lat1, lon1, lat2, lon2):
    R = 6371.0
    p1, p2 = np.radians(lat1), np.radians(lat2)
    dphi = p2 - p1
    dl = np.radians(lon2 - lon1)
    h = np.sin(dphi / 2.0) ** 2 + np.cos(p1) * np.cos(p2) * np.sin(dl / 2.0) ** 2
    return 2.0 * R * np.arcsin(np.sqrt(h))


def safe_num_series(df: pd.DataFrame, name: str, default: float = 0.0) -> pd.Series:
    """Вернёт числовую Series длины df, даже если столбца нет."""
    if name in df.columns:
        return pd.to_numeric(df[name], errors="coerce").fillna(default)
    return pd.Series(np.full(len(df), default, dtype=float), index=df.index)


def safe_text_series(df: pd.DataFrame, name: str) -> pd.Series:
    if name in df.columns:
        return df[name].astype(str).fillna("").str.strip()
    return pd.Series([""] * len(df), index=df.index)


def parse_wkt_point(s: str):
    if not isinstance(s, str):
        return np.nan, np.nan
    try:
        inner = s[s.find("(") + 1 : s.find(")")].replace(",", " ")
        lon, lat = [float(x) for x in inner.split()]
        return lat, lon
    except Exception:
        return np.nan, np.nan


def load_poi(poi_csv: str, cats_csv: str | None):
    poi = pd.read_csv(poi_csv)

    if "lat" not in poi.columns or "lon" not in poi.columns:
        if "coordinate" in poi.columns:
            latlon = poi["coordinate"].apply(parse_wkt_point)
            poi["lat"] = latlon.apply(lambda x: x[0])
            poi["lon"] = latlon.apply(lambda x: x[1])
    poi = poi.copy()
    poi["lat"] = pd.to_numeric(poi.get("lat", np.nan), errors="coerce")
    poi["lon"] = pd.to_numeric(poi.get("lon", np.nan), errors="coerce")
    poi = poi.dropna(subset=["lat", "lon"]).reset_index(drop=True)

    for c in ["title", "description", "tags", "kind", "address"]:
        poi[c] = safe_text_series(poi, c).str.lower()

    for c in ["popularity", "rating", "poi_pop_local"]:
        poi[c] = safe_num_series(poi, c, 0.0)

    if "id" not in poi.columns:
        poi["id"] = np.arange(1, len(poi) + 1, dtype=int)

    poi["__cats__"] = [[] for _ in range(len(poi))]
    if cats_csv:
        cats = pd.read_csv(cats_csv)
        key = "poi_id" if "poi_id" in cats.columns else "id"
        val = "categories" if "categories" in cats.columns else cats.columns[-1]
        mp = {}
        for r in cats.itertuples(index=False):
            rid = getattr(r, key)
            raw = getattr(r, val)
            if isinstance(raw, str):
                toks = [
                    t.strip().lower()
                    for t in raw.replace("|", ",").split(",")
                    if t.strip()
                ]
            else:
                toks = []
            mp[rid] = toks
        poi["__cats__"] = poi["id"].map(lambda x: mp.get(x, []))

    return poi


def load_queries(q_csv: str):
    q = pd.read_csv(q_csv)
    q["query_id"] = safe_text_series(q, "query_id")
    q["text"] = safe_text_series(q, "text").str.lower()
    q["hours"] = (
        pd.to_numeric(q.get("hours", np.nan), errors="coerce")
        .fillna(2.0)
        .clip(lower=0.25, upper=12.0)
    )

    # стартовые координаты
    q["start_lat"] = pd.to_numeric(q.get("start_lat", np.nan), errors="coerce")
    q["start_lon"] = pd.to_numeric(q.get("start_lon", np.nan), errors="coerce")

    # интересы: в колонке interests_set может быть строка "a;b;c" или "['a','b']"
    def parse_interests(s: str):
        if not isinstance(s, str) or not s.strip():
            return set()
        s = s.strip().strip("[]")
        s = s.replace(";", ",")
        toks = [t.strip(" '\"").lower() for t in s.split(",") if t.strip(" '\"")]
        return set(toks)

    q["interests"] = safe_text_series(q, "interests_set").apply(parse_interests)
    return q


def stay_minutes(kind: str) -> float:
    if isinstance(kind, str) and "park" in kind:
        return PARK_STAY_MIN
    return DEFAULT_STAY_MIN


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("queries_csv")
    ap.add_argument("poi_csv")
    ap.add_argument("out_parquet")
    ap.add_argument("top_k", type=int, nargs="?", default=30)
    ap.add_argument("--cats", default=None, help="CSV с категориями: poi_id,categories")
    args = ap.parse_args()

    q = load_queries(args.queries_csv)
    poi = load_poi(args.poi_csv, args.cats)

    rows = []
    for qi, qrow in q.iterrows():
        lat0, lon0 = qrow["start_lat"], qrow["start_lon"]
        if np.isnan(lat0) or np.isnan(lon0):
            # если нет старта — пропустим
            continue

        # расстояние до всех POI
        dist = haversine_km(lat0, lon0, poi["lat"].values, poi["lon"].values)
        order = np.argsort(dist)[: max(args.top_k, 1)]
        sub = poi.iloc[order].copy()
        sub_dist = dist[order]

        # текст запроса
        qtext = qrow["text"]
        qinter = qrow["interests"]

        def tok_overlap(r):
            txt = f"{r.get('title','')} {r.get('description','')} {r.get('tags','')}"
            cnt = 0
            for tok in qinter:
                if tok and tok in txt:
                    cnt += 1
            return cnt

        # kind flags
        def has_kind(r, name):
            k = r.get("kind", "")
            return 1 if isinstance(k, str) and name in k else 0

        # признаки
        f = pd.DataFrame(
            {
                "distance_km": sub_dist,
                "inv_distance": 1.0 / (1e-6 + sub_dist),
                "log_distance": np.log1p(sub_dist),
                "rank_by_distance": pd.Series(sub_dist).rank(method="dense").values,
                "hours": float(qrow["hours"]),
                "poi_pop_local": sub["poi_pop_local"].astype(float).values,
                "eta_walk_min": (
                    sub_dist / (1.2 * 60 / 1000.0)
                ), 
                "stay_min": sub["kind"].map(stay_minutes).astype(float).values,
                "semantic_cos": 0.0,
                "token_overlap": sub.apply(tok_overlap, axis=1).astype(float).values,
                "interest_match_count": sub.apply(
                    lambda r: sum(
                        1 for t in qinter if t and t in r.get("__cats__", [])
                    ),
                    axis=1,
                )
                .astype(float)
                .values,
                "kind_match": sub.apply(
                    lambda r: (
                        1 if any(t in (r.get("__cats__", [])) for t in qinter) else 0
                    ),
                    axis=1,
                )
                .astype(int)
                .values,
                "view_hint": sub["tags"]
                .str.contains("view|панорам", case=False, na=False)
                .astype(int)
                .values,
                "eta_fit": 1,
            }
        )

        hb = pd.DataFrame(
            {
                "hb_le1": int(qrow["hours"] <= 1.0),
                "hb_1_2": int(1.0 < qrow["hours"] <= 2.0),
                "hb_2_3": int(2.0 < qrow["hours"] <= 3.0),
                "hb_gt3": int(qrow["hours"] > 3.0),
                "hb_unknown": 0,
            },
            index=f.index,
        )
        f = pd.concat([f, hb], axis=1)

        for kname in [
            "cafe",
            "dessert",
            "museum",
            "viewpoint",
            "street_art",
            "historic",
            "architecture",
            "park",
        ]:
            f[f"kind_{kname}"] = (
                sub["kind"].str.contains(kname, case=False, na=False).astype(int).values
            )

        f["query_id"] = qrow["query_id"]
        f["poi_id"] = sub["id"].values
        f["lat"] = sub["lat"].values
        f["lon"] = sub["lon"].values
        f["title"] = sub["title"].values
        f["tags_str"] = sub["tags"].values

        for c in COLS:
            if c not in f.columns:
                f[c] = 0
        f = f[["query_id", "poi_id", "lat", "lon", "title", "tags_str"] + COLS]

        rows.append(f)

    out = (
        pd.concat(rows, ignore_index=True)
        if rows
        else pd.DataFrame(columns=["query_id", "poi_id"] + COLS)
    )
    out.to_parquet(args.out_parquet, index=False)
    print(f"features: {len(out)} rows -> {args.out_parquet}")


if __name__ == "__main__":
    main()
