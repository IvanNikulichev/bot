# build_features.py
# usage: python build_features.py queries_geo.csv poi_csv.csv features.parquet [max_candidates]
# speed: 3.6 m/s (12.96 km/h). В парках stay=30 мин, иначе 3 мин.

import sys, json, math, re
import numpy as np
import pandas as pd

PTN_WKT = re.compile(r"POINT\s*\(\s*([\-0-9.]+)\s+([\-0-9.]+)\s*\)", re.I)
WALK_KMH = 12.96
MIN_PER_KM = 60.0 / WALK_KMH  # ≈ 4.63 мин/км
PARK_STAY_MIN = 30
DEFAULT_STAY_MIN = 3

# ---- helpers ----------------------------------------------------------------

def parse_point_wkt(s: str):
    if not isinstance(s, str):
        return None
    m = PTN_WKT.search(s)
    if not m:
        return None
    lon = float(m.group(1)); lat = float(m.group(2))
    return lat, lon

def ensure_latlon(df: pd.DataFrame) -> pd.DataFrame:
    if {"lat","lon"}.issubset(df.columns):
        pass
    elif {"latitude","longitude"}.issubset(df.columns):
        df = df.rename(columns={"latitude":"lat","longitude":"lon"})
    elif {"y","x"}.issubset(df.columns):
        df = df.rename(columns={"y":"lat","x":"lon"})
    else:
        for col in ("coordinate","geom","geometry","point"):
            if col in df.columns:
                pair = df[col].astype(str).apply(parse_point_wkt)
                df["lat"] = pair.apply(lambda t: t[0] if t else np.nan)
                df["lon"] = pair.apply(lambda t: t[1] if t else np.nan)
                break
    if not {"lat","lon"}.issubset(df.columns):
        raise KeyError("Не найдены lat/lon в POI. Ожидались lat/lon или WKT POINT(...) в 'coordinate'.")
    df["lat"] = pd.to_numeric(df["lat"], errors="coerce")
    df["lon"] = pd.to_numeric(df["lon"], errors="coerce")
    return df.dropna(subset=["lat","lon"]).copy()

def haversine_km(lat0, lon0, lat1, lon1):
    # принимает float или вектор numpy
    R = 6371.0088
    lat0 = np.radians(lat0); lon0 = np.radians(lon0)
    lat1 = np.radians(lat1); lon1 = np.radians(lon1)
    dlat = lat1 - lat0
    dlon = lon1 - lon0
    a = np.sin(dlat/2.0)**2 + np.cos(lat0)*np.cos(lat1)*np.sin(dlon/2.0)**2
    return 2*R*np.arcsin(np.sqrt(a))

def norm_tags(x):
    if pd.isna(x): return []
    if isinstance(x, (list, tuple)): return [str(t).strip().lower() for t in x if str(t).strip()]
    s = str(x).strip()
    try:
        v = json.loads(s)
        if isinstance(v, (list, tuple)):
            return [str(t).strip().lower() for t in v if str(t).strip()]
    except Exception:
        pass
    # comma/; separated
    parts = re.split(r"[,;|/]+", s.lower())
    return [p.strip() for p in parts if p.strip()]

def parse_interests(s):
    if pd.isna(s) or s=="":
        return []
    if isinstance(s, (list, tuple)):
        return [str(t).strip().lower() for t in s]
    s = str(s)
    try:
        v = json.loads(s)
        if isinstance(v, (list, tuple)):
            return [str(t).strip().lower() for t in v]
    except Exception:
        pass
    return [t.strip().lower() for t in s.split(",") if t.strip()]

def radius_by_hours(h):
    # консервативный радиус, чтобы успеть и походить, и постоять
    # ориентир: ~2.5 км в час хода в одну сторону максимум
    h = float(h) if pd.notna(h) else 1.0
    return max(0.8, min(6.0, 2.5*h))

# ---- feature builders -------------------------------------------------------

KIND_LIST = ["cafe","dessert","museum","viewpoint","street_art","historic","architecture","park"]

def is_park(tags, kind):
    if "park" in tags: return True
    if isinstance(kind, str) and "park" in kind.lower(): return True
    return False

def build_rows_for_query(q, p_all, max_candidates):
    qid = q["query_id"]
    lat0 = float(q["start_lat"]); lon0 = float(q["start_lon"])
    hours = float(q["hours"]) if pd.notna(q["hours"]) else 1.0
    interests = parse_interests(q.get("interests_set",""))

    # расстояния и фильтр радиуса
    dist = haversine_km(lat0, lon0, p_all["lat"].values, p_all["lon"].values)
    p = p_all.copy()
    p["distance_km"] = dist
    p = p.sort_values("distance_km", ascending=True)
    p = p[p["distance_km"] <= radius_by_hours(hours)]
    p = p.head(max_candidates).copy()

    if p.empty:
        return []

    # ранги
    p["rank_by_distance"] = np.arange(1, len(p)+1)

    # stay_min
    p["stay_min"] = np.where(
        p.apply(lambda r: is_park(r["tags_list"], r.get("kind","")), axis=1),
        PARK_STAY_MIN, DEFAULT_STAY_MIN
    )

    # скорость ходьбы -> ETA туда-до POI (не туда-обратно, это фича для ранжирования)
    p["eta_walk_min"] = (p["distance_km"] * MIN_PER_KM).astype(float)

    # match по интересам
    def interest_match(row):
        tags = set(row["tags_list"])
        k = str(row.get("kind","")).lower()
        if k: tags.add(k)
        inter = set(interests)
        return len(tags & inter)

    p["interest_match_count"] = p.apply(interest_match, axis=1)
    p["kind_match"] = (p["interest_match_count"] > 0).astype(int)

    # простая эвристика "видовая точка"
    p["view_hint"] = p.apply(
        lambda r: int(("view" in r["tags_list"]) or ("вид" in r["name"].lower()) or ("viewpoint" in str(r.get("kind","")).lower())),
        axis=1
    )

    # токенный оверлап между текстом запроса и именем
    text_tokens = set(re.findall(r"[a-zа-яё0-9]+", str(q.get("text","")).lower()))
    def tok_ov(name):
        nt = set(re.findall(r"[a-zа-яё0-9]+", str(name).lower()))
        if not nt or not text_tokens: return 0
        return len(nt & text_tokens)
    p["token_overlap"] = p["name"].apply(tok_ov)

    # семантика-плейсхолдер: 0
    p["semantic_cos"] = 0.0

    # прочее
    p["hours"] = hours
    p["inv_distance"] = 1.0 / (p["distance_km"] + 1e-6)
    p["log_distance"] = np.log1p(p["distance_km"])
    p["poi_pop_local"] = 1.0 / (p["rank_by_distance"] + 5)  # простая убывающая

    # ETA укладывается в доступное время? (ходьба до POI + стоянка, без обратного пути)
    p["eta_fit"] = ((p["eta_walk_min"] + p["stay_min"]) <= hours*60*0.9).astype(int)

    # бины по часам
    hb = {
        "hb_le1": float(hours <= 1),
        "hb_1_2": float(1 < hours <= 2),
        "hb_2_3": float(2 < hours <= 3),
        "hb_gt3": float(hours > 3),
        "hb_unknown": float(pd.isna(hours)),
    }
    for k,v in hb.items():
        p[k] = v

    # one-hot по kind
    kstr = p.get("kind","").astype(str).str.lower()
    for k in KIND_LIST:
        p[f"kind_{k}"] = (kstr.str.contains(k)).astype(int)

    # групп id
    p["query_id"] = qid

    # нужные колонки
    cols_keep = [
        "query_id","poi_id","name","kind","lat","lon","tags",
        "distance_km","inv_distance","log_distance","rank_by_distance","hours",
        "poi_pop_local","eta_walk_min","stay_min","semantic_cos","token_overlap","interest_match_count",
        "kind_match","view_hint","eta_fit",
        "hb_le1","hb_1_2","hb_2_3","hb_gt3","hb_unknown",
        "kind_cafe","kind_dessert","kind_museum","kind_viewpoint","kind_street_art","kind_historic","kind_architecture","kind_park"
    ]
    for c in cols_keep:
        if c not in p.columns:
            p[c] = 0
    return [p[cols_keep]]

# ---- main -------------------------------------------------------------------

def main(q_path, poi_path, out_path, max_candidates):
    q = pd.read_csv(q_path)
    p = pd.read_csv(poi_path)

    # нормализация POI
    p = ensure_latlon(p)
    if "poi_id" not in p.columns:
        # стабильный id
        p["poi_id"] = np.arange(1, len(p)+1)
    if "name" not in p.columns:
        p["name"] = p.get("title","").astype(str)
    if "kind" not in p.columns:
        # вытаскиваем из tags по ключевым словам
        p["kind"] = ""
    if "tags" not in p.columns:
        p["tags"] = ""

    p["tags_list"] = p["tags"].apply(norm_tags)

    # нормализация queries
    need = ["query_id","text","hours","start_lat","start_lon","interests_set"]
    for c in need:
        if c not in q.columns:
            q[c] = np.nan
    q = q.dropna(subset=["start_lat","start_lon"]).copy()

    # строим строки
    parts = []
    for _, row in q.iterrows():
        try:
            parts.extend(build_rows_for_query(row, p, max_candidates))
        except Exception as e:
            # пропускаем битые строки, но не падаем
            continue

    if not parts:
        pd.DataFrame().to_parquet(out_path, index=False)
        print("features: 0 rows ->", out_path)
        return

    feats = pd.concat(parts, ignore_index=True)

    # финальное приведение типов
    num_cols = ["distance_km","inv_distance","log_distance","rank_by_distance","hours",
                "poi_pop_local","eta_walk_min","stay_min","semantic_cos","token_overlap","interest_match_count"]
    feats[num_cols] = feats[num_cols].apply(pd.to_numeric, errors="coerce").fillna(0)

    feats.to_parquet(out_path, index=False)
    print(f"features: {len(feats)} rows -> {out_path}")

if __name__ == "__main__":
    if len(sys.argv) < 4:
        print("usage: python build_features.py queries_geo.csv poi_csv.csv features.parquet [max_candidates]")
        sys.exit(1)
    q_path = sys.argv[1]
    poi_path = sys.argv[2]
    out_path = sys.argv[3]
    mc = int(sys.argv[4]) if len(sys.argv) > 4 else 30
    main(q_path, poi_path, out_path, mc)
