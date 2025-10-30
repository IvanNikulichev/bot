# build_features.py
import math, json, sys, pandas as pd, numpy as np
from sentence_transformers import SentenceTransformer

EMB = None
def ensure_emb():
    global EMB
    if EMB is None:
        EMB = SentenceTransformer("intfloat/multilingual-e5-small")  # CPU ОК

def hav_km(a,b):
    R=6371.0
    la1,lo1,la2,lo2 = map(math.radians,[a[0],a[1],b[0],b[1]])
    dlat=la2-la1; dlon=lo2-lo1
    s=math.sin(dlat/2)**2+math.cos(la1)*math.cos(la2)*math.sin(dlon/2)**2
    return 2*R*math.asin(min(1.0, math.sqrt(s)))

def hours_bin(h):
    if h is None: return "unknown"
    if h<=1: return "le1"
    if h<=2: return "1_2"
    if h<=3: return "2_3"
    return "gt3"

KIND_MAP = {
    "viewpoint":"view","street_art":"street_art","architecture":"architecture",
    "historic":"history","museum":"museum","cafe":"coffee","dessert":"dessert",
    "park":"park","attraction":"history","other":"other"
}
KIND_LIST = ["cafe","dessert","museum","viewpoint","street_art","historic","architecture","park"]

def stay_min(kind):
    return {"cafe":20,"dessert":20,"museum":45,"viewpoint":25,
            "street_art":25,"historic":30,"architecture":30,"park":25}.get(kind,30)

def poi_text(row):
    tags = row.get("tags")
    if isinstance(tags, str):
        try: tags = json.loads(tags)
        except: tags = {}
    parts = [str(row.get("name","")), str(row.get("kind",""))]
    if isinstance(tags, dict):
        for k in ("cuisine","amenity","shop","tourism","artwork_type","description","name:ru","alt_name"):
            if tags.get(k): parts.append(str(tags[k]))
    return " | ".join([p for p in parts if p])

def cosine(a:np.ndarray, b:np.ndarray) -> float:
    if a is None or b is None: return 0.0
    return float(np.dot(a, b))

def embed(texts:list[str]) -> np.ndarray:
    ensure_emb()
    return EMB.encode(texts, normalize_embeddings=True)

def parse_coord_point(geom:str):
    # "POINT (44.003277 56.331576)" → lat,lon
    if not isinstance(geom, str): return None
    m = geom.strip().replace(",", " ").split()
    # ожидаем POINT (lon lat)
    try:
        lon = float(m[-2].strip("POINT()"))
        lat = float(m[-1].strip("POINT()"))
        return lat, lon
    except:
        return None

def radius_by_hours(h):
    if not h: return 3000
    if h <= 1: return 1200
    if h <= 2: return 2200
    if h <= 3: return 3200
    if h <= 4: return 4000
    return 5000

def build_features(queries_csv:str, poi_csv:str, out_parquet:str, max_candidates:int=30):
    q = pd.read_csv(queries_csv)
    p = pd.read_csv(poi_csv) if poi_csv.endswith(".csv") else pd.read_excel(poi_csv)
    # нормализуем координаты POI (поддержка колонки "coordinate" как на скрине)
    if "lat" not in p.columns or "lon" not in p.columns:
        if "coordinate" in p.columns:
            coords = p["coordinate"].apply(parse_coord_point)
            p["lat"] = coords.apply(lambda x: x[0] if x else None)
            p["lon"] = coords.apply(lambda x: x[1] if x else None)
    p = p.dropna(subset=["lat","lon"]).copy()
    if "kind" not in p.columns:
        p["kind"] = "other"
    if "name" not in p.columns and "title" in p.columns:
        p["name"] = p["title"]

    # эмбеддинги запроса и POI
    q["q_emb"] = list(embed(q["text"].fillna("").tolist()))
    p["p_text"] = p.apply(poi_text, axis=1)
    p["p_emb"] = list(embed(p["p_text"].tolist()))

    rows=[]
    for qr in q.itertuples(index=False):
        try:
            lat, lon = float(qr.start_lat), float(qr.start_lon)
        except:
            # без координат кандидаты строить нельзя
            continue
        start=(lat,lon)
        rad = radius_by_hours(qr.hours)
        cand = p.copy()
        cand["distance_km"] = ((cand["lat"]-start[0])**2 + (cand["lon"]-start[1])**2)**0.5 * 111.0  # быстрая оценка
        cand = cand[cand["distance_km"] <= rad/1000 + 2.0]  # небольшой запас
        cand = cand.sort_values("distance_km").head(max_candidates).copy()
        if cand.empty: 
            continue

        # локальная доля kind
        kind_share = (cand["kind"].value_counts(normalize=True)).to_dict()

        # интересы
        try:
            interests = set(json.loads(qr.interests_set)) if isinstance(qr.interests_set, str) else set()
        except:
            interests = set()

        # признаки по каждому POI
        for rank, pr in enumerate(cand.itertuples(index=False), 1):
            kmap = KIND_MAP.get(pr.kind, "other")
            kind_oh = {f"kind_{k}": int(pr.kind == k) for k in KIND_LIST}
            hb = hours_bin(qr.hours)
            hb_oh = {f"hb_{k}": int(hb==k) for k in ["le1","1_2","2_3","gt3","unknown"]}

            invd = 1.0/(1.0+pr.distance_km)
            logd = math.log1p(pr.distance_km)
            sem = cosine(qr.q_emb, pr.p_emb)
            eta_walk = max(4.0, pr.distance_km/0.07)
            sm = float(stay_min(pr.kind))
            eta_fit = int(eta_walk + sm <= (qr.hours or 2.0)*60.0)

            rows.append({
                "query_id": qr.query_id,
                "poi_id": getattr(pr, "id", getattr(pr, "poi_id", rank)),
                "name": pr.name,
                "kind": pr.kind,
                "distance_km": pr.distance_km,
                "inv_distance": invd,
                "log_distance": logd,
                "rank_by_distance": rank,
                "hours": float(qr.hours) if qr.hours else 2.0,
                **hb_oh,
                "kind_match": int(kmap in interests),
                "interest_match_count": int(kmap in interests),
                "semantic_cos": sem,
                "token_overlap": 0.0,  # можно добавить позже
                "view_hint": int(("вид" in qr.text.lower() or "панорам" in qr.text.lower()) and pr.kind in ("viewpoint","attraction")),
                **kind_oh,
                "poi_pop_local": float(kind_share.get(pr.kind, 0.0)),
                "eta_walk_min": eta_walk,
                "stay_min": sm,
                "eta_fit": eta_fit,
            })

    df = pd.DataFrame(rows)
    df.to_parquet(out_parquet, index=False)
    print(f"features: {len(df)} rows -> {out_parquet}")

if __name__ == "__main__":
    # usage: python build_features.py queries_parsed.csv poi.csv features.parquet [max_candidates]
    mc = int(sys.argv[4]) if len(sys.argv)>4 else 30
    build_features(sys.argv[1], sys.argv[2], sys.argv[3], mc)
