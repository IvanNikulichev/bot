# bot.py
import os, re, math, json, logging, asyncio
from functools import lru_cache
from typing import Tuple, List, Optional

import numpy as np
import pandas as pd
from aiogram import Bot, Dispatcher, types
from aiogram.filters import CommandStart
from aiogram.client.default import DefaultBotProperties
from aiogram.enums import ParseMode
from catboost import CatBoostRanker
from dotenv import load_dotenv

from geo_utils import find_street

logging.basicConfig(level=logging.INFO, format="%(levelname)s:%(name)s:%(message)s")
log = logging.getLogger("walkbot")

WALK_MPS = 3.6
WALK_KMPM = WALK_MPS * 60 / 1000.0
STAY_MIN_DEFAULT = 3.0
STAY_MIN_BY_KIND = {"park": 30.0}
TIME_TOL = 0.20

W_INTEREST_MATCH = 1.5
W_TOKEN_OVERLAP  = 0.5

def _norm(s: Optional[str]) -> str:
    return (s or "").strip().lower()

def haversine_km(a: Tuple[float,float], b: Tuple[float,float]) -> float:
    R = 6371.0
    lat1, lon1 = math.radians(a[0]), math.radians(a[1])
    lat2, lon2 = math.radians(b[0]), math.radians(b[1])
    dlat, dlon = lat2 - lat1, lon2 - lon1
    h = math.sin(dlat/2)**2 + math.cos(lat1)*math.cos(lat2)*math.sin(dlon/2)**2
    return 2 * R * math.asin(math.sqrt(h))

def eta_walk_min(km: float) -> float:
    return km / WALK_KMPM

def stay_minutes(kind: str) -> float:
    return STAY_MIN_BY_KIND.get(kind, STAY_MIN_DEFAULT)

def load_poi(path: str = "poi_csv.csv") -> pd.DataFrame:
    df = pd.read_csv(path)

    if "id" not in df.columns:
        df.insert(0, "id", np.arange(len(df), dtype=np.int64))

    for cn in ("lat", "lon"):
        if cn not in df.columns:
            df[cn] = np.nan
        df[cn] = pd.to_numeric(df[cn], errors="coerce")

    for cn in ("kind", "tags", "title", "description", "address"):
        if cn not in df.columns:
            df[cn] = ""
        df[cn] = df[cn].astype(str).fillna("").str.strip().str.lower()

    if "poi_pop_local" not in df.columns:
        df["poi_pop_local"] = 0.0
    df["poi_pop_local"] = pd.to_numeric(df["poi_pop_local"], errors="coerce").fillna(0.0)

    # только валидные координаты
    df = df[np.isfinite(df["lat"]) & np.isfinite(df["lon"])].copy()

    df["tags_list"] = df["tags"].apply(lambda s: [t.strip() for t in str(s).split(",") if t.strip()])
    df["kind_norm"] = df["kind"].str.replace(r"\s+", " ", regex=True).str.strip()
    return df

@lru_cache(maxsize=1)
def load_model() -> Optional[CatBoostRanker]:
    try:
        m = CatBoostRanker()
        m.load_model("poi_ranker.cbm")
        log.info("CatBoost модель загружена: poi_ranker.cbm")
        return m
    except Exception as e:
        log.warning("Не удалось загрузить модель: %s. Использую фолбэк-скоринг.", e)
        return None

@lru_cache(maxsize=1)
def load_featlist() -> Optional[List[str]]:
    try:
        with open("model_features.json", "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception as e:
        log.warning("model_features.json не найден/повреждён: %s. Продолжу без него.", e)
        return None

INTERESTS = {
    "coffee": ["кофе","кофейня","латте","эспрессо","капучино","каппучино","фильтр"],
    "dessert": ["десерт","пирожн","эклер","торт","выпечк","сладост"],
    "street_art": ["стрит-арт","мурал","граффити","графити","street art","murals"],
    "architecture": ["архитектур","особняк","модерн","усадьб","фасад","ампир","конструктивизм","готик"],
    "historic": ["историч","купеческ","старинн","церков","монастыр","крепост","кремл"],
    "viewpoint": ["вид","панорам","ракурс","фото","инст","обзорн","набережн"],
    "park": ["парк","сквер","сад","набережной парк"],
    "museum": ["музей","выставк","экспозиц"],
}

def extract_hours(text: str) -> Optional[float]:
    t = _norm(text)
    m = re.search(r"(\d+(?:[.,]\d+)?)\s*(?:час|ч)\b", t)
    if m: return float(m.group(1).replace(",", "."))
    m = re.search(r"(\d+)\s*мин", t)
    if m: return max(0.5, int(m.group(1))/60.0)
    if "полтора" in t: return 1.5
    if re.search(r"\bчас\b", t): return 1.0
    return None

def extract_interests(text: str) -> List[str]:
    t = _norm(text)
    return sorted({k for k, toks in INTERESTS.items() if any(tok in t for tok in toks)})

def extract_coords(text: str):
    t = _norm(text)
    m = re.search(r"([+-]?\d{1,2}\.\d+)[, ]+\s*([+-]?\d{1,3}\.\d+)", t)
    if m:
        lat = float(m.group(1)); lon = float(m.group(2))
        if 40.0 < lon < 60.0 and 50.0 < lat < 60.0:
            return (lat, lon), f"{lat:.6f},{lon:.6f}"
    street, house = find_street(t)
    if street:
        return None, f"{street}{(' '+house) if house else ''}"
    return None, None

def build_features_online(poi_df: pd.DataFrame,
                          start_ll: Tuple[float,float],
                          hours: float,
                          interests: List[str]) -> pd.DataFrame:
    interest_tokens = set()
    for k in interests:
        interest_tokens |= set(INTERESTS.get(k, []))

    def interest_overlap(row):
        text = " ".join([str(row.get("title","")).lower(),
                         str(row.get("description","")).lower(),
                         str(row.get("tags","")).lower()])
        return sum(1 for tok in interest_tokens if tok in text)

    dist_km = poi_df.apply(
        lambda r: haversine_km(start_ll, (float(r["lat"]), float(r["lon"]))), axis=1
    )

    df = pd.DataFrame({
        "distance_km": dist_km,
        "inv_distance": 1.0 / (1e-6 + dist_km),
        "log_distance": np.log1p(dist_km),
        "rank_by_distance": pd.Series(dist_km).rank(method="dense"),
        "hours": float(hours),
        "poi_pop_local": pd.to_numeric(poi_df.get("poi_pop_local", 0.0), errors="coerce").fillna(0.0),
        "eta_walk_min": eta_walk_min(dist_km),
        "stay_min": poi_df["kind"].map(lambda k: stay_minutes(str(k))),
        "semantic_cos": 0.0,
        "token_overlap": poi_df.apply(interest_overlap, axis=1),
        "interest_match_count": poi_df.apply(lambda r: sum(int(k in str(r.get("kind",""))) for k in interests), axis=1),
        "kind_match": poi_df.apply(lambda r: int(any(k in str(r.get("kind","")) for k in interests)), axis=1),
        "view_hint": poi_df["tags"].str.contains("view|панорам", case=False, na=False).astype(int),
        "eta_fit": 1
    })

    # фикс: часы — скаляр, размножаем в колонках через np.full
    n = len(poi_df)
    hb_le1 = int(hours <= 1.0)
    hb_1_2 = int((hours > 1.0) and (hours <= 2.0))
    hb_2_3 = int((hours > 2.0) and (hours <= 3.0))
    hb_gt3 = int(hours > 3.0)

    hb = pd.DataFrame({
        "hb_le1": np.full(n, hb_le1, dtype=int),
        "hb_1_2": np.full(n, hb_1_2, dtype=int),
        "hb_2_3": np.full(n, hb_2_3, dtype=int),
        "hb_gt3": np.full(n, hb_gt3, dtype=int),
        "hb_unknown": np.zeros(n, dtype=int)
    })
    df = pd.concat([df, hb], axis=1)

    for k in ["cafe","dessert","museum","viewpoint","street_art","historic","architecture","park"]:
        df[f"kind_{k}"] = poi_df["kind"].str.contains(k, case=False, na=False).astype(int)

    cols = load_featlist()
    if cols:
        df = df.reindex(columns=cols, fill_value=0)
    return df

def pick_route(poi_df: pd.DataFrame,
               start_ll: Tuple[float,float],
               hours: float,
               interests: List[str]):
    X = build_features_online(poi_df, start_ll, hours, interests)
    model = load_model()

    if model is not None:
        base = model.predict(X)
    else:
        base = (
            1.5 * X.get("inv_distance", 0).values +
            0.7 * X.get("interest_match_count", 0).values +
            0.3 * X.get("token_overlap", 0).values
        )

    boost = (W_INTEREST_MATCH * X.get("interest_match_count", 0).values +
             W_TOKEN_OVERLAP  * X.get("token_overlap", 0).values)
    score = np.asarray(base).reshape(-1) + np.asarray(boost).reshape(-1)

    cand = poi_df.copy()
    cand["score"] = score
    cand = cand.sort_values("score", ascending=False).reset_index(drop=True)

    time_budget = hours * 60.0
    used, used_ids, cur, spent = [], set(), start_ll, 0.0

    for _, row in cand.iterrows():
        rid = row.get("id")
        if rid in used_ids:
            continue
        dist = haversine_km(cur, (float(row["lat"]), float(row["lon"])))
        walk = eta_walk_min(dist)
        stay = stay_minutes(str(row.get("kind","")))
        add = walk + stay
        if spent + add <= time_budget * (1 + TIME_TOL):
            used.append(row)
            used_ids.add(rid)
            spent += add
            cur = (float(row["lat"]), float(row["lon"]))
        if spent >= time_budget * (1 - TIME_TOL):
            break

    return used, spent

def format_reply(start_str: Optional[str],
                 start_ll: Tuple[float,float],
                 items: List[pd.Series],
                 total_min: float) -> str:
    lines = [f"Старт: {start_str or f'{start_ll[0]:.5f},{start_ll[1]:.5f}'}"]
    prev = start_ll
    for i, r in enumerate(items, 1):
        dist = haversine_km(prev, (float(r["lat"]), float(r["lon"])))
        walk = eta_walk_min(dist)
        prev = (float(r["lat"]), float(r["lon"]))
        stay = stay_minutes(str(r.get("kind","")))
        why = []
        if r.get("kind"):
            why.append(str(r["kind"]))
        if r.get("tags"):
            why.append("теги: " + ", ".join([t for t in str(r["tags"]).split(",")[:3]]))
        title = str(r.get("title","Без названия"))
        lines.append(f"{i}) {title} — {walk:.0f} мин пешком • остановка {stay:.0f} мин" +
                     (f" • {'; '.join(why)}" if why else ""))
    lines.append(f"Итого по времени ≈ {total_min:.0f} мин")
    return "\n".join(lines)

# ----------------------------- BOT ------------------------------
load_dotenv()
TOKEN = os.getenv("TELEGRAM_BOT_TOKEN")
if not TOKEN:
    raise RuntimeError("Нет TELEGRAM_BOT_TOKEN в .env")

bot = Bot(token=TOKEN, default=DefaultBotProperties(parse_mode=ParseMode.HTML))
dp = Dispatcher()

@dp.message(CommandStart())
async def start_cmd(msg: types.Message):
    await msg.answer(
        "Напиши: откуда старт, сколько времени и интересы.\n"
        "Примеры: «2 часа, от пл. Минина, модерн и кофейни»; "
        "«56.328, 44.005; 1.5 часа; хочу стрит-арт и виды»."
    )

@dp.message()
async def handle(msg: types.Message):
    text = msg.text or ""
    hours = extract_hours(text) or 2.0
    interests = extract_interests(text)
    start_ll, start_str = extract_coords(text)

    poi = load_poi()
    if start_ll is None:
        start_ll = (56.3269, 44.0060)
        if not start_str:
            start_str = "Центр (по умолчанию)"

    picked, spent = pick_route(poi, start_ll, hours, interests)
    if not picked:
        await msg.answer("Не удалось подобрать точки. Уточни интересы или отправь координаты/адрес старта.")
        return

    await msg.answer(format_reply(start_str, start_ll, picked, spent))

async def main():
    logging.getLogger("aiogram").setLevel(logging.INFO)
    await dp.start_polling(bot)

if __name__ == "__main__":
    asyncio.run(main())
