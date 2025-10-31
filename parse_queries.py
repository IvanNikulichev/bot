import re, json, sys, pandas as pd
from difflib import get_close_matches

# словарь интересов и синонимов/опечаток
CANON = {
    "coffee": ["кофе", "кафе", "фильтр", "капуч", "латт", "раф", "кoфе", "коф"],
    "dessert": ["слад", "десерт", "кондит", "выпеч", "пекар", "лимонад"],
    "street_art": [
        "стрит",
        "street",
        "мурал",
        "графф",
        "графити",
        "граффити",
        "арт",
        "андерграунд",
        "неформал",
    ],
    "museum": ["музе", "выстав", "экспоз"],
    "view": ["вид", "панорам", "ракурс", "фото", "инста", "фотогенич"],
    "history": ["истор", "купечеств", "ярмарк", "памятн", "монумен"],
    "architecture": [
        "архит",
        "модерн",
        "здание",
        "усадеб",
        "церков",
        "храм",
        "собор",
        "деревян",
        "зодч",
    ],
    "park": ["парк", "сквер", "набереж", "прогулка", "тих", "нетурист"],
}
CANON_KEYS = list(CANON.keys())
VOCAB = {w: k for k, arr in CANON.items() for w in arr}

# грубый геопарсер: координаты или адрес-текст (адрес вы геокодите позже)
COORD_RX = re.compile(r"(-?\d+(?:[.,]\d+)?)\s*[, ]\s*(-?\d+(?:[.,]\d+)?)")


def parse_hours(text: str):
    m = re.search(r"(\d+(?:[.,]\d+)?)\s*(?:ч|час|часа|часов)\b", text.lower())
    if m:
        return float(m.group(1).replace(",", "."))
    # fallback: «полтора», «45 минут»
    if "полтора" in text.lower():
        return 1.5
    m = re.search(r"(\d+)\s*мин", text.lower())
    if m:
        return max(0.25, float(m.group(1)) / 60.0)
    return None


def parse_start(text: str):
    m = COORD_RX.search(text)
    if m:
        lat = float(m.group(1).replace(",", "."))
        lon = float(m.group(2).replace(",", "."))
        # если перепутаны местами
        if abs(lat) <= 90 and abs(lon) <= 180:
            return {"start_lat": lat, "start_lon": lon, "start_raw": f"{lat},{lon}"}
    return {"start_lat": None, "start_lon": None, "start_raw": text.strip()}


def normalize_interests(text: str):
    t = re.findall(r"[а-яa-z0-9\-]+", text.lower())
    found = set()
    for token in t:
        if token in VOCAB:
            found.add(VOCAB[token])
            continue
        # опечатки
        cand = get_close_matches(token, VOCAB.keys(), n=1, cutoff=0.86)
        if cand:
            found.add(VOCAB[cand[0]])
    # подсказки по составным словам
    if "панорам" in text.lower():
        found.add("view")
    if "набереж" in text.lower():
        found.add("park")
        found.add("view")
    return sorted(found)


def process_queries(csv_in: str, csv_out: str):
    df = pd.read_csv(csv_in)
    # ожидается столбец "query_id" и "text"; если нет — генерируем id
    if "query_id" not in df.columns:
        df["query_id"] = [f"Q{i:05d}" for i in range(1, len(df) + 1)]
    out_rows = []
    for r in df.itertuples(index=False):
        qid = getattr(r, "query_id")
        text = getattr(r, "text") if "text" in df.columns else " ".join(map(str, r))
        hours = (
            parse_hours(text) or getattr(r, "hours", None)
            if hasattr(r, "hours")
            else parse_hours(text)
        )
        start = parse_start(text)
        interests = normalize_interests(text)
        out_rows.append(
            {
                "query_id": qid,
                "text": text,
                "hours": hours,
                **start,
                "interests_set": json.dumps(interests, ensure_ascii=False),
            }
        )
    pd.DataFrame(out_rows).to_csv(csv_out, index=False)


if __name__ == "__main__":
    process_queries(sys.argv[1], sys.argv[2])
