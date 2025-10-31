import re, sys, time, pandas as pd, requests

CITY = sys.argv[3]
UA = f"geo-script (+{sys.argv[4]})"

ALIASES = {
    "чкаловск": "Чкаловская лестница",
    "чкаловской": "Чкаловская лестница",
    "пл. минина": "площадь Минина и Пожарского",
    "московск": "Московский вокзал",
    "ильинк": "улица Ильинская",
    "нижне-волж": "Нижне-Волжская набережная",
    "верхне-волж": "Верхне-Волжская набережная",
    "покровск": "улица Большая Покровская",
    "федоровск": "наб. Федоровского",
    "стрелк": "метро Стрелка",
    "канат": "Канатная дорога Нижний Новгород",
    "варварск": "улица Варварская",
    "сенная": "площадь Сенная",
    "добролюб": "улица Добролюбова",
    "ковалихин": "улица Ковалихинская",
    "звездинк": "улица Звездинка",
}

COORD = re.compile(r"(-?\d+(?:[.,]\d+)?)\s*[, ]\s*(-?\d+(?:[.,]\d+)?)")
SPAN = re.compile(
    r"(?:от|старт(?:ую)?(?:\s+от)?|я у|я возле|у|рядом с)\s+([^.,;:!?]+)", re.I
)


def find_address_hint(text: str) -> list[str]:
    t = text.strip()
    out = []
    m = COORD.search(t)
    if m:
        lat = m.group(1).replace(",", ".")
        lon = m.group(2).replace(",", ".")
        return [f"{lat},{lon}"]
    m = SPAN.search(t)
    if m:
        out.append(m.group(1).strip())
    # резерв: первые 6–10 слов после «на/от/у»
    for pat in ["на ", "от ", "у "]:
        i = t.lower().find(pat)
        if i >= 0:
            out.append(t[i + len(pat) :].split(".")[0][:80])
    # алиасы
    low = t.lower()
    for k, v in ALIASES.items():
        if k in low:
            out.append(v)
    # если ничего — весь текст как крайний случай
    if not out:
        out.append(t[:120])
    # нормализация
    out = [s.strip(" \"'«»") for s in out if s.strip()]
    # удаляем «Нижний Новгород» если уже есть
    return list(dict.fromkeys(out))


def geocode(q: str):
    if COORD.fullmatch(q):
        lat, lon = q.split(",")
        return float(lat), float(lon)
    if "нижн" not in q.lower():
        q = f"{q}, {CITY}"
    r = requests.get(
        "https://nominatim.openstreetmap.org/search",
        params={"q": q, "format": "jsonv2", "limit": 1},
        headers={"User-Agent": UA},
        timeout=15,
    )
    if r.ok and r.json():
        j = r.json()[0]
        return float(j["lat"]), float(j["lon"])
    return None


def main(inp, outp):
    df = pd.read_csv(inp)
    lats, lons = [], []
    for i, r in enumerate(df.itertuples(index=False), 1):
        if pd.notna(getattr(r, "start_lat", None)) and pd.notna(
            getattr(r, "start_lon", None)
        ):
            lats.append(r.start_lat)
            lons.append(r.start_lon)
            continue
        text = getattr(r, "start_raw", getattr(r, "text", ""))
        tried = 0
        got = None
        for hint in find_address_hint(str(text)):
            got = geocode(hint)
            tried += 1
            if got:
                break
        lats.append(None if not got else got[0])
        lons.append(None if not got else got[1])
        time.sleep(1.0)  
        if i % 20 == 0:
            print(f"[{i}/{len(df)}] geocoded; last tried {tried} variants", flush=True)
    df["start_lat"] = lats
    df["start_lon"] = lons
    df.to_csv(outp, index=False)


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2])
