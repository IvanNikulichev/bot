import re, requests

def build_aliases():
    return {
        "чкаловск": "Чкаловская лестница",
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

COORD_RX = re.compile(r'(-?\d+(?:[.,]\d+)?)\s*[, ]\s*(-?\d+(?:[.,]\d+)?)')
SPAN_RX  = re.compile(r'(?:от|старт(?:ую)?(?:\s+от)?|я у|я возле|у|рядом с)\s+([^.,;:!?]+)', re.I)

def _address_hints(text: str):
    t = text.strip(); hints=[]
    m = COORD_RX.search(t)
    if m:
        lat = m.group(1).replace(",", "."); lon = m.group(2).replace(",", ".")
        return [f"{lat},{lon}"]
    m = SPAN_RX.search(t)
    if m: hints.append(m.group(1).strip())
    for pat in [" на ", " от ", " у "]:
        i = t.lower().find(pat)
        if i >= 0: hints.append(t[i+len(pat):].split(".")[0][:80])
    low = t.lower()
    for k,v in build_aliases().items():
        if k in low: hints.append(v)
    if not hints: hints.append(t[:120])
    return list(dict.fromkeys([s.strip(' "\'«»') for s in hints if s.strip()]))

def parse_start(text: str, contact_email: str):
    m = COORD_RX.search(text)
    if m:
        lat=float(m.group(1).replace(",", ".")); lon=float(m.group(2).replace(",", "."))
        return lat, lon, f"{lat:.5f},{lon:.5f}"
    ua = f"tg-bot/route (+{contact_email})"
    for hint in _address_hints(text):
        q = hint if "нижн" in hint.lower() else f"{hint}, Нижний Новгород, Россия"
        try:
            r = requests.get("https://nominatim.openstreetmap.org/search",
                             params={"q":q,"format":"jsonv2","limit":1},
                             headers={"User-Agent": ua}, timeout=15)
            if r.ok and r.json():
                j = r.json()[0]
                return float(j["lat"]), float(j["lon"]), j.get("display_name", hint)
        except Exception:
            pass
    return None, None, "Адрес не распознан"
