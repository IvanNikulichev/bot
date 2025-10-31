# geo_utils.py
import os
import re
import requests

CONTACT_EMAIL = os.getenv("CONTACT_EMAIL", "you@example.com")

# координаты вида "56.32, 44.01"
COORD_RX = re.compile(r"([+-]?\d{1,2}(?:[.,]\d+))\s*[, ]\s*([+-]?\d{1,3}(?:[.,]\d+))")

# хватаем кусок после маркеров "я на/у, старт от, рядом с ..."
SPAN_RX  = re.compile(r"(?:^|\b)(?:от|старт(?:ую)?(?:\s+от)?|я у|я возле|я на|на|у|рядом с)\s+([^.,;:!?]+)", re.I)

# простая форма: тип + имя (в одном падеже)
SIMPLE_ADDR_RX = re.compile(
    r"\b(улиц[аыеиу]|ул\.?|площад[ьиью]|пл\.?|набережн(?:ая|ой|ую|е)|наб\.?)\s+([А-ЯЁA-Z][\w\-]+)",
    re.I
)

# приведение косвенных падежей к нормальной форме слова типа
def _normalize_type(s: str) -> str:
    t = s.lower().strip(" .")
    if t.startswith(("ул", "улиц")):       return "улица"
    if t.startswith(("пл", "площад")):     return "площадь"
    if t.startswith(("наб", "набережн")):  return "набережная"
    return s

# очень простая «лемматизация» последнего прилагательного: Львовской → Львовская, Советской → Советская
def _adj_feminative_nom(word: str) -> str:
    w = word
    if w.endswith("ской"):   return w[:-3] + "ская"
    if w.endswith("цкой"):   return w[:-3] + "цкая"
    if w.endswith("ой"):     return w[:-2] + "ая"
    if w.endswith("ей"):     return w[:-2] + "ея"
    return w

def _canon_toponym(text: str) -> str:
    """
    Делает «на улице Львовской» → «улица Львовская»,
    «набережной Федоровского» → «набережная Федоровского».
    """
    s = text.strip(' "\'«»')
    # сначала попробуем простой «тип + имя»
    m = SIMPLE_ADDR_RX.search(s)
    if m:
        typ = _normalize_type(m.group(1))
        name = _adj_feminative_nom(m.group(2))
        return f"{typ} {name}"

    # иначе если это просто кусок после "я на/у/от ..."
    m = SPAN_RX.search(s)
    if m:
        frag = m.group(1).strip()
        # попытка выровнять тип
        mm = SIMPLE_ADDR_RX.search(frag)
        if mm:
            typ = _normalize_type(mm.group(1))
            name = _adj_feminative_nom(mm.group(2))
            return f"{typ} {name}"
        # если «улице Львовской» без явного матча
        frag = re.sub(r"\b(улице|улицу|улицы|ул\.)\b", "улица", frag, flags=re.I)
        frag = re.sub(r"\b(площади|площадью|пл\.)\b", "площадь", frag, flags=re.I)
        frag = re.sub(r"\b(набережной|набережную|наб\.)\b", "набережная", frag, flags=re.I)
        # последнее слово как имя
        parts = frag.split()
        if parts and parts[0].lower() in {"улица", "площадь", "набережная"}:
            parts[-1] = _adj_feminative_nom(parts[-1])
            return " ".join(parts)
        return frag

    return s

def _iter_hints(text: str):
    """
    Генерирует варианты подсказок для геокодера от более «умных» к более общим.
    """
    # 1) нормализованный кусок после «я на/у/рядом с …»
    m = SPAN_RX.search(text)
    if m:
        yield _canon_toponym(m.group(0))

    # 2) простая форма «тип + имя»
    m = SIMPLE_ADDR_RX.search(text)
    if m:
        typ = _normalize_type(m.group(1))
        name = _adj_feminative_nom(m.group(2))
        yield f"{typ} {name}"

    # 3) сырая строка как есть
    tail = text.strip(' "\'«»')
    if tail:
        yield tail

def parse_start(text: str):
    """
    Возвращает (lat, lon, display) или (None, None, reason).
    """
    # 1) голые координаты
    m = COORD_RX.search(text)
    if m:
        lat = float(m.group(1).replace(",", "."))
        lon = float(m.group(2).replace(",", "."))
        return lat, lon, f"{lat:.5f},{lon:.5f}"

    ua = f"tg-bot/route ({CONTACT_EMAIL})"
    for hint in _iter_hints(text):
        q = _canon_toponym(hint)
        # добавим город, если не указан
        if "нижн" not in q.lower():
            q = f"{q}, Нижний Новгород, Россия"
        try:
            r = requests.get(
                "https://nominatim.openstreetmap.org/search",
                params={"q": q, "format": "jsonv2", "limit": 1},
                headers={"User-Agent": ua},
                timeout=10,
            )
            if r.ok:
                js = r.json()
                if js:
                    j = js[0]
                    return float(j["lat"]), float(j["lon"]), j.get("display_name", q)
        except Exception:
            pass

    return None, None, "Адрес не распознан"
