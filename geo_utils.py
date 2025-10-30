# geo_utils.py
import re, unicodedata, pathlib
from functools import lru_cache

# Сокращения -> нормальная форма
ABBR = {
    r"\bул\.\b": "улица",
    r"\bпр-?т\b": "проспект",
    r"\bпросп\.\b": "проспект",
    r"\bпл\.\b": "площадь",
    r"\bпер\.\b": "переулок",
    r"\bнаб\.\b": "набережная",
    r"\bб-?р\b": "бульвар",
    r"\bш\.\b": "шоссе",
}

def _norm(s: str) -> str:
    s = s.lower()
    s = unicodedata.normalize("NFKC", s)
    for k, v in ABBR.items():
        s = re.sub(k, v, s)
    s = re.sub(r"[^\w\s\-]", " ", s)   # <- поправлено
    s = re.sub(r"\s+", " ", s).strip()
    return s

@lru_cache(maxsize=1)
def load_streets():
    p = pathlib.Path("data/streets_nn.txt")
    streets = []
    if p.exists():
        for line in p.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if line:
                streets.append(_norm(line))
    # индекс по первому слову
    idx = {}
    for s in streets:
        first = s.split()[0]
        idx.setdefault(first, []).append(s)
    return streets, idx

def find_street(text: str):
    """
    Возвращает (улица_норм, номер_дома|None) либо (None, None)
    """
    q = _norm(text or "")
    if not q:
        return None, None
    streets, idx = load_streets()
    if not streets:
        return None, None

    toks = q.split()
    cand = set()
    for i, t in enumerate(toks):
        bucket = idx.get(t, [])
        if not bucket:
            continue
        for L in (1, 2, 3):
            frag = " ".join(toks[i:i+L])
            if not frag:
                continue
            for s in bucket:
                if s.startswith(frag):
                    cand.add(s)

    # самое длинное совпадение, реально встречающееся в тексте
    street = None
    for s in sorted(cand, key=len, reverse=True):
        if s in q:
            street = s
            break
    if not street:
        return None, None

    # номер дома: 12, 12а, 12-А, 12к2 и т.п.
    m = re.search(re.escape(street) + r"\s*,?\s*([0-9]+(?:[А-Яа-яA-Za-z\-]?)(?:к\d+)?)", q)
    house = m.group(1) if m else None
    return street, house
