import os, re, math, json, asyncio, logging, datetime, requests, html
from urllib.parse import quote_plus
from aiogram import Bot, Dispatcher, F
from aiogram.types import Message
from aiogram.filters import CommandStart
from dotenv import load_dotenv
from geo_utils import parse_start
from aiogram import Dispatcher
dp = Dispatcher()  
def get_dispatcher():
    return dp


load_dotenv()


TOKEN = os.getenv("TELEGRAM_BOT_TOKEN")
CONTACT_EMAIL = os.getenv("CONTACT_EMAIL", "you@example.com")
POI_CSV = os.getenv("POI_CSV", "poi_csv.csv")
MODEL_PATH = os.getenv("MODEL_PATH", "poi_ranker.cbm")
MODEL_FEATURES = os.getenv("MODEL_FEATURES", "model_features.json")

USE_MODEL = True

# скорость и параметры построения
WALK_SPEED_MPS = 1.2
CAND_POOL = 60
STOP_FIXED_S = 3 * 60
STOP_PARK_S = 30 * 60
TIME_TOL = 0.20
CLOSE_LOOP_KM = 1.0

LOOP_WEIGHT_BASE = 1.0
LOOP_BACK_DIVISOR = 800.0

TG_HARD_LIMIT = 4096
TG_SAFE_LIMIT = 4000
YANDEX_MAX_POINTS = 20

logging.basicConfig(level=logging.INFO)
bot = Bot(TOKEN)
dp = Dispatcher()

# ------------ Словари интересов/алиасы ------------
ALIASES = {
    "чкаловск": "Чкаловская лестница",
    "пл. минина": "площадь Минина и Пожарского",
    "московск": "Московский вокзал",
    "ильинк": "улица Ильинская",
    "нижне-волж": "Нижне-Волжская набережная",
    "верхне-волж": "Верхне-Волжская набережная",
    "покровск": "улица Большая Покровская",
    "федоровск": "наб. Федоровского",
    "стрелк": "метро Стрелака",
    "канат": "Канатная дорога Нижний Новгород",
    "варварск": "улица Варварская",
    "сенная": "площадь Сенная",
    "добролюб": "улица Добролюбова",
    "ковалихин": "улица Ковалихинская",
    "звездинк": "улица Звездинка",
}

INTERESTS = {
    "coffee": {
        "кофе",
        "кофей",
        "coffee",
        "эспрессо",
        "espresso",
        "капучино",
        "латте",
        "раф",
        "флэт уайт",
        "aeropress",
        "аэропресс",
        "v60",
        "спешалти",
        "specialty",
    },
    "tea": {"чай", "tea", "матча", "маття", "пуэр", "улун", "сенча", "гойчай"},
    "dessert": {
        "десерт",
        "пирожн",
        "эклер",
        "макарон",
        "чизкейк",
        "маффин",
        "кейк",
        "штрудель",
        "тирамису",
        "выпечк",
        "торт",
        "морожен",
        "ice cream",
        "gelato",
    },
    "bakery": {
        "пекарн",
        "булочн",
        "круассан",
        "круасан",
        "хлеб",
        "багет",
        "фокачч",
        "bakery",
        "boulangerie",
    },
    "brunch": {
        "бранч",
        "завтрак",
        "скрэмбл",
        "омлет",
        "яичниц",
        "авокадо тост",
        "панкейк",
        "pancake",
        "французский тост",
    },
    "pizza": {"пицц", "pizzeria", "пиццер"},
    "pasta": {"паста", "равиоли", "тальятелле", "спагетт", "лазан"},
    "steak": {"стейк", "гриль", "bbq", "барбекю", "брискет"},
    "burger": {"бургер", "burger", "котлета", "фри"},
    "shawarma": {"шаурм", "шаверм", "донер", "кебаб"},
    "georgian": {"грузин", "хинкал", "хачапур", "лобио", "сацив"},
    "italian": {"итал", "траттор", "остерия", "risotto", "рикотт"},
    "japanese": {"япон", "суши", "ролл", "рамен", "удон", "донбури", "якитор"},
    "chinese": {"китай", "димсам", "лапша по", "бао", "гунбао", "сычуан"},
    "korean": {"корее", "кимчи", "рамён", "коги", "ттокпокки", "самгёпсаль"},
    "thai": {"тайск", "том ям", "том кха", "пад тай", "сом там"},
    "vietnamese": {"вьетнам", "фо", "бон бо", "бан ми"},
    "indian": {"индий", "карри", "наан", "масала", "тандури"},
    "uzbek": {"узбек", "плов", "самса", "лагман", "манты"},
    "caucasian": {"кавказ", "шашлык", "долма", "чурчхел"},
    "mexican": {"мексик", "тако", "буррито", "начос", "кесадиль"},
    "turkish": {"турец", "пиде", "люля", "донер", "бахлава", "баклава"},
    "lebanese": {"ливан", "хумус", "фалафель", "табуле", "шаварма"},
    "vegan": {"веган", "vegan", "plant-based", "без мяса"},
    "vegetarian": {"вегетари", "vegetarian"},
    "gluten_free": {"без глютен", "gluten free"},
    "halal": {"халал", "halal"},
    "kosher": {"кошер", "kosher"},
    "bar": {
        "бар",
        "pub",
        "паб",
        "винн",
        "вино",
        "винный",
        "сидр",
        "крафт",
        "пивн",
        "brewery",
        "пивовар",
        "коктейл",
        "cocktail",
        "mixology",
        "speakeasy",
        "роофтап",
        "rooftop",
    },
    "hookah": {"кальян", "hookah", "shisha"},
    "rooftop": {"rooftop", "крыша", "панорамный бар", "видовой бар"},
    "street_art": {"стрит-арт", "street art", "мурал", "муралы", "граффити"},
    "gallery": {
        "галере",
        "арт-цент",
        "выставк",
        "арт-пространств",
        "art space",
        "центр соврем",
    },
    "museum": {"музей", "экспозици", "ретро", "диорама"},
    "theatre": {"театр", "драм", "опер", "балет", "сцена"},
    "cinema": {"кино", "cinema", "movie", "киноцентр"},
    "music": {"концерт", "клуб", "джаз", "рок", "live", "филармони"},
    "monument": {"памятник", "монумент", "скульптур", "стела", "бюст"},
    "library": {"библиотек", "читальн", "mediatheque"},
    "bookstore": {"книжн", "bookshop", "bookstore"},
    "history": {
        "историческ",
        "купеческ",
        "кремль",
        "арсенал",
        "фортификац",
        "краевед",
        "музей истории",
    },
    "architecture": {
        "архитектур",
        "фасад",
        "особняк",
        "доходн",
        "усадебн",
        "памятник архитектуры",
        "ордер",
        "пилястр",
    },
    "baroque": {"барокк"},
    "classicism": {"классицизм", "ампир", "empire"},
    "art_nouveau": {"модерн", "арт-нуво", "jugendstil", "сецессион"},
    "constructivism": {"конструктивизм", "авангар", "советск модерн", "баухауз"},
    "brutalism": {"брутализм"},
    "soviet_modernism": {"советск модерн", "модернизм 60", "нииб"},
    "wooden": {"деревянн", "резн", "наличник", "деревянное зодчество"},
    "manor": {"усадьб", "помещич", "дворянск усадьб"},
    "palace": {"дворец", "палас"},
    "fortress": {"крепост", "форт", "валы"},
    "kremlin": {"кремль"},
    "cemetery": {"кладбищ", "некропол"},
    "church": {"церковь", "собор", "храм", "колокольн", "монастыр", "часовн", "лавра"},
    "mosque": {"мечет"},
    "synagogue": {"синагог"},
    "view": {
        "вид",
        "виды",
        "панорама",
        "обзорная",
        "обзорн",
        "viewpoint",
        "смотор",
        "белведер",
        "колесо обозр",
    },
    "embankment": {"набережн", "бережн", "boulevard", "bulvar", "бульвар"},
    "river_volga": {"волга", "стрелка", "ярмарочн площад"},
    "river_oka": {"ока"},
    "park": {"парк", "сквер", "сад", "ботаническ", "дендрар", "аллея"},
    "nature": {
        "лес",
        "роща",
        "овраг",
        "ущелье",
        "утес",
        "берег",
        "пляж",
        "остров",
        "луга",
        "речн",
    },
    "garden": {"сад", "оранжере", "ботсад"},
    "kids": {"дет", "коляск", "семейн", "игровая площад", "playground", "детск"},
    "amusements": {
        "аттракцион",
        "колесо обозр",
        "парк развлечен",
        "тир",
        "квест",
        "батут",
    },
    "zoo": {"зоопарк", "дельфинари", "террариум", "аквариум", "питомник"},
    "sport": {
        "спорт",
        "скейтпарк",
        "роллер",
        "стадион",
        "фитнес",
        "workout",
        "воркаут",
        "кроссфит",
    },
    "ice": {"каток", "ледов", "ice rink", "хоккей"},
    "swim": {"бассейн", "аквапарк", "сауна", "термы", "термальный"},
    "climb": {"скалодром", "скалолаз"},
    "bike": {"велодорожк", "прокат велосипед", "bikeshare", "самокат"},
    "market": {
        "рынок",
        "ярмарк",
        "базар",
        "фермерск",
        "экомаркет",
        "фудкорт",
        "фуд-холл",
        "gastronom",
        "food hall",
    },
    "mall": {"тц", "торговый центр", "mall", "outlet", "аутлет"},
    "souvenir": {
        "сувенир",
        "керамик",
        "хохлом",
        "гжель",
        "резьб",
        "ремесл",
        "handmade",
        "craft",
    },
    "vintage": {"винтаж", "блош", "flea", "секонд", "second hand", "комиссионн"},
    "antique": {"антиквар", "антик"},
    "university": {"университет", "вуз", "кампус", "академ"},
    "science": {"научн", "лаборатор", "лектор", "просветител"},
    "planetarium": {"планетарий", "обсерватор"},
    "tech": {
        "технопарк",
        "айти",
        "it-парк",
        "кластер",
        "коворкинг",
        "makerspace",
        "фаблаб",
        "лаборатори",
    },
    "library_science": {"научная библиотек", "техн библиотек"},
    "railway": {
        "вокзал",
        "железнодорож",
        "депо",
        "электродепо",
        "ретро-поезд",
        "музей ж/д",
        "станция",
    },
    "tram_museum": {"трамвайн музей", "трамвайное депо"},
    "bridge": {"мост", "переправ", "эстакад"},
    "pier": {"пристан", "пирс", "причал", "river port", "речной вокзал"},
    "cable_car": {"канатная дорог", "ropeway", "канатк", "фуникулер", "фуникулёр"},
    "photo": {
        "фото",
        "фотогенич",
        "инстаграм",
        "insta",
        "ракурс",
        "фотозон",
        "street photo",
        "sunset",
        "закат",
        "рассвет",
    },
    "nightclub": {"клуб", "night club", "ночн клуб"},
    "karaoke": {"караоке"},
    "concert": {"концерт", "live", "джаз-клуб", "рок-клуб"},
}

COORD_RX = re.compile(r"(-?\d+(?:[.,]\d+)?)\s*[, ]\s*(-?\d+(?:[.,]\d+)?)")
SPAN_RX = re.compile(
    r"(?:от|старт(?:ую)?(?:\s+от)?|я у|я возле|у|рядом с)\s+([^.,;:!?]+)", re.I
)


def haversine(a, b, c, d):
    R = 6371000.0
    p1, p2 = math.radians(a), math.radians(c)
    dphi = p2 - p1
    dl = math.radians(d - b)
    h = math.sin(dphi / 2) ** 2 + math.cos(p1) * math.cos(p2) * math.sin(dl / 2) ** 2
    return 2 * R * math.asin(math.sqrt(h))


# ---- чистим HTML и возвращаем ПОЛНОЕ описание ----
_TAGS_RX = re.compile(r"<[^>]+>")


def _clean_desc(text: str) -> str:
    if not text:
        return ""
    s = html.unescape(str(text))
    s = _TAGS_RX.sub(" ", s)
    s = s.replace("\xa0", " ")
    s = re.sub(r"\s+", " ", s).strip()
    return s


def _parse_wkt_point(s: str):
    if not isinstance(s, str):
        return None
    try:
        s = s.strip()
        s = s[s.find("(") + 1 : s.find(")")]
        lon, lat = [float(x) for x in s.replace(",", " ").split()]
        return lat, lon
    except:
        return None


def load_poi(path=POI_CSV):
    import pandas as pd

    df = pd.read_csv(path)
    c = df["coordinate"].apply(_parse_wkt_point)
    df["lat"] = c.apply(lambda x: x[0] if x else None)
    df["lon"] = c.apply(lambda x: x[1] if x else None)
    df = df.dropna(subset=["lat", "lon"]).copy()
    for col in ["title", "description", "kind"]:
        if col not in df.columns:
            df[col] = ""

    def tags_for(row):
        t = (str(row.get("title", "")) + " " + str(row.get("description", ""))).lower()
        tags = set()
        for k, v in INTERESTS.items():
            if any(w in t for w in v):
                tags.add(k)
        return tags

    df["__tags__"] = df.apply(tags_for, axis=1)
    for col in ["popularity", "rating"]:
        if col not in df.columns:
            df[col] = None
    if "id" not in df.columns:
        df["id"] = range(1, len(df) + 1)
    df["__desc_clean__"] = df["description"].map(_clean_desc)
    return df


POI = None
def ensure_data():
    """Ленивая инициализация данных для маршрутов."""
    global POI
    if POI is not None:
        return
    try:
        import pandas as pd  # noqa: F401
    except ImportError:
        raise RuntimeError("pandas not installed; disable ML or add it to requirements")
    POI = load_poi()


# ---- модель ----
model = None
feature_names = None
if USE_MODEL:
    try:
        from catboost import CatBoostRanker

        if os.path.exists(MODEL_PATH):
            model = CatBoostRanker()
            model.load_model(MODEL_PATH)
            logging.info("CatBoost модель загружена: %s", MODEL_PATH)
            if os.path.exists(MODEL_FEATURES):
                with open(MODEL_FEATURES, "r", encoding="utf-8") as f:
                    feature_names = json.load(f)
                logging.info("Загружен список признаков: %s", MODEL_FEATURES)
            else:
                feature_names = [
                    "dist_m",
                    "inv_dist",
                    "tags_overlap",
                    "query_len",
                    "hours",
                    "poi_popularity",
                    "poi_rating",
                    *[f"want_{k}" for k in sorted(INTERESTS.keys())],
                ]
                logging.warning(
                    "Файл с признаками не найден, используем дефолт: %d фич",
                    len(feature_names),
                )
        else:
            logging.warning("MODEL_PATH не найден, переключаемся на правила")
            USE_MODEL = False
    except Exception as e:
        logging.warning("Не удалось загрузить модель, фолбэк на правила: %s", e)
        USE_MODEL = False


def build_candidate_features(lat0, lon0, text_raw, hours, want_tags, pool=CAND_POOL):
    import pandas as pd

    rows_meta = []
    feats = []
    base_text = (text_raw or "").lower()
    want_onehot = {f"want_{k}": (1 if k in want_tags else 0) for k in INTERESTS.keys()}
    for r in POI.itertuples(index=False):
        dist = haversine(lat0, lon0, r.lat, r.lon)
        tags_overlap = (
            len(want_tags & getattr(r, "__tags__", set())) if want_tags else 0
        )
        rows_meta.append((dist, r, tags_overlap))
        feats.append(
            {
                "dist_m": dist,
                "inv_dist": 1.0 / (1.0 + dist),
                "tags_overlap": float(tags_overlap),
                "query_len": float(len(base_text)),
                "hours": float(hours if hours is not None else 2.0),
                "poi_popularity": float(getattr(r, "popularity", 0) or 0),
                "poi_rating": float(getattr(r, "rating", 0) or 0),
                **{f"want_{k}": want_onehot[f"want_{k}"] for k in INTERESTS.keys()},
            }
        )
    rows_tmp = sorted(
        zip(feats, rows_meta), key=lambda x: (-x[0]["tags_overlap"], x[0]["dist_m"])
    )[:pool]
    feats = [x[0] for x in rows_tmp]
    rows_meta = [x[1] for x in rows_tmp]
    import pandas as pd

    df = pd.DataFrame(feats)
    global feature_names
    for col in feature_names:
        if col not in df.columns:
            df[col] = 0.0
    df = df[[c for c in feature_names if c in df.columns]]
    return df, rows_meta


def rank_candidates(lat0, lon0, want_tags, text_raw, hours):
    if USE_MODEL and model is not None:
        X, meta = build_candidate_features(
            lat0, lon0, text_raw, hours, want_tags, pool=CAND_POOL
        )
        preds = model.predict(X)
        scored = list(zip(preds, meta))
        scored.sort(key=lambda x: float(x[0]), reverse=True)
        return [(None, dist, r, ov) for (pred, (dist, r, ov)) in scored]
    rows = []
    for r in POI.itertuples(index=False):
        dist = haversine(lat0, lon0, r.lat, r.lon)
        overlap = len(want_tags & getattr(r, "__tags__", set())) if want_tags else 0
        score = 2.0 * overlap - dist / 400.0
        rows.append((score, dist, r, overlap))
    rows.sort(reverse=True, key=lambda x: x[0])
    if want_tags:
        hit = [x for x in rows if x[3] > 0][:CAND_POOL]
        miss = [x for x in rows if x[3] == 0][: max(0, CAND_POOL - len(hit))]
        return hit + miss
    return rows[:CAND_POOL]


def _is_park_row(r) -> bool:
    try:
        tags = getattr(r, "__tags__", set())
        if isinstance(tags, set):
            if "park" in tags or "nature" in tags:
                return True
    except Exception:
        pass
    kind = str(getattr(r, "kind", "")).lower()
    if "park" in kind:
        return True
    text = (
        str(getattr(r, "title", "")) + " " + str(getattr(r, "description", ""))
    ).lower()
    return ("парк" in text) or ("сквер" in text)


def build_route(lat0, lon0, hours, interests_raw, text_raw):
    """
    Мягкая закольцовка + лимит точек до 20 (включая старт в ссылке).
    """
    want = canon_interests(interests_raw)
    cand = rank_candidates(lat0, lon0, want, text_raw, hours)

    target = (hours if hours else 2.0) * 3600.0
    route = []
    used = set()
    cur = (lat0, lon0)
    walk_dist = 0.0
    dwell_sum = 0.0
    dwell_list = []


    ROUTE_MAX_POINTS = min(
        50, YANDEX_MAX_POINTS - 1
    ) 

    while True:
        # стоп по количеству
        if len(route) >= ROUTE_MAX_POINTS:
            break

        progress = ((walk_dist / WALK_SPEED_MPS) + dwell_sum) / max(1.0, target)
        progress = max(0.0, min(1.5, progress))
        best = None

        for score0, _, r, ov in cand[:30]:
            if r.id in used:
                continue
            leg = haversine(cur[0], cur[1], r.lat, r.lon)
            dwell_add = STOP_PARK_S if _is_park_row(r) else STOP_FIXED_S
            total_if_add = (walk_dist + leg) / WALK_SPEED_MPS + (dwell_sum + dwell_add)

            if total_if_add > target * (1 + TIME_TOL) and route:
                continue

            back_to_start = haversine(r.lat, r.lon, lat0, lon0)
            loop_penalty = (
                (progress**2) * LOOP_WEIGHT_BASE * (back_to_start / LOOP_BACK_DIVISOR)
            )
            val = (1.5 * (ov or 0)) - (leg / 500.0) - loop_penalty

            if (best is None) or (val > best[0]):
                best = (val, leg, r, dwell_add)

        if best is None:
            break

        _, leg, r, dwell_add = best
        route.append((r, leg))
        used.add(r.id)
        walk_dist += leg
        dwell_sum += dwell_add
        dwell_list.append(dwell_add)
        cur = (r.lat, r.lon)

        if ((walk_dist / WALK_SPEED_MPS) + dwell_sum) >= target * (1 - TIME_TOL):
            break

    walk_time = walk_dist / WALK_SPEED_MPS
    total_time = walk_time + dwell_sum
    return route, walk_dist, walk_time, dwell_list, total_time


def hhmm(seconds):
    m = int(round(seconds / 60.0))
    h, m = divmod(m, 60)
    return f"{h} ч {m:02d} мин" if h else f"{m} мин"


# ---- ссылка на Яндекс.Картах ----
def make_yamaps_link(start_ll, route, max_pts=YANDEX_MAX_POINTS, close_loop=False):
    pts = [f"{start_ll[0]:.6f},{start_ll[1]:.6f}"]
    for r, _ in route:
        if len(pts) >= max_pts:  
            break
        pts.append(f"{r.lat:.6f},{r.lon:.6f}")
    if close_loop and len(pts) < max_pts:
        pts.append(f"{start_ll[0]:.6f},{start_ll[1]:.6f}")
    rtext = "~".join(pts)
    return f"https://yandex.ru/maps/?rtext={quote_plus(rtext)}&rtt=pedestrian"


# ---------------- Утилита: отправка длинных сообщений ----------------
def _chunk_text(txt: str, limit: int = TG_SAFE_LIMIT):
    if len(txt) <= limit:
        return [txt]
    parts = []
    for para in txt.split("\n\n"):
        if len(para) <= limit:
            parts.append(para)
            continue
        buf = ""
        for line in para.split("\n"):
            if len(line) > limit:
                while len(line) > 0:
                    take = min(limit, len(line))
                    cut = line.rfind(" ", 0, take)
                    if cut < int(take * 0.6):
                        cut = take
                    parts.append((buf + line[:cut]).strip())
                    buf = ""
                    line = line[cut:].lstrip()
            else:
                if len(buf) + 1 + len(line) > limit:
                    parts.append(buf.strip())
                    buf = line
                else:
                    buf = (buf + "\n" + line) if buf else line
        if buf:
            parts.append(buf.strip())
        parts.append("")
    if parts and parts[-1] == "":
        parts.pop()
    chunks, cur = [], ""
    for p in parts:
        add = (("\n\n" if cur else "") + p) if p else "\n\n"
        if len(cur) + len(add) > limit:
            if cur:
                chunks.append(cur)
            cur = p
        else:
            cur += add
    if cur:
        chunks.append(cur)
    fixed = []
    for c in chunks:
        if len(c) <= TG_HARD_LIMIT:
            fixed.append(c)
        else:
            s = c
            while len(s) > 0:
                fixed.append(s[:TG_HARD_LIMIT])
                s = s[TG_HARD_LIMIT:]
    return [x for x in fixed if x]


async def send_long(msg: Message, text: str):
    for chunk in _chunk_text(text):
        await msg.answer(chunk)


# ---------------- Диалог на 3 шага ----------------
SESS = {}


def _parse_hours_loose(text: str):
    t = text.lower().strip()
    m = re.search(r"(\d+(?:[.,]\d+)?)\s*час", t)
    if m:
        return float(m.group(1).replace(",", "."))
    m = re.search(r"(\d+)\s*мин", t)
    if m:
        return max(0.5, int(m.group(1)) / 60.0)
    m = re.fullmatch(r"\d+(?:[.,]\d+)?", t)
    if m:
        return float(m.group(0).replace(",", "."))
    return None


def canon_interests(raw_list):
    q = " ".join(x.lower() for x in raw_list)
    got = set()
    for k, vocab in INTERESTS.items():
        if any(w in q for w in vocab):
            got.add(k)
    return got


# ------------ Handlers ------------
@dp.message(CommandStart())
async def on_start(m: Message):
    SESS[m.from_user.id] = {"step": "addr"}
    await send_long(
        m,
        "Шаг 1/3. Укажи адрес старта или координаты «56.328, 44.005». Напиши «сброс» чтобы начать заново.",
    )


@dp.message(F.text.casefold() == "сброс")
async def on_reset(m: Message):
    SESS[m.from_user.id] = {"step": "addr"}
    await send_long(m, "Ок. Шаг 1/3. Адрес старта или координаты?")


@dp.message(F.text.len() > 0)
async def on_text(m: Message):
    uid = m.from_user.id
    state = SESS.get(uid)
    if not state:
        SESS[uid] = {"step": "addr"}
        await send_long(m, "Шаг 1/3. Адрес старта или координаты?")
        return

    step = state["step"]

    if step == "addr":
        lat, lon, where = parse_start(m.text)
        if lat is None:
            await send_long(
                m,
                "Не распознал старт. Отправь адрес или координаты в формате «56.328, 44.005».",
            )
            return
        state.update({"lat": lat, "lon": lon, "where": where, "step": "time"})
        await send_long(
            m,
            "Шаг 2/3. Сколько времени на прогулку? Примеры: «2 часа», «1.5», «45 минут».",
        )
        return

    if step == "time":
        hours = _parse_hours_loose(m.text)
        if hours is None or hours <= 0:
            await send_long(
                m,
                "Не понял время. Напиши число часов, например: 2, 1.5 или «45 минут».",
            )
            return
        state.update({"hours": hours, "step": "int"})
        await send_long(
            m,
            "Шаг 3/3. Интересы через запятую. Примеры: «архитектура, кофе», «музеи, виды», «стрит-арт».",
        )
        return

    if step == "int":
        parts = [p.strip() for p in m.text.split(",") if p.strip()]
        lat, lon, where = state["lat"], state["lon"], state["where"]
        hours = state["hours"]

        route, walk_dist, walk_time, dwell_list, total_time = build_route(
            lat, lon, hours, parts, m.text
        )

        lines = [
            f"Старт: {where}",
            f"Цель: ~{hours:g} ч • План: ходьба {hhmm(walk_time)} + остановки {hhmm(sum(dwell_list))} = {hhmm(total_time)}",
            "",
        ]
        t0 = datetime.datetime.now()
        cur = (lat, lon)
        tsec = 0.0
        for i, (r, d) in enumerate(route, 1):
            walk_seg = d / WALK_SPEED_MPS
            tsec += walk_seg
            eta = (t0 + datetime.timedelta(seconds=tsec)).strftime("%H:%M")
            title = getattr(r, "title", "Без названия")
            dwell_i = dwell_list[i - 1] if i - 1 < len(dwell_list) else STOP_FIXED_S
            park_mark = (
                " • парк 30 мин" if dwell_i == STOP_PARK_S else " • остановка 3 мин"
            )
            descr = getattr(r, "__desc_clean__", "") or _clean_desc(
                getattr(r, "description", "")
            )
            if descr:
                lines.append(
                    f"{i}) {title} • {int(d)} м пешком • прибытие ~{eta}{park_mark}\n   — {descr}\n"
                )
            else:
                lines.append(
                    f"{i}) {title} • {int(d)} м пешком • прибытие ~{eta}{park_mark}\n"
                )
            tsec += dwell_i
            cur = (r.lat, r.lon)

        
        close_loop = False
        if route:
            last_lat, last_lon = route[-1][0].lat, route[-1][0].lon
            back_m = haversine(last_lat, last_lon, lat, lon)
            back_km = back_m / 1000.0
            if back_km > CLOSE_LOOP_KM:
                back_time = back_m / WALK_SPEED_MPS
                total_time += back_time
                lines.append(
                    f"\nФиниш далеко от старта (≈{back_km:.1f} км), добавляю возврат пешком: +{hhmm(back_time)}."
                )
                close_loop = True
            else:
                lines.append(
                    f"\nФиниш в пределах ≈{back_km:.1f} км от старта — возврат можно не учитывать."
                )

        
        url = make_yamaps_link(
            (lat, lon), route, max_pts=YANDEX_MAX_POINTS, close_loop=close_loop
        )
        lines.append(f"Ссылка на маршрут в Яндекс.Картах:\n{url}")

        await send_long(m, "\n".join(lines))
        SESS.pop(uid, None)
        return



async def main():
    await dp.start_polling(bot)


if __name__ == "__main__":
    import asyncio
    from aiogram import Bot
    token = os.getenv("TELEGRAM_BOT_TOKEN")  # ты уже читаешь его так
    if not token:
        raise RuntimeError("TELEGRAM_BOT_TOKEN is not set")

    bot = Bot(token)

    async def _main():
        await dp.start_polling(
            bot, allowed_updates=dp.resolve_used_update_types()
        )

    asyncio.run(_main())
