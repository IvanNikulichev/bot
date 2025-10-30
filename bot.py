# bot.py — AI-режим (CatBoost) + безопасный фолбэк на правила
import os, re, math, json, asyncio, logging, datetime, requests
from aiogram import Bot, Dispatcher, F
from aiogram.types import Message
from aiogram.filters import CommandStart
from dotenv import load_dotenv

load_dotenv()

# ------------ Конфиг ------------
TOKEN = os.getenv("TELEGRAM_BOT_TOKEN")
CONTACT_EMAIL = os.getenv("CONTACT_EMAIL", "you@example.com")
POI_CSV = os.getenv("POI_CSV", "poi_csv.csv")
MODEL_PATH = os.getenv("MODEL_PATH", "poi_ranker.cbm")
MODEL_FEATURES = os.getenv("MODEL_FEATURES", "model_features.json")  # список признаков в нужном порядке

USE_MODEL = True                      # можешь выключить модель, оставив фолбэк
WALK_SPEED_MPS = 3.6                  # скорость пешком
STOPS_COUNT   = 5                     # всегда 5 точек
CAND_POOL     = 60                    # сколько кандидатов брать на ранжирование
MIN_DWELL_S   = 5 * 60
MAX_DWELL_S   = 25 * 60

logging.basicConfig(level=logging.INFO)
bot = Bot(TOKEN)
dp  = Dispatcher()

# ------------ Словари интересов/алиасы ------------
ALIASES = {
    "чкаловск":"Чкаловская лестница","пл. минина":"площадь Минина и Пожарского",
    "московск":"Московский вокзал","ильинк":"улица Ильинская",
    "нижне-волж":"Нижне-Волжская набережная","верхне-волж":"Верхне-Волжская набережная",
    "покровск":"улица Большая Покровская","федоровск":"наб. Федоровского",
    "стрелк":"метро Стрелка","канат":"Канатная дорога Нижний Новгород",
    "варварск":"улица Варварская","сенная":"площадь Сенная",
    "добролюб":"улица Добролюбова","ковалихин":"улица Ковалихинская","звездинк":"улица Звездинка",
}
INTERESTS = {
    # ===== ЕДА / НАПИТКИ =====
    "coffee": {"кофе","кофей","coffee","эспрессо","espresso","капучино","латте","раф","флэт уайт","aeropress","аэропресс","v60","спешалти","specialty"},
    "tea": {"чай","tea","матча","маття","пуэр","улун","сенча","гойчай"},
    "dessert": {"десерт","пирожн","эклер","макарон","чизкейк","маффин","кейк","штрудель","тирамису","выпечк","торт","морожен","ice cream","gelato"},
    "bakery": {"пекарн","булочн","круассан","круасан","хлеб","багет","фокачч","bakery","boulangerie"},
    "brunch": {"бранч","завтрак","скрэмбл","омлет","яичниц","авокадо тост","панкейк","pancake","французский тост"},
    "pizza": {"пицц","pizzeria","пиццер"},
    "pasta": {"паста","равиоли","тальятелле","спагетт","лазан"},
    "steak": {"стейк","гриль","bbq","барбекю","брискет"},
    "burger": {"бургер","burger","котлета","фри"},
    "shawarma": {"шаурм","шаверм","донер","кебаб"},
    "georgian": {"грузин","хинкал","хачапур","лобио","сацив"},
    "italian": {"итал","траттор","остерия","risotto","рикотт"},
    "japanese": {"япон","суши","ролл","рамен","удон","донбури","якитор"},
    "chinese": {"китай","димсам","лапша по","бао","гунбао","сычуан"},
    "korean": {"корее","кимчи","рамён","коги","ттокпокки","самгёпсаль"},
    "thai": {"тайск","том ям","том кха","пад тай","сом там"},
    "vietnamese": {"вьетнам","фо","бон бо","бан ми"},
    "indian": {"индий","карри","наан","масала","тандури"},
    "uzbek": {"узбек","плов","самса","лагман","манты"},
    "caucasian": {"кавказ","шашлык","долма","чурчхел"},
    "mexican": {"мексик","тако","буррито","начос","кесадиль"},
    "turkish": {"турец","пиде","люля","донер","бахлава","баклава"},
    "lebanese": {"ливан","хумус","фалафель","табуле","шаварма"},
    "vegan": {"веган","vegan","plant-based","без мяса"},
    "vegetarian": {"вегетари","vegetarian"},
    "gluten_free": {"без глютен","gluten free"},
    "halal": {"халал","halal"},
    "kosher": {"кошер","kosher"},
    "bar": {"бар","pub","паб","винн","вино","винный","сидр","крафт","пивн","brewery","пивовар","коктейл","cocktail","mixology","speakeasy","роофтап","rooftop"},
    "hookah": {"кальян","hookah","shisha"},
    "rooftop": {"rooftop","крыша","панорамный бар","видовой бар"},

    # ===== ИСКУССТВО / КУЛЬТУРА =====
    "street_art": {"стрит-арт","street art","мурал","муралы","граффити"},
    "gallery": {"галере","арт-цент","выставк","арт-пространств","art space","центр соврем"},
    "museum": {"музей","экспозици","ретро","диорама"},
    "theatre": {"театр","драм","опер","балет","сцена"},
    "cinema": {"кино","cinema","movie","киноцентр"},
    "music": {"концерт","клуб","джаз","рок","live","филармони"},
    "monument": {"памятник","монумент","скульптур","стела","бюст"},
    "library": {"библиотек","читальн","mediatheque"},
    "bookstore": {"книжн","bookshop","bookstore"},

    # ===== ИСТОРИЯ / АРХИТЕКТУРА / РЕЛИГИЯ =====
    "history": {"историческ","купеческ","кремль","арсенал","фортификац","краевед","музей истории"},
    "architecture": {"архитектур","фасад","особняк","доходн","усадебн","памятник архитектуры","ордер","пилястр"},
    "baroque": {"барокк"},
    "classicism": {"классицизм","ампир","empire"},
    "art_nouveau": {"модерн","арт-нуво","jugendstil","сецессион"},
    "constructivism": {"конструктивизм","авангар","советск модерн","баухауз"},
    "brutalism": {"брутализм"},
    "soviet_modernism": {"советск модерн","модернизм 60","нииб"},
    "wooden": {"деревянн","резн","наличник","деревянное зодчество"},
    "manor": {"усадьб","помещич","дворянск усадьб"},
    "palace": {"дворец","палас"},
    "fortress": {"крепост","форт","валы"},
    "kremlin": {"кремль"},
    "cemetery": {"кладбищ","некропол"},
    "church": {"церковь","собор","храм","колокольн","монастыр","часовн","лавра"},
    "mosque": {"мечет"},
    "synagogue": {"синагог"},

    # ===== ПРИРОДА / ВИДЫ / НАБЕРЕЖНЫЕ =====
    "view": {"вид","виды","панорама","обзорная","обзорн","viewpoint","смотор","белведер","колесо обозр"},
    "embankment": {"набережн","бережн","boulevard","bulvar","бульвар"},
    "river_volga": {"волга","стрелка","ярмарочн площад"},
    "river_oka": {"ока"},
    "park": {"парк","сквер","сад","ботаническ","дендрар","аллея"},
    "nature": {"лес","роща","овраг","ущелье","утес","берег","пляж","остров","луга","речн"},
    "garden": {"сад","оранжере","ботсад"},

    # ===== СЕМЬЯ / ДЕТИ / РАЗВЛЕЧЕНИЯ / СПОРТ =====
    "kids": {"дет","коляск","семейн","игровая площад","playground","детск"},
    "amusements": {"аттракцион","колесо обозр","парк развлечен","тир","квест","батут"},
    "zoo": {"зоопарк","дельфинари","террариум","аквариум","питомник"},
    "sport": {"спорт","скейтпарк","роллер","стадион","фитнес","workout","воркаут","кроссфит"},
    "ice": {"каток","ледов","ice rink","хоккей"},
    "swim": {"бассейн","аквапарк","сауна","термы","термальный"},
    "climb": {"скалодром","скалолаз"},
    "bike": {"велодорожк","прокат велосипед","bikeshare","самокат"},

    # ===== ШОПИНГ / РЫНКИ / ЯРМАРКИ =====
    "market": {"рынок","ярмарк","базар","фермерск","экомаркет","фудкорт","фуд-холл","gastronom","food hall"},
    "mall": {"тц","торговый центр","mall","outlet","аутлет"},
    "souvenir": {"сувенир","керамик","хохлом","гжель","резьб","ремесл","handmade","craft"},
    "vintage": {"винтаж","блош","flea","секонд","second hand","комиссионн"},
    "antique": {"антиквар","антик"},

    # ===== ОБРАЗОВАНИЕ / ТЕХНО / ПЛАНЕТАРИИ =====
    "university": {"университет","вуз","кампус","академ"},
    "science": {"научн","лаборатор","лектор","просветител"},
    "planetarium": {"планетарий","обсерватор"},
    "tech": {"технопарк","айти","it-парк","кластер","коворкинг","makerspace","фаблаб","лаборатори"},
    "library_science": {"научная библиотек","техн библиотек"},

    # ===== ТРАНСПОРТ / ИНФРА =====
    "railway": {"вокзал","железнодорож","депо","электродепо","ретро-поезд","музей ж/д","станция"},
    "tram_museum": {"трамвайн музей","трамвайное депо"},
    "bridge": {"мост","переправ","эстакад"},
    "pier": {"пристан","пирс","причал","river port","речной вокзал"},
    "cable_car": {"канатная дорог","ropeway","канатк","фуникулер","фуникулёр"},

    # ===== ФОТО / НОЧНАЯ ЖИЗНЬ =====
    "photo": {"фото","фотогенич","инстаграм","insta","ракурс","фотозон","street photo","sunset","закат","рассвет"},
    "nightclub": {"клуб","night club","ночн клуб"},
    "karaoke": {"караоке"},
    "concert": {"концерт","live","джаз-клуб","рок-клуб"},
}


# ------------ Гео/парсинг ------------
COORD_RX = re.compile(r"(-?\d+(?:[.,]\d+)?)\s*[, ]\s*(-?\d+(?:[.,]\d+)?)")
SPAN_RX  = re.compile(r"(?:от|старт(?:ую)?(?:\s+от)?|я у|я возле|у|рядом с)\s+([^.,;:!?]+)", re.I)

def haversine(a,b,c,d):
    R=6371000.0
    p1,p2=math.radians(a),math.radians(c)
    dphi=p2-p1; dl=math.radians(d-b)
    h=math.sin(dphi/2)**2+math.cos(p1)*math.cos(p2)*math.sin(dl/2)**2
    return 2*R*math.asin(math.sqrt(h))

def _address_hints(text:str):
    t=text.strip(); hints=[]
    m=COORD_RX.search(t)
    if m:
        lat=m.group(1).replace(",","."); lon=m.group(2).replace(",",".")
        return [f"{lat},{lon}"]
    m=SPAN_RX.search(t)
    if m: hints.append(m.group(1).strip())
    for pat in [" на ", " от ", " у "]:
        i=t.lower().find(pat)
        if i>=0: hints.append(t[i+len(pat):].split(".")[0][:80])
    low=t.lower()
    for k,v in ALIASES.items():
        if k in low: hints.append(v)
    if not hints: hints.append(t[:120])
    return list(dict.fromkeys([s.strip(' "\'«»') for s in hints if s.strip()]))

def parse_start(text:str):
    m=COORD_RX.search(text)
    if m:
        lat=float(m.group(1).replace(",", ".")); lon=float(m.group(2).replace(",", "."))
        return lat, lon, f"{lat:.5f},{lon:.5f}"
    ua=f"tg-bot/route (+{CONTACT_EMAIL})"
    for hint in _address_hints(text):
        q = hint if "нижн" in hint.lower() else f"{hint}, Нижний Новгород, Россия"
        try:
            r=requests.get("https://nominatim.openstreetmap.org/search",
                           params={"q":q,"format":"jsonv2","limit":1},
                           headers={"User-Agent":ua}, timeout=15)
            if r.ok and r.json():
                j=r.json()[0]
                return float(j["lat"]), float(j["lon"]), j.get("display_name", hint)
        except Exception:
            pass
    return None, None, "Адрес не распознан"

def parse_hours(text:str):
    m=re.search(r"(\d+(?:[.,]\d+)?)\s*час", text.lower())
    return float(m.group(1).replace(",", ".")) if m else None

def canon_interests(raw_list):
    q = " ".join(x.lower() for x in raw_list)
    got=set()
    for k, vocab in INTERESTS.items():
        if any(w in q for w in vocab):
            got.add(k)
    return got

# ------------ POI и признаки ------------
def _parse_wkt_point(s:str):
    if not isinstance(s,str): return None
    try:
        s=s.strip(); s=s[s.find("(")+1:s.find(")")]
        lon,lat=[float(x) for x in s.replace(","," ").split()]
        return lat,lon
    except: return None

def load_poi(path=POI_CSV):
    import pandas as pd
    df=pd.read_csv(path)
    c=df["coordinate"].apply(_parse_wkt_point)
    df["lat"]=c.apply(lambda x: x[0] if x else None)
    df["lon"]=c.apply(lambda x: x[1] if x else None)
    df=df.dropna(subset=["lat","lon"]).copy()

    # авто-теги из текста
    def tags_for(row):
        t=(str(row.get("title",""))+" "+str(row.get("description",""))).lower()
        tags=set()
        for k,v in INTERESTS.items():
            if any(w in t for w in v): tags.add(k)
        return tags
    df["__tags__"]=df.apply(tags_for, axis=1)
    # безопасные поля: popularity/rating/kind если есть
    for col in ["popularity","rating","kind"]:
        if col not in df.columns: df[col]=None
    # id нужен для маршрута
    if "id" not in df.columns:
        df["id"] = range(1, len(df)+1)
    return df

POI = load_poi()

# Попытка загрузить модель и список признаков
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
                # дефолтный набор (должен совпасть с train_ranker.py)
                feature_names = [
                    "dist_m","inv_dist","tags_overlap","query_len","hours",
                    "poi_popularity","poi_rating",
                    # бинарные интересы (one-hot) — устойчиво и прозрачно
                    *[f"want_{k}" for k in sorted(INTERESTS.keys())],
                ]
                logging.warning("Файл с признаками не найден, используем дефолт: %d фич", len(feature_names))
        else:
            logging.warning("MODEL_PATH не найден, переключаемся на правила")
            USE_MODEL = False
    except Exception as e:
        logging.warning("Не удалось загрузить модель, фолбэк на правила: %s", e)
        USE_MODEL = False

def build_candidate_features(lat0, lon0, text_raw, hours, want_tags, pool=CAND_POOL):
    """Возвращает список (df, rows_meta) — df с признаками для модели, rows_meta для сборки ответа."""
    import pandas as pd
    rows_meta=[]
    feats=[]
    base_text = text_raw.lower()
    want_onehot = {f"want_{k}": (1 if k in want_tags else 0) for k in INTERESTS.keys()}
    for r in POI.itertuples(index=False):
        dist = haversine(lat0, lon0, r.lat, r.lon)
        tags_overlap = len(want_tags & getattr(r, "__tags__", set())) if want_tags else 0
        rows_meta.append((dist, r, tags_overlap))
        feats.append({
            "dist_m": dist,
            "inv_dist": 1.0/(1.0+dist),
            "tags_overlap": float(tags_overlap),
            "query_len": float(len(base_text)),
            "hours": float(hours if hours is not None else 2.0),
            "poi_popularity": float(getattr(r, "popularity", 0) or 0),
            "poi_rating": float(getattr(r, "rating", 0) or 0),
            **{f"want_{k}": want_onehot[f"want_{k}"] for k in INTERESTS.keys()},
        })
    # отсортировать верхний пул по простому скору (чтобы не кормить модель весь город)
    rows_tmp = sorted(zip(feats, rows_meta), key=lambda x: (-x[0]["tags_overlap"], x[0]["dist_m"]))[:pool]
    feats = [x[0] for x in rows_tmp]
    rows_meta = [x[1] for x in rows_tmp]
    # dataframe + упорядочить/добить признаки
    df = pd.DataFrame(feats)
    # приведём к ожидаемому порядку, недостающие — нули, лишние — отбросим
    global feature_names
    for col in feature_names:
        if col not in df.columns: df[col] = 0.0
    df = df[[c for c in feature_names if c in df.columns]]
    return df, rows_meta

# ------------ Ранжирование + маршрут/тайминг ------------
def rank_candidates(lat0, lon0, want_tags, text_raw, hours):
    if USE_MODEL and model is not None:
        X, meta = build_candidate_features(lat0, lon0, text_raw, hours, want_tags, pool=CAND_POOL)
        preds = model.predict(X)
        scored = list(zip(preds, meta))
        scored.sort(key=lambda x: float(x[0]), reverse=True)
        return [(None, dist, r, ov) for (pred, (dist, r, ov)) in scored]
    # фолбэк-правила
    rows=[]
    for r in POI.itertuples(index=False):
        dist = haversine(lat0, lon0, r.lat, r.lon)
        overlap = len(want_tags & getattr(r,"__tags__", set())) if want_tags else 0
        score = 2.0*overlap - dist/400.0
        rows.append((score, dist, r, overlap))
    rows.sort(reverse=True, key=lambda x: x[0])
    if want_tags:
        hit=[x for x in rows if x[3]>0][:CAND_POOL]
        miss=[x for x in rows if x[3]==0][:max(0,CAND_POOL-len(hit))]
        return hit+miss
    return rows[:CAND_POOL]

def build_route(lat0, lon0, hours, interests_raw, text_raw):
    want = canon_interests(interests_raw)
    cand = rank_candidates(lat0, lon0, want, text_raw, hours)
    route=[]; used=set(); cur=(lat0,lon0)
    # жадный выбор 5 точек
    while len(route) < STOPS_COUNT and cand:
        best=None
        for sc,_,r,ov in cand[:20]:
            if r.id in used: continue
            d = haversine(cur[0],cur[1], r.lat, r.lon)
            val = (1.5*ov if want else 0.0) - d/500.0 - 0.05*len(route)
            if (best is None) or (val > best[0]): best=(val,d,r,ov)
        if best is None: break
        _, d, r, ov = best
        route.append((r,d))
        used.add(r.id); cur=(r.lat,r.lon)
    # если не хватило — доберём ближайшими
    i=0
    while len(route)<STOPS_COUNT and i<len(cand):
        r=cand[i][2]
        if r.id not in used:
            d=haversine(cur[0],cur[1], r.lat, r.lon)
            route.append((r,d)); used.add(r.id); cur=(r.lat,r.lon)
        i+=1
    # тайминг под заданные часы
    walk_dist = sum(d for _,d in route)
    walk_time = walk_dist / WALK_SPEED_MPS
    target = (hours if hours else 2.0) * 3600.0
    dwell_total = max(0.0, target - walk_time)
    per_stop = dwell_total / max(1,len(route))
    per_stop = max(MIN_DWELL_S, min(MAX_DWELL_S, per_stop))
    total_time = walk_time + per_stop*len(route)
    return route, walk_dist, walk_time, per_stop, total_time

def hhmm(seconds):
    m = int(round(seconds/60.0))
    h, m = divmod(m, 60)
    return (f"{h} ч {m:02d} мин" if h else f"{m} мин")

# ------------ Handlers ------------
@dp.message(CommandStart())
async def on_start(m:Message):
    mode = "AI-модель" if (USE_MODEL and model is not None) else "правила"
    await m.answer(
        f"Готов строить прогулки (режим: {mode}).\n"
        "Напиши: «2 часа от Чкаловской лестницы, архитектура и виды» "
        "или «56.328, 44.005, 1.5 часа, стрит-арт, кофе»."
    )

@dp.message(F.text.len()>0)
async def on_text(m:Message):
    lat, lon, where = parse_start(m.text)
    if lat is None:
        await m.answer("Не распознал старт. Укажи адрес или координаты.")
        return
    parts=[p.strip() for p in m.text.split(",")]
    interests = parts[1:] if len(parts)>1 else []
    hours = parse_hours(m.text)
    route, walk_dist, walk_time, dwell_per_stop, total_time = build_route(lat, lon, hours, interests, m.text)

    lines=[f"Старт: {where}",
           f"Цель: ~{(hours or 2.0):g} ч • План: ходьба {hhmm(walk_time)} + остановки по {hhmm(dwell_per_stop)} × {len(route)} = {hhmm(total_time)}",
           ""]
    t0 = datetime.datetime.now()
    cur=(lat,lon); tsec=0.0
    for i,(r,d) in enumerate(route,1):
        walk_seg = d / WALK_SPEED_MPS
        tsec += walk_seg
        eta = (t0 + datetime.timedelta(seconds=tsec)).strftime("%H:%M")
        title = getattr(r,"title","Без названия")
        lines.append(f"{i}) {title} • {int(d)} м пешком • прибытие ~{eta}")
        tsec += dwell_per_stop
        cur=(r.lat,r.lon)
    await m.answer("\n".join(lines))

# ------------ Main ------------
async def main():
    await dp.start_polling(bot)

if __name__=="__main__":
    asyncio.run(main())
