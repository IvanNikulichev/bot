import sys, json
from pathlib import Path
import numpy as np
import pandas as pd
from catboost import CatBoostRanker, Pool

SEED = 42


def make_fallback_labels(df: pd.DataFrame) -> pd.Series:
    km = df.get("kind_match", 0).fillna(0).astype(int)
    dist = pd.to_numeric(df.get("distance_km", np.nan), errors="coerce")
    y = ((km == 1) & (dist <= 1.5)).astype(int)
    rng = np.random.default_rng(SEED)
    flip = rng.random(len(y)) < 0.12
    y = y.copy()
    y[flip] = 1 - y[flip]
    return y


def load_feature_list(df: pd.DataFrame) -> list[str]:
    jf = Path("model_features.json")
    if jf.exists():
        cols = [
            c for c in json.loads(jf.read_text(encoding="utf-8")) if c in df.columns
        ]
        if not cols:
            raise RuntimeError("model_features.json не совпадает с parquet")
        return cols
    num = df.select_dtypes(include=["number", "float", "int", "bool"]).columns.tolist()
    drop = {"y", "query_id", "_gid", "group_id", "qid", "poi_id"}
    cols = [c for c in num if c not in drop]
    if not cols:
        raise RuntimeError("Не найдено числовых фич")
    jf.write_text(json.dumps(cols, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"Saved feature list to model_features.json ({len(cols)} cols)")
    return cols


def mkpool(x: pd.DataFrame, cols: list[str]) -> Pool:
    return Pool(x[cols], label=x["y"], group_id=x["_gid"])


def train(parquet_path: str, model_out: str = "poi_ranker.cbm"):
    df = pd.read_parquet(parquet_path)

    # метка
    df["y"] = df["y"].astype(int) if "y" in df.columns else make_fallback_labels(df)

    if "query_id" not in df.columns:
        raise RuntimeError("В parquet отсутствует query_id")
    qcodes, quniques = pd.factorize(df["query_id"].astype(str), sort=False)
    df["_gid"] = qcodes.astype(int)

    # список фич
    COLS = load_feature_list(df)

    # разбиение по группам
    gids = np.unique(df["_gid"].values)
    rng = np.random.default_rng(SEED)
    rng.shuffle(gids)
    cut = int(0.8 * len(gids))
    tr_set, va_set = set(gids[:cut]), set(gids[cut:])
    tr = df[df["_gid"].isin(tr_set)].copy()
    va = df[df["_gid"].isin(va_set)].copy()

    model = CatBoostRanker(
        loss_function="YetiRankPairwise",
        eval_metric="NDCG:top=10",
        iterations=700,
        depth=6,
        learning_rate=0.06,
        random_seed=SEED,
        verbose=100,
    )
    model.fit(mkpool(tr, COLS), eval_set=mkpool(va, COLS))
    model.save_model(model_out)
    print("saved:", model_out)


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("usage: python train_ranker.py features.parquet [poi_ranker.cbm]")
        sys.exit(2)
    out = sys.argv[2] if len(sys.argv) > 2 else "poi_ranker.cbm"
    train(sys.argv[1], out)
