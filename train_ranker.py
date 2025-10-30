# train_ranker.py
import sys
import numpy as np, pandas as pd
from catboost import CatBoostRanker, Pool
import json, pathlib

NUM  = ["distance_km","inv_distance","log_distance","rank_by_distance","hours",
        "poi_pop_local","eta_walk_min","stay_min","semantic_cos","token_overlap","interest_match_count"]
BIN  = ["kind_match","view_hint","eta_fit"]
HB   = ["hb_le1","hb_1_2","hb_2_3","hb_gt3","hb_unknown"]
KIND = ["kind_cafe","kind_dessert","kind_museum","kind_viewpoint","kind_street_art",
        "kind_historic","kind_architecture","kind_park"]
COLS = NUM + BIN + HB + KIND

def make_labels(df: pd.DataFrame) -> pd.Series:
    y = ((df["kind_match"]==1) & (df["distance_km"]<=1.5)).astype(int)
    rng = np.random.default_rng(42)
    flip = rng.random(len(y)) < 0.12
    y = y.copy(); y[flip] = 1 - y[flip]
    return y

def ensure_cols(df: pd.DataFrame, cols) -> pd.DataFrame:
    out = df.copy()
    for c in cols:
        if c not in out.columns:
            out[c] = 0.0
    return out

def train(features_parquet:str, model_out:str="poi_ranker.cbm"):
    df = pd.read_parquet(features_parquet)
    df = ensure_cols(df, COLS)

    # >>> сохраняем порядок признаков, на которых обучаемся
    feat_path = pathlib.Path("model_features.json")
    with open(feat_path, "w", encoding="utf-8") as f:
        json.dump(COLS, f, ensure_ascii=False, indent=2)
    print("Saved feature list to", feat_path)
    # <<<

    df["y"] = make_labels(df)

    qids = df["query_id"].unique()
    rng = np.random.default_rng(0); rng.shuffle(qids)
    cut = int(0.8*len(qids))
    tr_q, va_q = set(qids[:cut]), set(qids[cut:])
    tr = df[df.query_id.isin(tr_q)].copy()
    va = df[df.query_id.isin(va_q)].copy()

    def mkpool(x: pd.DataFrame):
        return Pool(x[COLS], label=x["y"], group_id=x["query_id"])

    m = CatBoostRanker(
        loss_function="YetiRankPairwise",
        iterations=400, depth=6, learning_rate=0.08,
        random_seed=42, verbose=100, eval_metric="NDCG:top=10"
    )
    m.fit(mkpool(tr), eval_set=mkpool(va))
    m.save_model(model_out)
    print("saved:", model_out)

if __name__ == "__main__":
    out = sys.argv[2] if len(sys.argv)>2 else "poi_ranker.cbm"
    train(sys.argv[1], out)
