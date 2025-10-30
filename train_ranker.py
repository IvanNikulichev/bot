# -*- coding: utf-8 -*-
import sys, numpy as np, pandas as pd
from catboost import CatBoostRanker, Pool

NUM  = ["distance_km","inv_distance","log_distance","rank_by_distance","hours",
        "eta_walk_min","stay_min","semantic_cos","token_overlap","interest_match_count"]
BIN  = ["kind_match","view_hint","eta_fit"]
HB   = []  # бины по часам теперь не нужны
KIND = ["kind_cafe","kind_dessert","kind_museum","kind_viewpoint",
        "kind_street_art","kind_historic","kind_architecture","kind_park"]
COLS = NUM + BIN + KIND

def make_labels(df: pd.DataFrame) -> pd.Series:
    y = ((df["kind_match"] == 1) & (df["distance_km"] <= 1.5)).astype(int)
    rng = np.random.default_rng(42)
    flip = rng.random(len(y)) < 0.12
    y = y.copy(); y[flip] = 1 - y[flip]
    return y

def train(features_parquet: str, model_out: str = "poi_ranker.cbm"):
    df = pd.read_parquet(features_parquet)

    # веса «интересов» → сильнее влияние предпочтений
    df["w"] = (
        1.0*df.get("interest_match_count", 0).fillna(0) +
        0.3*df.get("token_overlap", 0).fillna(0) +
        0.2*df.get("semantic_cos", 0).fillna(0)
    ).clip(upper=5.0)

    df["y"] = make_labels(df)

    # сплит по запросам
    qids = df["query_id"].astype(str).unique()
    rng = np.random.default_rng(0); rng.shuffle(qids)
    cut = int(0.8*len(qids))
    tr_q, va_q = set(qids[:cut]), set(qids[cut:])
    tr = df[df.query_id.astype(str).isin(tr_q)].copy()
    va = df[df.query_id.astype(str).isin(va_q)].copy()

    def mkpool(x: pd.DataFrame) -> Pool:
        return Pool(x[COLS], label=x["y"], group_id=x["query_id"], weight=x["w"])

    m = CatBoostRanker(
        loss_function="YetiRankPairwise",
        iterations=600,
        depth=6,
        learning_rate=0.08,
        random_seed=42,
        verbose=100,
        eval_metric="NDCG:top=10"
    )
    m.fit(mkpool(tr), eval_set=mkpool(va))
    m.save_model(model_out)
    print("saved:", model_out)

if __name__ == "__main__":
    out = sys.argv[2] if len(sys.argv) > 2 else "poi_ranker.cbm"
    train(sys.argv[1], out)
