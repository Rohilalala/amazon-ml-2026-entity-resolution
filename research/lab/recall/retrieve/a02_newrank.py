"""HOLDOUT/HEAD: recovery vs budget for rules on the dense lists: raw rank <= k, 'new rank' (rank among the S1's dense
neighbours of that source that are NOT sparse candidates) <= m, and relative floor cos >= top1_cos - delta.
-> out/a02_rules.json"""
import json, sys, numpy as np, polars as pl
sys.path.insert(0, __file__.rsplit("/", 1)[0])
from a01_eval import ROOT, D, dense, cand_pairs, new_pairs, COUNTRIES
truth = pl.read_parquet(f"{ROOT}/v8_lab/ce/setup/out/truth_train.parquet").select("i", "j", pl.lit(1, pl.Int8).alias("y"))
R = {}
for name in ("holdout", "head"):
    rows = np.load(f"{D}/sets/{name}_rows.npy")
    d = new_pairs(pl.concat([dense("train", c, rows) for c in COUNTRIES["train"]]), cand_pairs("train", rows))
    d = d.join(truth.filter(pl.col("i").is_in(pl.Series(rows).implode())), on=["i", "j"], how="left").with_columns(pl.col("y").fill_null(0))
    d = d.with_columns(pl.col("cos").max().over("i", "src").alias("top1"))
    n = d.filter(~pl.col("in_cand")).with_columns(pl.col("rank").rank("ordinal").over("i", "src").alias("new_rank"))
    out = []
    for m in (1, 2, 3, 4, 5):
        s = n.filter(pl.col("new_rank") <= m)
        out.append({"rule": f"new_rank<={m}", "new_per_s1": round(s.height / len(rows), 3), "recovered": int(s["y"].sum())})
    for m in (2, 3, 5):
        for dl in (0.01, 0.02, 0.03, 0.05):
            s = n.filter((pl.col("new_rank") <= m) & (pl.col("cos") >= pl.col("top1") - dl))
            out.append({"rule": f"new_rank<={m}&cos>=top1-{dl}", "new_per_s1": round(s.height / len(rows), 3), "recovered": int(s["y"].sum())})
    for k in (5, 10, 20):
        for dl in (0.01, 0.02, 0.03, 0.05):
            s = n.filter((pl.col("rank") <= k) & (pl.col("cos") >= pl.col("top1") - dl))
            out.append({"rule": f"rank<={k}&cos>=top1-{dl}", "new_per_s1": round(s.height / len(rows), 3), "recovered": int(s["y"].sum())})
    R[name] = out
    for e in out: print(name, e, flush=True)
json.dump(R, open(f"{D}/a02_rules.json", "w"), indent=1)
