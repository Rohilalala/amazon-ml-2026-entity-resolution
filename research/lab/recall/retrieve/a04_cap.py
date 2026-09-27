"""Hub S1 check: union(fwd new_rank<=m, rev_rank<=1) with the reverse-only pairs capped at C per S1 (highest cos first).
-> out/a04_cap.json"""
import json, sys, numpy as np, polars as pl
sys.path.insert(0, __file__.rsplit("/", 1)[0])
from a01_eval import ROOT, D, CACHE, COUNTRIES
from a03_rev import pairs
truth = pl.read_parquet(f"{ROOT}/v8_lab/ce/setup/out/truth_train.parquet").select("i", "j", pl.lit(1, pl.Int8).alias("y"))
recs = pl.read_parquet(f"{CACHE['test']}/recs.parquet", columns=["row", "src", "country_key"])
R = {}


def rules(d, n_s1, has_y):
    fw = lambda m: pl.col("new_rank").is_not_null() & (pl.col("new_rank") <= m)
    rv = pl.col("rev_rank").is_not_null() & (pl.col("rev_rank") <= 1)
    d = d.with_columns(pl.coalesce("cos", "rcos").alias("c"))
    out = {}
    for m in (1, 2):
        for cap in (3, 5, 10, 1000000):
            ro = d.filter(rv & ~fw(m)).with_columns(pl.col("c").rank("ordinal", descending=True).over("i").alias("rr")).filter(pl.col("rr") <= cap)
            s = pl.concat([d.filter(fw(m)).select("i", "j", *(["y"] if has_y else [])), ro.select("i", "j", *(["y"] if has_y else []))])
            out[f"m{m}_r1_cap{cap}"] = {"new_per_s1": round(s.height / n_s1, 3), "pairs": s.height} | ({"recovered": int(s["y"].sum())} if has_y else {})
    return out


for name in ("holdout", "head"):
    rows = np.load(f"{D}/sets/{name}_rows.npy")
    d = pl.concat([pairs("train", c, rows) for c in COUNTRIES["train"]]).join(truth, on=["i", "j"], how="left").with_columns(pl.col("y").fill_null(0))
    R[name] = rules(d, len(rows), True); print(name, R[name], flush=True)
tot = {}
for c in COUNTRIES["test"]:
    rows = recs.filter((pl.col("country_key") == c) & (pl.col("src") == 1))["row"].to_numpy()
    R[f"test_{c}"] = rules(pairs("test", c, rows), len(rows), False)
    for k, v in R[f"test_{c}"].items(): tot[k] = tot.get(k, 0) + v["pairs"]
n = recs.filter(pl.col("src") == 1).height
R["test_all_new_per_s1"] = {k: round(v / n, 3) for k, v in tot.items()}
print("test", R["test_all_new_per_s1"], flush=True)
json.dump(R, open(f"{D}/a04_cap.json", "w"), indent=1)
