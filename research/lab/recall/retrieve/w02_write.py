"""Write new-pair lists (pairs outside the sparse candidate set) for HOLDOUT, HEAD and TEST.
usage: w02_write.py M:R:CAP [M:R:CAP ...]
  rule file: forward new_rank <= M (per source) OR reverse rev_rank <= R, the reverse-only pairs capped at CAP per S1
             (highest cosine first); no cosine floor
  wide file: forward raw dense_rank <= 10 OR reverse rev_rank <= 3 (superset; filter on the rank columns)
-> out/pairs/{holdout,head,test}_{m{M}_r{R}_cap{CAP},wide_k10_r3}.parquet with columns
   s1_row, j_row, dense_cos, dense_rank (forward raw rank, null = reverse only), new_rank, rev_rank, source, country[, y]"""
import json, os, resource, sys
import numpy as np, polars as pl
sys.path.insert(0, __file__.rsplit("/", 1)[0])
from a01_eval import ROOT, D, CACHE, COUNTRIES
from a03_rev import pairs
RULES = [tuple(int(x) for x in a.split(":")) for a in sys.argv[1:]]
os.makedirs(f"{D}/pairs", exist_ok=True)
truth = pl.read_parquet(f"{ROOT}/v8_lab/ce/setup/out/truth_train.parquet").select(
    pl.col("i").alias("s1_row"), pl.col("j").alias("j_row"), pl.lit(1, pl.Int8).alias("y"))
recs = {s: pl.read_parquet(f"{CACHE[s]}/recs.parquet", columns=["row", "src", "country_key"]) for s in CACHE}
fw = lambda m: pl.col("new_rank").is_not_null() & (pl.col("new_rank") <= m)
rv = lambda r: pl.col("rev_rank").is_not_null() & (pl.col("rev_rank") <= r)
W = {}
for name in ("holdout", "head", "test"):
    split = "test" if name == "test" else "train"
    n_s1 = recs["test"].filter(pl.col("src") == 1).height if name == "test" else len(np.load(f"{D}/sets/{name}_rows.npy"))
    acc = {}
    for c in COUNTRIES[split]:
        cr = recs[split].filter((pl.col("country_key") == c) & (pl.col("src") == 1))["row"].to_numpy()
        rows = cr if name == "test" else np.intersect1d(np.load(f"{D}/sets/{name}_rows.npy"), cr)
        d = pairs(split, c, rows).with_columns(pl.coalesce("cos", "rcos").alias("dense_cos"), pl.lit(c).alias("country"))
        for m, r, cap in RULES:
            ro = d.filter(rv(r) & ~fw(m)).with_columns(pl.col("dense_cos").rank("ordinal", descending=True).over("i").alias("_rr"))
            acc.setdefault(f"m{m}_r{r}_cap{cap}", []).append(pl.concat([d.filter(fw(m)), ro.filter(pl.col("_rr") <= cap).drop("_rr")]))
        acc.setdefault("wide_k10_r3", []).append(d.filter((pl.col("rank").is_not_null() & (pl.col("rank") <= 10)) | rv(3)))
    for key, parts in acc.items():
        x = pl.concat(parts).select(pl.col("i").alias("s1_row"), pl.col("j").alias("j_row"), "dense_cos",
                                    pl.col("rank").alias("dense_rank"), "new_rank", "rev_rank",
                                    pl.col("src").alias("source"), "country")
        if name != "test":
            x = x.join(truth, on=["s1_row", "j_row"], how="left").with_columns(pl.col("y").fill_null(0))
        assert x.select(pl.struct("s1_row", "j_row").n_unique()).item() == x.height
        p = f"{D}/pairs/{name}_{key}.parquet"
        x.sort("s1_row", "source", "dense_cos", descending=[False, False, True]).write_parquet(p)
        W[f"{name}_{key}"] = {"path": p, "pairs": x.height, "per_s1": round(x.height / n_s1, 3),
                              "s1_with_pairs": x["s1_row"].n_unique()} | ({"trues": int(x["y"].sum())} if name != "test" else {})
        print(W[f"{name}_{key}"], flush=True)
W["peak_rss_gb"] = round(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 2**30, 2)
json.dump(W, open(f"{D}/pairs/w02.json", "w"), indent=1)
