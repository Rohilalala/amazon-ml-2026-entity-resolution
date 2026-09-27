"""Checks on the written lists: disjoint from the sparse candidate set, S1/j sources and countries consistent with the
cache records, holdout/head labels equal the truth file, head S1 disjoint from v7 sample + holdout."""
import json, numpy as np, polars as pl
ROOT = "/Users/adityarohilla/Documents/amazon-ml-2026"; D = f"{ROOT}/v8_lab/recall/retrieve/out"
CACHE = {"train": f"{ROOT}/cache/58f0a60c5c264fdd", "test": f"{ROOT}/cache/3ee42f05d06efbcb"}
R = {}
for name, split in (("holdout", "train"), ("head", "train"), ("test", "test")):
    recs = pl.read_parquet(f"{CACHE[split]}/recs.parquet", columns=["row", "src", "country_key"])
    for key in ("m1_r1_cap10", "m2_r1_cap10", "wide_k10_r3"):
        x = pl.read_parquet(f"{D}/pairs/{name}_{key}.parquet")
        cand = pl.scan_parquet(f"{CACHE[split]}/pairs.parquet").select(pl.col("i").alias("s1_row"), pl.col("j").alias("j_row")) \
            .filter(pl.col("s1_row").is_in(x["s1_row"].unique().implode())).collect()
        a = x.join(recs.select(pl.col("row").alias("s1_row"), pl.col("src").alias("si"), pl.col("country_key").alias("ci")), on="s1_row") \
             .join(recs.select(pl.col("row").alias("j_row"), pl.col("src").alias("sj"), pl.col("country_key").alias("cj")), on="j_row")
        R[f"{name}_{key}"] = {"rows": x.height, "in_sparse_cand": x.join(cand, on=["s1_row", "j_row"], how="semi").height,
                              "s1_src_ok": bool((a["si"] == 1).all()), "j_src_ok": bool((a["sj"].cast(pl.UInt8) == a["source"]).all()),
                              "country_ok": bool(((a["ci"] == a["cj"]) & (a["ci"] == a["country"])).all()), "joined": a.height}
        print(name, key, R[f"{name}_{key}"], flush=True)
truth = pl.read_parquet(f"{ROOT}/v8_lab/ce/setup/out/truth_train.parquet")
for name in ("holdout", "head"):
    x = pl.read_parquet(f"{D}/pairs/{name}_wide_k10_r3.parquet")
    y2 = x.join(truth.select(pl.col("i").alias("s1_row"), pl.col("j").alias("j_row"), pl.lit(1, pl.Int8).alias("t")), on=["s1_row", "j_row"], how="left")
    R[f"{name}_labels_match"] = bool((y2["t"].fill_null(0) == y2["y"]).all())
sp = pl.read_parquet(f"{ROOT}/v8_lab/day1b/r3_regime/out/regime/split.parquet", columns=["row", "holdout", "sample"])
head = np.load(f"{D}/sets/head_rows.npy")
R["head_overlap_sample_or_holdout"] = sp.filter(pl.col("row").is_in(pl.Series(head).implode()) & (pl.col("holdout") | pl.col("sample"))).height
print(R); json.dump(R, open(f"{D}/v01_check.json", "w"), indent=1)
