"""Fresh pairs for cross-encoder v2 (continued fine-tuning of v1).
Pool = train S1 outside v7's 500k sample, the 100k holdout and v1's 360k CE-TRAIN S1 (incl. its 20k CE-VAL).
N_FRESH random S1 of the pool (random draw keeps the US/India mix); N_VAL2 of them flagged is_val2 (VAL2).
Per S1: top-8 candidates by cached retrieval score plus every true pair among its candidates (v1's s03 recipe);
the fit filter (t01) is v1's: not VAL2, j-disjoint, rank <= 6 or y == 1.
j flags: j is the S2/S3 record of a STAGE4 active pair / of a HOLDOUT pair active in either regime (active | active_cap;
v1 used the uncapped flag only, which let 71 capped-only holdout pairs' records into CE training).
usage (ce venv): d01_data.py -> out/d01_pairs.parquet, out/d01_s1.parquet, out/d01.json"""
import json, os, resource, threading, time
from pathlib import Path
import numpy as np, polars as pl

ROOT = Path("/Users/adityarohilla/Documents/amazon-ml-2026")
CE = ROOT / "v8_lab/ce"; SET = CE / "setup/out"; OUT = Path(__file__).parent / "out"
TRAIN_CACHE = ROOT / "cache/58f0a60c5c264fdd"
N_FRESH, N_VAL2, TOPK = 300_000, 10_000, 8


def guard(limit=3.0e9):
    def run():
        while True:
            if resource.getrusage(resource.RUSAGE_SELF).ru_maxrss > limit:
                print("RSS guard: peak above 3.0 GB, aborting", flush=True); os._exit(3)
            time.sleep(0.5)
    threading.Thread(target=run, daemon=True).start()


guard()
R = {}
recs = pl.read_parquet(TRAIN_CACHE / "recs.parquet", columns=["row", "src", "country_key"])
s1 = recs.filter(pl.col("src") == 1).select(pl.col("row").alias("i"), pl.col("country_key").alias("country"))
del recs
truth = pl.read_parquet(SET / "truth_train.parquet")
s1 = s1.join(truth.group_by("i").len("n_true"), on="i", how="left").with_columns(pl.col("n_true").fill_null(0))
sample = pl.read_parquet(SET / "stage4_s1.parquet")["i"]
hold = pl.read_parquet(SET / "holdout_s1.parquet")["i"]
v1 = pl.read_parquet(SET / "ce_s1.parquet")["i"]
oof_i = pl.scan_parquet(ROOT / "code/business_entity_resolution/artifacts/v7/oof.parquet").select("i").unique().collect()["i"]
assert set(sample.to_list()) == set(oof_i.to_list())  # stage4_s1 == v7's 500k sample
used = pl.concat([sample, hold, v1])
assert used.n_unique() == used.len() == 960_000
pool = s1.filter(~pl.col("i").is_in(used.implode()))
fresh = pool.sample(N_FRESH, seed=21).sort("i")
val2 = fresh.sample(N_VAL2, seed=22)["i"]
fresh = fresh.with_columns(pl.col("i").is_in(val2.implode()).alias("is_val2"))
for name, s in (("v7_sample", oof_i), ("holdout", hold), ("v1_ce_train", v1)):
    assert not fresh["i"].is_in(s.implode()).any(), name
R["s1"] = {"train_s1": s1.height, "pool": pool.height, "pool_by_country": dict(pool.group_by("country").len().iter_rows()),
           "fresh": fresh.height, "fresh_by_country": dict(fresh.group_by("country").len().iter_rows()), "val2": N_VAL2,
           "overlap_v7_sample": 0, "overlap_holdout": 0, "overlap_v1_ce_train": 0}
print(R["s1"], flush=True)

P = (pl.scan_parquet(TRAIN_CACHE / "pairs.parquet").select("i", "j", "score")
     .filter(pl.col("i").is_in(fresh["i"].implode())).collect(engine="streaming"))
P = (P.join(truth.with_columns(pl.lit(1, dtype=pl.Int8).alias("y")), on=["i", "j"], how="left").with_columns(pl.col("y").fill_null(0))
     .with_columns(pl.col("score").rank("ordinal", descending=True).over("i").cast(pl.UInt16).alias("rank")))
n_cand, true_in_cands = P.height, int(P["y"].sum())
s4j = pl.scan_parquet(SET / "stage4_train.parquet").filter("active").select("j").unique().collect()["j"]
hoj = pl.scan_parquet(SET / "holdout_pairs.parquet").filter(pl.col("active") | pl.col("active_cap")).select("j").unique().collect()["j"]
D = (P.filter((pl.col("rank") <= TOPK) | (pl.col("y") == 1))
     .join(fresh.select("i", "country", "is_val2"), on="i", how="left")
     .with_columns(pl.col("j").is_in(s4j.implode()).alias("j_in_stage4_active"), pl.col("j").is_in(hoj.implode()).alias("j_in_holdout_active"))
     .sort("i", "rank"))
del P
D.write_parquet(OUT / "d01_pairs.parquet")
fresh.write_parquet(OUT / "d01_s1.parquet")
fit = D.filter(~pl.col("is_val2") & ~pl.col("j_in_stage4_active") & ~pl.col("j_in_holdout_active") & ((pl.col("rank") <= 6) | (pl.col("y") == 1)))
v = D.filter("is_val2")
R["pairs"] = {"all_cand_pairs": n_cand, "true_in_cands": true_in_cands, "top8_plus_true": D.height, "positives": int(D["y"].sum()),
              "fit": fit.height, "fit_pos": int(fit["y"].sum()), "fit_s1": fit["i"].n_unique(),
              "fit_by_country": dict(fit.group_by("country").len().iter_rows()),
              "val2_pairs": v.height, "val2_pos": int(v["y"].sum()),
              "val2_fit_like": v.filter(~pl.col("j_in_stage4_active") & ~pl.col("j_in_holdout_active") & ((pl.col("rank") <= 6) | (pl.col("y") == 1))).height,
              "stage4_active_j": s4j.len(), "holdout_active_j_either_regime": hoj.len()}
R["peak_rss_gb"] = round(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1e9, 2)
(OUT / "d01.json").write_text(json.dumps(R, indent=1))
print(json.dumps(R, indent=1))
