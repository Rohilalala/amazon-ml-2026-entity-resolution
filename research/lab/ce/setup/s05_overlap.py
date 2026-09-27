"""S2/S3-record overlap between CE-TRAIN and the stage-4 sets. S1 records are disjoint by construction, but an S2/S3
record can be a candidate of a CE-TRAIN S1 and of a STAGE4-TRAIN / HOLDOUT S1. Adds flags to ce_train.parquet so the
training step can make the cross-encoder j-disjoint from the pairs it will later score for stage 4:
 j_in_stage4_active, j_in_holdout_active (j is the S2/S3 record of an ACTIVE pair of that set).
usage: s05_overlap.py -> rewrites out/ce_train.parquet with the two flags, out/s05.json"""
import json
from pathlib import Path
import polars as pl

OUT = Path(__file__).parent / "out"
ce = pl.read_parquet(OUT / "ce_train.parquet").drop("j_in_stage4_active", "j_in_holdout_active", strict=False)
s4 = pl.read_parquet(OUT / "stage4_train.parquet", columns=["j", "y", "active"]).filter("active")
ho = pl.read_parquet(OUT / "holdout_pairs.parquet", columns=["j", "y", "active"]).filter("active")
ce = ce.with_columns(pl.col("j").is_in(s4["j"].unique().implode()).alias("j_in_stage4_active"),
                     pl.col("j").is_in(ho["j"].unique().implode()).alias("j_in_holdout_active"))
ce.write_parquet(OUT / "ce_train.parquet")
fit = ce.filter(~pl.col("is_val"))
clean = fit.filter(~pl.col("j_in_stage4_active") & ~pl.col("j_in_holdout_active"))
jfit, jpos = fit["j"].unique(), fit.filter(pl.col("y") == 1)["j"].unique()
R = {"ce_fit_pairs": fit.height, "ce_fit_pairs_j_disjoint": clean.height, "ce_fit_pos_j_disjoint": int(clean["y"].sum()),
     "ce_fit_share_dropped_for_j_disjoint": round(1 - clean.height / fit.height, 4)}
for name, d in (("stage4_active", s4), ("holdout_active", ho)):
    R[name] = {"pairs": d.height, "share_j_seen_in_ce_fit": round(float(d["j"].is_in(jfit.implode()).mean()), 4),
               "share_j_is_ce_fit_positive": round(float(d["j"].is_in(jpos.implode()).mean()), 4),
               "true_pairs_share_j_seen_in_ce_fit": round(float(d.filter(pl.col("y") == 1)["j"].is_in(jfit.implode()).mean()), 4)}
(OUT / "s05.json").write_text(json.dumps(R, indent=1))
print(json.dumps(R, indent=1))
