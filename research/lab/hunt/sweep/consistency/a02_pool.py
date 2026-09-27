"""Holdout: copies of a decided j anywhere in the train pool (S2/S3). Where do they go, and are they true for the S1?"""
import sys, numpy as np, polars as pl
from pathlib import Path
D = Path(__file__).parent; R = Path("/Users/adityarohilla/Documents/amazon-ml-2026")
sys.path.insert(0, str(R / "code/business_entity_resolution"))
from src.metric import f05_counts
if not (D / "train_keys.parquet").exists():
    (pl.scan_parquet(R / "cache/58f0a60c5c264fdd/recs.parquet").select("row", "src", "country_key", "name", "core", "addr")
     .join(pl.scan_parquet(R / "v8_lab/ce/setup/out/text_train.parquet"), on="row")
     .select("row", "src", "country_key", k_text=pl.col("text").hash(), k_norm=pl.concat_str("name", "addr", separator="|").hash(),
             k_core=pl.concat_str("core", "addr", separator="|").hash(), e_addr=pl.col("addr").str.len_chars() == 0)
     .collect(engine="streaming").write_parquet(D / "train_keys.parquet"))
TK = pl.read_parquet(D / "train_keys.parquet").filter(pl.col("src") > 1)
C = pl.read_parquet(D / "hold_cand.parquet")
F = C.filter("dec").select("i", "j", "y")
S = pl.read_parquet(R / "v8_lab/ce/setup/out/holdout_s1.parquet").sort("i")
T = pl.scan_parquet(R / "v8_lab/ce/setup/out/truth_train.parquet").filter(pl.col("i").is_in(S["i"].implode())).collect()
owner_all = F.group_by("j").agg(pl.col("i").first().alias("own"), pl.len().alias("n_own"))
for key in sys.argv[1:] or ["k_text", "k_norm"]:
    kj = TK.select(j="row", key=key, e="e_addr")
    fk = F.join(kj, on="j")
    cop = fk.join(kj.rename({"j": "j2"}), on="key").filter(pl.col("j2") != pl.col("j")).select("i", "j", "y", "j2", "e")
    # copy j2 not already decided for the same i
    cop = cop.join(F.select("i", pl.col("j").alias("j2"), same=pl.lit(True)), on=["i", "j2"], how="left").filter(pl.col("same").is_null())
    cop = cop.join(owner_all.rename({"j": "j2"}), on="j2", how="left")
    cop = cop.join(T.rename({"j": "j2"}).with_columns(y2=pl.lit(1)), on=["i", "j2"], how="left").with_columns(pl.col("y2").fill_null(0))
    cop = cop.join(C.select("i", pl.col("j").alias("j2"), incand=pl.lit(True)), on=["i", "j2"], how="left").with_columns(pl.col("incand").fill_null(False))
    print(f"\n== {key}: decided pairs {F.height}; missing copies {cop.height} (unique i,j2 {cop.unique(['i','j2']).height})")
    print(cop.group_by(owned=pl.col("own").is_not_null(), incand="incand", src_y="y").agg(n=pl.len(), true_j2=pl.col("y2").sum(), eaddr=pl.col("e").sum()).sort("owned", "incand", "src_y"))
    cop.write_parquet(D / f"hold_copies_{key}.parquet")
