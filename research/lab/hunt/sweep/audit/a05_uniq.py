"""Empty-address own candidates p in [0.3, 0.9) not kept: y-rate by how many S1 records in the pool share j's core name."""
import polars as pl, numpy as np, sys
R = "/Users/adityarohilla/Documents/amazon-ml-2026"
K = pl.read_parquet("hold_keeps.parquet").with_columns(pmax_j=pl.col("p").max().over("j"), nk=pl.col("keep").sum().over("i"))
C = pl.scan_parquet(f"{R}/cache/58f0a60c5c264fdd/recs.parquet").select("row", "src", "core", "addr", "name")
cnt = C.filter(pl.col("src") == 1).group_by("core").agg(n1=pl.len()).collect(engine="streaming")
cntall = C.group_by("core").agg(nall=pl.len()).collect(engine="streaming")
U = K.filter((pl.col("p") >= 0.3) & (pl.col("p") >= pl.col("pmax_j")))
J = C.filter(pl.col("row").is_in(pl.concat([U["j"], U["i"]]).unique().implode())).collect()
U = U.join(J.select(pl.col("row").alias("j"), jcore="core", jaddr="addr"), on="j").join(J.select(pl.col("row").alias("i"), icore="core"), on="i")
U = U.join(cnt, left_on="jcore", right_on="core", how="left").join(cntall, left_on="jcore", right_on="core", how="left").fill_null(0)
U = U.with_columns(jemp=pl.col("jaddr").fill_null("").str.strip_chars() == "", same=pl.col("jcore") == pl.col("icore"))
U.write_parquet("uniq_cand.parquet")
for kp in (False, True):
    print("keep", kp); print(U.filter((pl.col("keep") == kp)).group_by("jemp", "same", n1c=pl.col("n1").clip(0, 3)).agg(pl.len(), yr=pl.col("y").mean(), pm=pl.col("p").mean()).sort("jemp", "same", "n1c"))
