"""Singleton checks: S1 whose only decision is one weak keep (drop it?) and empty S1 with a strong unkept top candidate (keep it?)."""
import numpy as np, polars as pl
R = "/Users/adityarohilla/Documents/amazon-ml-2026"
K = pl.read_parquet("hold_keeps.parquet"); T = pl.read_parquet("s1_table.parquet")
K = K.with_columns(pmax_j=pl.col("p").max().over("j"))
one = K.filter("keep").join(T.filter((pl.col("kept") == 1) & (pl.col("n_add") == 0)).select("i", "n_true", "f"), on="i")
print("single-keep S1 by p bin: n, true, nt0 (drop gains 1 each), F lost if true")
print(one.group_by(pb=(pl.col("p") * 20).floor() / 20).agg(pl.len(), tru=(pl.col("y") == 1).sum(), nt0=(pl.col("n_true") == 0).sum(),
      floss=pl.col("f").filter(pl.col("y") == 1).sum()).with_columns(net=pl.col("nt0") - pl.col("floss")).sort("pb").head(8))
emp = T.filter(pl.col("kept") == 0).select("i", "n_true")
top = K.join(emp, on="i").filter(pl.col("p") >= pl.col("pmax_j")).sort("p", descending=True).group_by("i").first()
# keeping top: if true, F = 1.25/(0.25 n_true + 1) ; if n_true=0 lose 1
top = top.with_columns(fgain=pl.when(pl.col("y") == 1).then(1.25 / (0.25 * pl.col("n_true") + 1)).otherwise(0.0), nt0=pl.col("n_true") == 0)
print("empty S1 by top p bin: net gain of keeping the top candidate")
print(top.group_by(pb=(pl.col("p") * 20).floor() / 20).agg(pl.len(), tru=(pl.col("y") == 1).sum(), g=pl.col("fgain").sum(), nt0=pl.col("nt0").sum())
      .with_columns(net=pl.col("g") - pl.col("nt0")).sort("pb", descending=True).head(8))
