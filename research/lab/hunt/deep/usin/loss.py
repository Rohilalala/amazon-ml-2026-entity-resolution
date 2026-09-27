import polars as pl
R = "/Users/adityarohilla/Documents/amazon-ml-2026/"; O = R + "v8_lab/hunt/deep/usin/out/"
h = pl.read_parquet(O + "holdout_cats.parquet")
s = pl.read_parquet(R + "v8_lab/ce/setup/out/holdout_s1.parquet")
a = h.group_by("i").agg(k=pl.col("keep").sum(), tp=((pl.col("y") == 1) & pl.col("keep")).sum())
s = s.join(a, on="i", how="left").fill_null(0)
b = 0.25
s = s.with_columns(f=pl.when((pl.col("n_true") == 0) & (pl.col("k") == 0)).then(1.0).when((pl.col("k") == 0) | (pl.col("n_true") == 0) | (pl.col("tp") == 0)).then(0.0)
                   .otherwise((1 + b) * pl.col("tp") / (b * pl.col("n_true") + pl.col("k"))))
print(s.group_by("country").agg(pl.col("f").mean(), loss=(1 - pl.col("f")).sum(), n=pl.len()))
s = s.with_columns(kind=pl.when((pl.col("n_true") == 0) & (pl.col("k") > 0)).then(pl.lit("empty_truth_kept"))
                   .when((pl.col("n_true") > 0) & (pl.col("k") == 0)).then(pl.lit("nothing_kept"))
                   .when(pl.col("tp") < pl.col("k")).then(pl.lit("has_fp")).when(pl.col("tp") < pl.col("n_true")).then(pl.lit("fn_only")).otherwise(pl.lit("perfect")))
print(s.group_by("country", "kind").agg(n=pl.len(), loss=(1 - pl.col("f")).sum()).sort("country", "loss", descending=True))
s.write_parquet(O + "holdout_s1_f.parquet")
# FN by category, restricted to fn_only S1
fn = h.filter((pl.col("y") == 1) & ~pl.col("keep")).join(s.select("i", "kind", "k", "n_true"), on="i")
print(fn.group_by("country", "kind", "hr", "nr").len().sort("len", descending=True).head(25))
fp = h.filter((pl.col("y") == 0) & pl.col("keep")).join(s.select("i", "kind"), on="i")
print("FP", fp.height); print(fp.group_by("country", "kind", "hr", "nr").len().sort("len", descending=True).head(15))
