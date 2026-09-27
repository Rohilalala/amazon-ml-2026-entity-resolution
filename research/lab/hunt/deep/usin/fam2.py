import polars as pl
R = "/Users/adityarohilla/Documents/amazon-ml-2026/"; O = R + "v8_lab/hunt/deep/usin/out/"
D = pl.read_parquet(O + "missexact_kept.parquet").filter(pl.col("a2empty") & (pl.col("ns1") == 1))
t = pl.read_parquet(O + "test_cats.parquet", columns=["i", "keep"]).filter(pl.col("keep")).group_by("i").agg(ktot=pl.len())
for side in ["holdout", "test"]:
    F = pl.read_parquet(O + side + "_fam.parquet")
    k = D.filter(pl.col("side") == side).group_by("i").agg(kae=pl.len())
    F = F.join(k, on="i")
    if side == "test": F = F.join(t, on="i")
    else:
        s = pl.read_parquet(O + "holdout_s1_f.parquet").select("i", pl.col("k").alias("ktot"), "n_true"); F = F.join(s, on="i")
    print(side, F.group_by("country", pl.col("maxfam").clip(0, 2)).agg(n=pl.len(), kae=pl.col("kae").mean(), kae2=(pl.col("kae") >= 2).mean(), ktot=pl.col("ktot").mean(), only_ae=(pl.col("ktot") == pl.col("kae")).mean()).sort("country", "maxfam"))
