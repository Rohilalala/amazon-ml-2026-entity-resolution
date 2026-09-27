"""Orphan family size: unassigned same-core pool records (different house from S1) grouped by their own house number."""
import polars as pl
R = "/Users/adityarohilla/Documents/amazon-ml-2026/"; O = R + "v8_lab/hunt/deep/usin/out/"
D = pl.read_parquet(O + "missexact_kept.parquet").filter(pl.col("a2empty") & (pl.col("ns1") == 1))
for side, cache in [("holdout", "58f0a60c5c264fdd"), ("test", "3ee42f05d06efbcb")]:
    S = D.filter(pl.col("side") == side).select("country", "i").unique()
    recs = pl.read_parquet(R + "cache/%s/recs.parquet" % cache, columns=["row", "src", "core", "house", "country_key"])
    S = S.join(recs.select(pl.col("row").alias("i"), pl.col("core").alias("c1"), pl.col("house").alias("h1")), on="i")
    pool = recs.filter(pl.col("src") != 1).select(pl.col("row").alias("j"), pl.col("core").alias("c1"), pl.col("country_key").alias("country"), pl.col("house").alias("h2"))
    del recs
    P = S.join(pool, on=["country", "c1"])
    if side == "holdout":
        own = pl.read_parquet(R + "v8_lab/ce/setup/out/truth_train.parquet")
    else:
        own = pl.read_parquet(O + "test_cats.parquet", columns=["i", "j", "keep"]).filter(pl.col("keep")).select("i", "j")
    anyj = own.select("j").unique().with_columns(pl.lit(True).alias("assigned"))
    P = P.join(anyj, on="j", how="left").filter(pl.col("assigned").is_null() & pl.col("h2").is_not_null() & (pl.col("h2") != "") & (pl.col("h2") != pl.col("h1")))
    fam = P.group_by("country", "i", "h2").len().group_by("country", "i").agg(maxfam=pl.col("len").max(), nh=pl.len())
    S = S.join(fam, on=["country", "i"], how="left").with_columns(pl.col("maxfam").fill_null(0))
    S.select("country", "i", "maxfam", "nh").write_parquet(O + side + "_fam.parquet")
    print(side, S.group_by("country", pl.col("maxfam").clip(0, 3)).len().sort("country", "maxfam").with_columns((pl.col("len") / pl.col("len").sum().over("country")).round(4).alias("frac")))
