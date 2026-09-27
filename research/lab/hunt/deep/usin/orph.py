import polars as pl
R = "/Users/adityarohilla/Documents/amazon-ml-2026/"; O = R + "v8_lab/hunt/deep/usin/out/"
D = pl.read_parquet(O + "missexact_kept.parquet").filter(pl.col("a2empty") & (pl.col("ns1") == 1))
out = {}
for side, cache in [("holdout", "58f0a60c5c264fdd"), ("test", "3ee42f05d06efbcb")]:
    S = D.filter(pl.col("side") == side).select("country", "i").unique()
    recs = pl.read_parquet(R + "cache/%s/recs.parquet" % cache, columns=["row", "src", "core", "addr", "house", "country_key"])
    S = S.join(recs.select(pl.col("row").alias("i"), pl.col("core").alias("c1"), pl.col("house").alias("h1"), pl.col("addr").alias("a1")), on="i")
    pool = recs.filter((pl.col("src") != 1)).select(pl.col("row").alias("j"), pl.col("core").alias("c1"), pl.col("country_key").alias("country"), pl.col("house").alias("h2"), pl.col("addr").alias("a2"))
    del recs
    P = S.join(pool, on=["country", "c1"])
    if side == "holdout":
        tt = pl.read_parquet(R + "v8_lab/ce/setup/out/truth_train.parquet").with_columns(pl.lit(True).alias("own"))
    else:
        tt = pl.read_parquet(O + "test_cats.parquet", columns=["i", "j", "keep"]).filter(pl.col("keep")).select("i", "j", pl.lit(True).alias("own"))
    P = P.join(tt, on=["i", "j"], how="left").with_columns(pl.col("own").fill_null(False))
    if side == "test":  # also: assigned to some other S1 by the best file?
        oth = pl.read_parquet(O + "test_cats.parquet", columns=["j", "keep"]).filter(pl.col("keep")).select("j", pl.lit(True).alias("anykeep")).unique()
        P = P.join(oth, on="j", how="left").with_columns(pl.col("anykeep").fill_null(False))
    else:
        oth = tt.select("j", pl.lit(True).alias("anykeep")).unique(); P = P.join(oth, on="j", how="left").with_columns(pl.col("anykeep").fill_null(False))
    P = P.with_columns(fulladdr=pl.col("h2").is_not_null() & (pl.col("h2") != ""), samehouse=(pl.col("h2") == pl.col("h1")))
    # orphan-like: not owned by i, not assigned anywhere, has a house number different from i's
    g = P.group_by("country", "i").agg(n_orph=(~pl.col("own") & ~pl.col("anykeep") & pl.col("fulladdr") & ~pl.col("samehouse")).sum(),
                                        n_other=(~pl.col("own") & pl.col("fulladdr") & ~pl.col("samehouse")).sum())
    out[side] = g; g.write_parquet(O + side + "_orph.parquet")
    print(side, g.group_by("country").agg(n=pl.len(), orph_ge1=(pl.col("n_orph") >= 1).mean(), other_ge1=(pl.col("n_other") >= 1).mean()))
