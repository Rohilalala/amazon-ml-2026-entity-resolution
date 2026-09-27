import polars as pl
R = "/Users/adityarohilla/Documents/amazon-ml-2026/"; O = R + "v8_lab/hunt/deep/usin/out/"
res = []
for side, cache, s1f in [("holdout", "58f0a60c5c264fdd", "holdout_s1"), ("test", "3ee42f05d06efbcb", "test_s1")]:
    d = pl.read_parquet(O + side + "_cats.parquet").filter(pl.col("keep") & (pl.col("hr") == "miss") & (pl.col("nr") == "exact"))
    recs = pl.read_parquet(R + "cache/%s/recs.parquet" % cache, columns=["row", "src", "core", "addr", "country_key"])
    # name counts per country: S1 records and pool records with the same core
    s1c = recs.filter(pl.col("src") == 1).group_by("country_key", "core").agg(ns1=pl.len())
    plc = recs.filter(pl.col("src") != 1).group_by("country_key", "core").agg(npool=pl.len())
    a = recs.select(pl.col("row").alias("j"), pl.col("addr").alias("a2"))
    d = d.join(a, on="j").join(s1c.rename({"core": "c1", "country_key": "country"}), on=["country", "c1"], how="left").join(plc.rename({"core": "c1", "country_key": "country"}), on=["country", "c1"], how="left")
    d = d.with_columns(h1miss=pl.col("h1").is_null() | (pl.col("h1") == ""), a2empty=pl.col("a2").is_null() | (pl.col("a2").str.strip_chars() == ""),
                       ns1b=pl.col("ns1").clip(1, 3), poolper=(pl.col("npool") / pl.col("ns1")).clip(0, 12).floor())
    d = d.with_columns(pl.lit(side).alias("side")); res.append(d.select("side", "country", "i", "j", "y" if side == "holdout" else pl.lit(None).cast(pl.Int8).alias("y"), "h1miss", "a2empty", "ns1", "npool", "ns1b", "poolper"))
    del recs
D = pl.concat(res); D.write_parquet(O + "missexact_kept.parquet")
ns = {("holdout", "us"): 59714, ("holdout", "india"): 40286, ("test", "us"): 663106, ("test", "india"): 809986}
g = D.group_by("side", "country", "h1miss", "a2empty", "ns1b").agg(n=pl.len(), prec=(pl.col("y") == 1).mean())
g = g.with_columns(ps=pl.struct("side", "country", "n").map_elements(lambda s: s["n"] / ns[(s["side"], s["country"])] * 1000, return_dtype=pl.Float64))
w = g.filter(pl.col("side") == "holdout").drop("side").join(g.filter(pl.col("side") == "test").drop("side", "prec"), on=["country", "h1miss", "a2empty", "ns1b"], suffix="_t", how="full", coalesce=True)
pl.Config.set_tbl_rows(40); pl.Config.set_tbl_width_chars(200)
print(w.with_columns(ratio=pl.col("ps_t") / pl.col("ps")).sort("country", "n_t", descending=True).with_columns(pl.col(pl.Float64).round(3)))
