"""Holdout: address-less exact-core candidates the decision rejects -- any slice with a high true rate?"""
import polars as pl
R = "/Users/adityarohilla/Documents/amazon-ml-2026/"; O = R + "v8_lab/hunt/deep/usin/out/"
recs = pl.read_parquet(R + "cache/58f0a60c5c264fdd/recs.parquet", columns=["row", "src", "core", "addr", "country_key"])
ae = recs.select(pl.col("row").alias("j"), a2empty=pl.col("addr").is_null() | (pl.col("addr").str.strip_chars() == ""))
ns1 = recs.filter(pl.col("src") == 1).group_by("country_key", "core").agg(ns1=pl.len())
# pool address-less records per core (how many address-less exact copies exist in pool for the name)
npae = recs.filter((pl.col("src") != 1) & (pl.col("addr").is_null() | (pl.col("addr").str.strip_chars() == ""))).group_by("country_key", "core").agg(npae=pl.len())
core = recs.select(pl.col("row").alias("i"), pl.col("core").alias("c1"))
del recs
M = pl.read_parquet(O + "holdout_missexact.parquet").join(ae, on="j").filter(pl.col("a2empty") & ~pl.col("h1miss")).join(core, on="i", how="left", suffix="_")
M = M.join(ns1.rename({"country_key": "country", "core": "c1"}), on=["country", "c1"], how="left").join(npae.rename({"country_key": "country", "core": "c1"}), on=["country", "c1"], how="left")
s = pl.read_parquet(O + "holdout_s1_f.parquet").select("i", "k", "n_true")
M = M.join(s, on="i").with_columns(ae_c=pl.len().over("i"), ae_k=pl.col("keep").sum().over("i"))
pl.Config.set_tbl_rows(60); pl.Config.set_tbl_width_chars(200)
x = M.filter(~pl.col("keep")).group_by("country", pl.col("ns1").clip(1, 3), pl.col("npae").clip(1, 4), pl.col("ae_k").clip(0, 1)).agg(n=pl.len(), true=pl.col("y").sum(), rate=pl.col("y").mean()).sort("country", "ns1", "npae", "ae_k")
print(x.with_columns(pl.col("rate").round(3)))
print(M.filter(~pl.col("keep")).group_by("country", pl.col("prob") > 0.3).agg(n=pl.len(), rate=pl.col("y").mean()))
