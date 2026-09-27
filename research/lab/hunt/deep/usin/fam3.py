"""All test US/India S1 with a unique core: does orphan-family presence raise the rate of kept address-less exact copies (absorption)?"""
import polars as pl
R = "/Users/adityarohilla/Documents/amazon-ml-2026/"; O = R + "v8_lab/hunt/deep/usin/out/"
recs = pl.read_parquet(R + "cache/3ee42f05d06efbcb/recs.parquet", columns=["row", "src", "core", "house", "addr", "country_key"])
s1 = recs.filter((pl.col("src") == 1) & pl.col("country_key").is_in(["us", "india"]))
ns1 = s1.group_by("country_key", "core").agg(ns1=pl.len())
S = s1.join(ns1, on=["country_key", "core"]).filter(pl.col("ns1") == 1).select(pl.col("row").alias("i"), pl.col("core").alias("c1"), pl.col("house").alias("h1"), pl.col("country_key").alias("country"))
pool = recs.filter((pl.col("src") != 1) & pl.col("country_key").is_in(["us", "india"])).select(pl.col("row").alias("j"), pl.col("core").alias("c1"), pl.col("country_key").alias("country"), pl.col("house").alias("h2"),
        a2empty=pl.col("addr").is_null() | (pl.col("addr").str.strip_chars() == ""))
del recs, s1
keep = pl.read_parquet(O + "test_cats.parquet", columns=["i", "j", "keep"]).filter(pl.col("keep")).select("i", "j")
P = S.join(pool, on=["country", "c1"]).join(keep.with_columns(pl.lit(True).alias("own")), on=["i", "j"], how="left").join(keep.select("j").with_columns(pl.lit(True).alias("assigned")), on="j", how="left")
fam = P.filter(pl.col("assigned").is_null() & pl.col("h2").is_not_null() & (pl.col("h2") != "") & (pl.col("h2") != pl.col("h1"))).group_by("i", "h2").len().group_by("i").agg(maxfam=pl.col("len").max())
ae = P.filter(pl.col("a2empty")).group_by("i").agg(ae_c=pl.len(), ae_k=pl.col("own").fill_null(False).sum(), ae_un=pl.col("assigned").is_null().sum())
S = S.join(fam, on="i", how="left").join(ae, on="i", how="left").fill_null(0)
S.write_parquet(O + "test_ns1_fam.parquet")
print(S.group_by("country", pl.col("maxfam").clip(0, 2)).agg(n=pl.len(), has_ae=(pl.col("ae_c") >= 1).mean(), ae_c=pl.col("ae_c").mean(), kept_ae=(pl.col("ae_k") >= 1).mean(), kae_mean=pl.col("ae_k").mean(), un_ae=(pl.col("ae_un") >= 1).mean()).sort("country", "maxfam"))
