"""Holdout emulation of the withheld-sibling effect on address-less exact-name copies."""
import polars as pl
R = "/Users/adityarohilla/Documents/amazon-ml-2026/"; O = R + "v8_lab/hunt/deep/usin/out/"
recs = pl.read_parquet(R + "cache/58f0a60c5c264fdd/recs.parquet", columns=["row", "src", "core", "addr", "country_key"])
s1c = recs.filter(pl.col("src") == 1).group_by("country_key", "core").agg(ns1=pl.len())
aempty = recs.select(pl.col("row").alias("j"), a2empty=pl.col("addr").is_null() | (pl.col("addr").str.strip_chars() == ""))
core = recs.select(pl.col("row").alias("i"), pl.col("core").alias("c1"), pl.col("country_key").alias("country"))
del recs
S = pl.read_parquet(O + "holdout_s1_f.parquet").join(core.drop("country"), on="i").join(s1c.rename({"core": "c1", "country_key": "country"}), on=["country", "c1"], how="left")
M = pl.read_parquet(O + "holdout_missexact.parquet").join(aempty, on="j").filter(pl.col("a2empty") & ~pl.col("h1miss"))
orph = pl.read_parquet(O + "holdout_orph.parquet")
b = 0.25
def f05(n_true, k, tp):
    return pl.when((n_true == 0) & (k == 0)).then(1.0).when((k == 0) | (n_true == 0) | (tp == 0)).then(0.0).otherwise((1 + b) * tp / (b * n_true + k))
agg = M.group_by("i").agg(ae_n=pl.len(), ae_true=pl.col("y").sum().cast(pl.Int64), ae_keep=pl.col("keep").sum().cast(pl.Int64), ae_tpk=((pl.col("y") == 1) & pl.col("keep")).sum().cast(pl.Int64))
S = S.join(agg, on="i", how="left").fill_null(0).join(orph.select("i", "n_orph"), on="i", how="left")
S = S.with_columns(pl.col("k").cast(pl.Int64), pl.col("tp").cast(pl.Int64), pl.col("n_true").cast(pl.Int64))
# A: ns1==1, kept >=1 addressless exact, orphan evidence -> drop kept addressless exact
A = S.filter((pl.col("ns1") == 1) & (pl.col("ae_keep") >= 1))
A = A.with_columns(f_drop=f05(pl.col("n_true"), pl.col("k") - pl.col("ae_keep"), pl.col("tp") - pl.col("ae_tpk")))
print("A (ns1=1, holdout)", A.group_by("country", pl.col("n_orph").fill_null(0) >= 1).agg(n=pl.len(), f=pl.col("f").mean(), f_drop=pl.col("f_drop").mean()).sort("country"))
# B: ns1==2 emulate sibling withheld: model keeps all addressless exact candidates of i
B = S.filter((pl.col("ns1") == 2) & (pl.col("ae_n") >= 1))
B = B.with_columns(k_w=pl.col("k") - pl.col("ae_keep") + pl.col("ae_n"), tp_w=pl.col("tp") - pl.col("ae_tpk") + pl.col("ae_true"))
B = B.with_columns(f_w=f05(pl.col("n_true"), pl.col("k_w"), pl.col("tp_w")), f_drop=f05(pl.col("n_true"), pl.col("k") - pl.col("ae_keep"), pl.col("tp") - pl.col("ae_tpk")))
print("B (ns1=2, sibling-withheld emulation)", B.group_by("country").agg(n=pl.len(), q=pl.col("ae_true").sum() / pl.col("ae_n").sum(), f_now=pl.col("f").mean(), f_withheld=pl.col("f_w").mean(), f_drop=pl.col("f_drop").mean()))
