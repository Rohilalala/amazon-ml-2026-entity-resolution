"""Per-S1 error table of the best holdout configuration (v3 keeps + recall adds) and loss decomposition."""
import numpy as np, polars as pl
R = "/Users/adityarohilla/Documents/amazon-ml-2026"
K = pl.read_parquet("hold_keeps.parquet"); A = pl.read_parquet("hold_adds.parquet").filter("add")
S = pl.read_parquet(f"{R}/v8_lab/ce/setup/out/holdout_s1.parquet")
# owner / j-max
K = K.with_columns(pmax_j=pl.col("p").max().over("j"), prank=pl.col("p").rank("ordinal", descending=True).over("i"),
                   n_ge50=(pl.col("p") >= 0.5).sum().over("i"))
dec = pl.concat([K.filter("keep").select("i", "j", "y", src=pl.lit("keep")), A.select("i", "j", "y", src=pl.lit("add"))])
per = dec.group_by("i").agg(kept=pl.len(), tp=(pl.col("y") == 1).sum(), n_add=(pl.col("src") == "add").sum())
T = S.join(per, on="i", how="left").fill_null(0).with_columns(fp=pl.col("kept") - pl.col("tp"), fn=pl.col("n_true") - pl.col("tp"))
t = T.select(pl.col("tp", "kept", "n_true").cast(pl.Float64))
p = t["tp"] / t["kept"]; r = t["tp"] / t["n_true"]
f = np.where(T["tp"] > 0, 1.25 * p * r / (0.25 * p + r), 0.0); f = np.where(T["n_true"] == 0, (T["kept"] == 0).cast(float), f)
T = T.with_columns(f=pl.Series(np.nan_to_num(f)))
print("macro", T["f"].mean(), T.group_by("country").agg(pl.col("f").mean()))
T = T.with_columns(kind=pl.when((pl.col("fp") > 0) & (pl.col("fn") > 0)).then(pl.lit("fp+fn")).when(pl.col("fp") > 0).then(pl.lit("fp"))
                   .when(pl.col("fn") > 0).then(pl.lit("fn")).otherwise(pl.lit("ok")))
T = T.with_columns(sub=pl.when(pl.col("n_true") == 0).then(pl.lit("nt0")).when(pl.col("kept") == 0).then(pl.lit("empty")).otherwise(pl.lit("")))
print(T.group_by("kind", "sub").agg(n=pl.len(), loss=(1 - pl.col("f")).sum() / 100000).sort("loss", descending=True))
T.write_parquet("s1_table.parquet")
# FN pairs in candidates: reason
fnk = K.filter((pl.col("y") == 1) & ~pl.col("keep")).join(T.select("i", "kept"), on="i").with_columns(
    reason=pl.when(pl.col("p") < pl.col("pmax_j")).then(pl.lit("owner")).when(pl.col("p") >= 0.5).then(pl.lit("cutoff_hi")).when(pl.col("p") >= 0.1).then(pl.lit("cutoff_mid")).otherwise(pl.lit("lowp")))
print(fnk.group_by("reason", "country").agg(pl.len(), s1_empty=(pl.col("kept") == 0).sum()).sort("len", descending=True))
fnk.write_parquet("fn_cand.parquet")
fpk = K.filter((pl.col("y") == 0) & pl.col("keep")).join(T.select("i", "n_true", "kept", "tp"), on="i")
print("FP keeps", fpk.height, fpk.group_by("country").len(), "p quantiles", fpk["p"].quantile(0.1), fpk["p"].median())
print(fpk.with_columns(nt0=pl.col("n_true") == 0, only=pl.col("kept") == 1).group_by("nt0", "only").len())
fpk.write_parquet("fp_keep.parquet")
