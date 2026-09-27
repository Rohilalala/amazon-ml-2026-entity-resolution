import polars as pl
from cls import *
t = pl.read_parquet("kept_test.parquet").filter(pl.col("b")!="")
t = add_class(t); t.write_parquet("kept_test_k.parquet")
h = pl.read_parquet("ho_pairs.parquet").filter(pl.col("keep"))
h = add_class(h); h.write_parquet("ho_kept_k.parquet")
a = t.group_by("cty","k").len().with_columns((pl.col("len")/pl.col("len").sum().over("cty")).round(4).alias("share")).pivot(on="cty", index="k", values="share")
b = h.group_by("k").agg(pl.len().alias("ho_n"), (~pl.col("y")).sum().alias("ho_fp"), (1-pl.col("y").mean()).round(4).alias("ho_fp_rate"))
pl.Config.set_tbl_rows(40)
print(a.join(b, on="k", how="left").sort("france", descending=True))
