import polars as pl
from cls import *; from sub import sub; from addr import hcls
R = "/Users/adityarohilla/Documents/amazon-ml-2026/"
def feats(df):
    df = add_class(df)
    df = df.with_columns(pl.when(pl.col("k")=="other").then(pl.struct("n1","n2").map_elements(lambda s: sub(s["n1"], s["n2"]), return_dtype=pl.String)).otherwise(pl.col("k")).alias("k"))
    return df.with_columns(pl.struct("ad1","ad2").map_elements(lambda s: hcls(s["ad1"], s["ad2"]), return_dtype=pl.String).alias("h"))
# holdout active
act = pl.read_parquet(R+"v8_lab/ce/setup/out/holdout_pairs.parquet", columns=["active"])["active"]
h = pl.read_parquet("ho_pairs.parquet").with_columns(act.alias("active")).filter(pl.col("active"))
feats(h).write_parquet("ho_act.parquet")
# france test active with p4 + kept flag from best file
tp = pl.read_parquet(R+"v8_lab/ce/setup/out/test_pairs.parquet", columns=["i","j","country","active"]).filter(pl.col("active"))
p = pl.read_parquet(R+"v8_lab/hunt/plan/s4ce/out_adapt/test_ACNRST_3s_adapt_prob.parquet")
tp = tp.with_columns(p["p4"]).filter(pl.col("country")=="france")
tx = pl.read_parquet(R+"v8_lab/ce/setup/out/text_test.parquet")
tx = tx.filter(pl.col("row").is_in(pl.concat([tp["i"], tp["j"]]).unique().implode()))
recs = pl.read_parquet(R+"cache/3ee42f05d06efbcb/recs.parquet", columns=["row","entity_id"])
tx = tx.join(recs, on="row").with_columns(pl.col("text").str.split_exact(" | ", 1).struct.rename_fields(["n","ad"])).unnest("text")
tp = tp.join(tx.select(pl.col("row").alias("i"), pl.col("entity_id").alias("a"), pl.col("n").alias("n1"), pl.col("ad").alias("ad1")), on="i") \
       .join(tx.select(pl.col("row").alias("j"), pl.col("entity_id").alias("b"), pl.col("n").alias("n2"), pl.col("ad").alias("ad2")), on="j")
kt = pl.read_parquet("kept_test.parquet").filter(pl.col("b")!="").select("a","b").with_columns(pl.lit(True).alias("keep"))
tp = tp.join(kt, on=["a","b"], how="left").with_columns(pl.col("keep").fill_null(False), pl.lit("france").alias("cty"))
feats(tp).write_parquet("fr_act.parquet"); print(tp.shape, tp["keep"].sum())
