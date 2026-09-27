"""Classify unkept+unclaimed candidates: France test (with stage-4 p4) and labelled holdout (US/India, with y)."""
import polars as pl
from ctype import ctype
R = "/Users/adityarohilla/Documents/amazon-ml-2026/"
def cls(e, recs):
    e = e.join(recs.select(pl.col("row").alias("i"), pl.col("name").alias("n1"), pl.col("core").alias("c1"), pl.col("house").alias("h1"), pl.col("addr").alias("a1")), on="i") \
         .join(recs.select(pl.col("row").alias("j"), pl.col("name").alias("n2"), pl.col("core").alias("c2"), pl.col("house").alias("h2"), pl.col("addr").alias("a2")), on="j")
    return e.with_columns(pl.struct("n1", "c1", "n2", "c2").map_elements(lambda r: ctype(r["n1"], r["c1"], r["n2"], r["c2"]), return_dtype=pl.Utf8).alias("t"))
cols = ["row", "entity_id", "name", "core", "house", "addr"]
fill = [pl.col(c).fill_null("") for c in ["name", "core", "house", "addr"]]
# test France
tp = pl.read_parquet(R + "v8_lab/ce/setup/out/test_pairs.parquet")
p4 = pl.read_parquet(R + "v8_lab/hunt/plan/s4ce/out_adapt/test_ACNRST_3s_adapt_prob.parquet")
act = tp.filter("active").select("i", "j")
assert (act["i"] == p4["i"]).all() and (act["j"] == p4["j"]).all()
act = act.with_columns(p4["p4"])
kept = pl.read_parquet("test_best_pairs.parquet").select("i", "j")
fr = tp.filter(pl.col("country") == "france").select("i", "j", "prob").join(act, on=["i", "j"], how="left")
fr = fr.join(kept.with_columns(pl.lit(True).alias("k")), on=["i", "j"], how="left").filter(pl.col("k").is_null()).drop("k")
fr = fr.filter(~pl.col("j").is_in(kept["j"].unique().implode()))
del tp
recs = pl.read_parquet(R + "cache/3ee42f05d06efbcb/recs.parquet", columns=cols).with_columns(fill)
cls(fr, recs).write_parquet("fr_unclaimed.parquet"); del recs
# holdout
hp = pl.read_parquet(R + "v8_lab/ce/setup/out/holdout_pairs.parquet", columns=["i", "j", "prob", "country", "y"])
hk = pl.read_parquet(R + "v8_lab/ce/v2_build/out/holdout_keeps.parquet", columns=["i", "j", "ACNRST_3s_v2ce"])
hp = hp.join(hk, on=["i", "j"])
recs = pl.read_parquet(R + "cache/58f0a60c5c264fdd/recs.parquet", columns=cols).with_columns(fill)
gt = pl.read_csv(R + "student_resource/dataset/train/train_ground_truth.tsv", separator="\t", quote_char=None, infer_schema=False)
hs1 = recs.filter(pl.col("row").is_in(hp["i"].unique().implode()))["entity_id"]
other_true = gt.filter(~pl.col("source1_entity_id").is_in(hs1.implode())).select(pl.col("matched_entity_ids").str.split(",")).explode("matched_entity_ids")
other_rows = recs.filter(pl.col("entity_id").is_in(other_true["matched_entity_ids"].implode()))["row"]
kj = hp.filter("ACNRST_3s_v2ce")["j"].unique()
hu = hp.filter(~pl.col("ACNRST_3s_v2ce") & ~pl.col("j").is_in(kj.implode()) & ~pl.col("j").is_in(other_rows.implode()))
print("holdout pairs", hp.height, "unclaimed", hu.height, "S1", hp["i"].n_unique())
cls(hu.drop("ACNRST_3s_v2ce"), recs).write_parquet("ho_unclaimed.parquet")
