"""Per-S1 kept counts by copy type: best file (test, per country) vs train truth (US/India, 50k S1 sample)."""
import polars as pl
from ctype import ctype
R = "/Users/adityarohilla/Documents/amazon-ml-2026/"
BEST = R + "v8_lab/hunt/final_adapt_all_sister/matching_results.tsv"
def pairs_of(tsv, recs, sample=None):
    b = pl.read_csv(tsv, separator="\t", quote_char=None, infer_schema=False)
    if sample: b = b.sample(sample, seed=0)
    ids = recs.select("entity_id", "row")
    s1 = b.select(pl.col("source1_entity_id").alias("entity_id")).join(recs.select("entity_id", "row", "country_key", "name", "core"), on="entity_id")
    e = b.with_columns(pl.col("matched_entity_ids").str.split(",")).explode("matched_entity_ids").filter(pl.col("matched_entity_ids").str.len_chars() > 0)
    e = e.join(ids.rename({"entity_id": "source1_entity_id", "row": "i"}), on="source1_entity_id").join(ids.rename({"entity_id": "matched_entity_ids", "row": "j"}), on="matched_entity_ids")
    e = e.select("i", "j").join(recs.select(pl.col("row").alias("i"), "country_key", pl.col("name").alias("n1"), pl.col("core").alias("c1")), on="i") \
         .join(recs.select(pl.col("row").alias("j"), pl.col("name").alias("n2"), pl.col("core").alias("c2")), on="j")
    e = e.with_columns(pl.struct("n1", "c1", "n2", "c2").map_elements(lambda r: ctype(r["n1"] or "", r["c1"] or "", r["n2"] or "", r["c2"] or ""), return_dtype=pl.Utf8).alias("t"))
    return s1, e
out = []
for tag, tsv, cache, samp in [("test_best", BEST, "3ee42f05d06efbcb", None), ("train_truth", R + "student_resource/dataset/train/train_ground_truth.tsv", "58f0a60c5c264fdd", 50000)]:
    recs = pl.read_parquet(R + f"cache/{cache}/recs.parquet", columns=["row", "entity_id", "country_key", "name", "core"]).with_columns(pl.col("name").fill_null(""), pl.col("core").fill_null(""))
    s1, e = pairs_of(tsv, recs, samp)
    e.write_parquet(f"{tag}_pairs.parquet")
    n = s1.group_by("country_key").len().rename({"len": "nS1"})
    t = e.group_by("country_key", "t").len().join(n, on="country_key").with_columns((pl.col("len") / pl.col("nS1")).alias("per_s1")).with_columns(pl.lit(tag).alias("src"))
    out.append(t)
t = pl.concat(out).with_columns(pl.concat_str("src", pl.lit(":"), "country_key").alias("k")).pivot(on="k", index="t", values="per_s1").sort("t")
pl.Config.set_tbl_rows(50); pl.Config.set_tbl_width_chars(200)
print(t.with_columns(pl.selectors.numeric().round(4)))
t.write_csv("profile_per_s1.csv")
