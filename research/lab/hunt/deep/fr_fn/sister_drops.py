"""List the France pairs the sister filter dropped (base ALL file minus best file) with their swapped words, p4 and FREQ."""
import polars as pl, collections, difflib
from ctype import ctype
R = "/Users/adityarohilla/Documents/amazon-ml-2026/"
def ex(f):
    b = pl.read_csv(f, separator="\t", quote_char=None, infer_schema=False).with_columns(pl.col("matched_entity_ids").str.split(",")).explode("matched_entity_ids")
    return b.filter(pl.col("matched_entity_ids").str.len_chars() > 0)
base, best = ex(R + "v8_lab/hunt/final_adapt_all/ALL/matching_results.tsv"), ex(R + "v8_lab/hunt/final_adapt_all_sister/matching_results.tsv")
d = base.join(best, on=["source1_entity_id", "matched_entity_ids"], how="anti")
recs = pl.read_parquet(R + "cache/3ee42f05d06efbcb/recs.parquet", columns=["row", "entity_id", "country_key", "name", "core", "addr"]).with_columns(pl.col("name", "core", "addr").fill_null(""))
d = d.join(recs.select(pl.col("entity_id").alias("source1_entity_id"), pl.col("row").alias("i"), "country_key", pl.col("name").alias("n1"), pl.col("core").alias("c1"), pl.col("addr").alias("a1")), on="source1_entity_id") \
     .join(recs.select(pl.col("entity_id").alias("matched_entity_ids"), pl.col("row").alias("j"), pl.col("name").alias("n2"), pl.col("core").alias("c2"), pl.col("addr").alias("a2")), on="matched_entity_ids")
p4 = pl.read_parquet(R + "v8_lab/hunt/plan/s4ce/out_adapt/test_ACNRST_3s_adapt_prob.parquet")
d = d.join(p4, on=["i", "j"], how="left")
fr = recs.filter(pl.col("country_key") == "france")
FREQ = collections.Counter(t for n in fr["core"].to_list() for t in n.split())
def sw(c1, c2):
    a, b = c1.split(), c2.split()
    x = [(p, q) for p, q in zip(a, b) if p != q]
    return x[0] if len(a) == len(b) and len(x) == 1 else (None, None)
rows = []
for r in d.iter_rows(named=True):
    a, b = sw(r["c1"], r["c2"])
    rows.append((a, b, FREQ.get(a, 0) if a else None, FREQ.get(b, 0) if b else None, difflib.SequenceMatcher(None, a, b).ratio() if a else None))
d = d.with_columns(pl.DataFrame(rows, schema=["wa", "wb", "fa", "fb", "sm"], orient="row"))
d.write_parquet("sister_drops.parquet")
print(d.height, d["country_key"].value_counts())
pl.Config.set_tbl_rows(60)
print(d.group_by("wa", "wb").agg(pl.len(), pl.col("fa").first(), pl.col("fb").first(), pl.col("p4").mean().round(3), pl.col("sm").first().round(2)).sort("fb").head(50))
