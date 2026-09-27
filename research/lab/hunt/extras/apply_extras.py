"""Small positive-expected-value fixes on top of ANY base file: drops where present (sister swaps missed by v2, generator
source-cap excess copies, US state conflicts); adds (France 'groupe' tails on names without a legal form) only if the j is
kept by no S1 in the base and the S1 row exists. usage: apply_extras.py BASE_TSV OUT_DIR"""
import os, sys, polars as pl
V = "/Users/adityarohilla/Documents/amazon-ml-2026/v8_lab/hunt/extras/"
BASE, OUT = sys.argv[1], sys.argv[2]
base = pl.read_csv(BASE, separator="\t", quote_char=None, infer_schema=False)
b = base.select(pl.col("source1_entity_id").alias("s1"), pl.col("matched_entity_ids").fill_null("").str.split(",").alias("m")).explode("m").filter(pl.col("m") != "")
drops = pl.read_parquet(V + "drops.parquet").select("s1", "m"); adds = pl.read_parquet(V + "adds.parquet").select("s1", "m")
kept = b.join(drops, on=["s1", "m"], how="anti")
ok = adds.filter(~pl.col("m").is_in(kept["m"].implode()) & pl.col("s1").is_in(base["source1_entity_id"].implode())).unique(subset=["m"], keep="first")
ex = pl.concat([kept, ok]); assert ex["m"].n_unique() == ex.height
out = base.select("source1_entity_id").join(ex.group_by("s1", maintain_order=True).agg(pl.col("m").str.join(",")).rename({"s1": "source1_entity_id", "m": "matched_entity_ids"}), on="source1_entity_id", how="left").with_columns(pl.col("matched_entity_ids").fill_null(""))
assert out.height == base.height and (out["source1_entity_id"] == base["source1_entity_id"]).all()
os.makedirs(OUT, exist_ok=True); out.write_csv(OUT + "/matching_results.tsv", separator="\t", quote_style="never")
print("dropped", b.height - kept.height, "added", ok.height, "->", ex.height)
