"""Apply the France add lists (adds_all = cie->compagnie + mid/and groupe|france tail swaps) to ANY base file: a pair is appended only if its j is kept by no S1 in that base.
usage: apply.py BASE_TSV OUT_DIR [adds_all|adds_cie|adds_tailmid]"""
import os, sys, polars as pl
F = "/Users/adityarohilla/Documents/amazon-ml-2026/v8_lab/hunt/sweep/france/"
BASE, OUT = sys.argv[1], sys.argv[2]; L = sys.argv[3] if len(sys.argv) > 3 else "adds_all"
base = pl.read_csv(BASE, separator="\t", quote_char=None, infer_schema=False)
b = base.with_columns(pl.col("matched_entity_ids").fill_null("").str.split(",")).explode("matched_entity_ids").filter(pl.col("matched_entity_ids") != "").rename({"source1_entity_id": "s1", "matched_entity_ids": "m"})
adds = pl.read_parquet(F + L + ".parquet").filter(~pl.col("m").is_in(b["m"].implode()))
ex = pl.concat([b, adds]); assert ex["m"].n_unique() == ex.height
out = base.select("source1_entity_id").join(ex.group_by("s1", maintain_order=True).agg(pl.col("m").str.join(",")).rename({"s1": "source1_entity_id", "m": "matched_entity_ids"}), on="source1_entity_id", how="left").with_columns(pl.col("matched_entity_ids").fill_null(""))
assert out.height == base.height and (out["source1_entity_id"] == base["source1_entity_id"]).all()
os.makedirs(OUT, exist_ok=True); out.write_csv(OUT + "/matching_results.tsv", separator="\t", quote_style="never")
print("base pairs", b.height, "| added", adds.height, "->", ex.height)
