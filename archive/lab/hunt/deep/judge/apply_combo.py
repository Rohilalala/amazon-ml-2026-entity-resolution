"""Apply the judge's France rule sets (groupe tail adds, different-street drops; derived vs the 0.989 file by build.py) to ANY
base file: drops are removed where present; adds are appended only if the j is kept by no S1 in that base.
usage: apply_combo.py BASE_TSV OUT_DIR"""
import os, sys, polars as pl
from pairs import pairs
H = "/Users/adityarohilla/Documents/amazon-ml-2026/v8_lab/hunt/"
REF = H + "final_adapt_all_sister/matching_results.tsv"
ref = pairs(REF)
adds = pairs(H + "deep/generator/out_R1/matching_results.tsv").join(ref, on=["s1", "m"], how="anti")
drops = ref.join(pairs(H + "deep/fr_fp/out_street/matching_results.tsv"), on=["s1", "m"], how="anti")
BASE, OUT = sys.argv[1], sys.argv[2]
base = pl.read_csv(BASE, separator="\t", quote_char=None, infer_schema=False)
b = pairs(BASE)
kept = b.join(drops, on=["s1", "m"], how="anti")
add_ok = adds.filter(~pl.col("m").is_in(kept["m"].implode()))
ex = pl.concat([kept, add_ok])
assert ex["m"].n_unique() == ex.height
out = base.select("source1_entity_id").join(ex.group_by("s1", maintain_order=True).agg(pl.col("m").str.join(",")).rename({"s1": "source1_entity_id", "m": "matched_entity_ids"}), on="source1_entity_id", how="left").with_columns(pl.col("matched_entity_ids").fill_null(""))
assert out.height == base.height and (out["source1_entity_id"] == base["source1_entity_id"]).all()
os.makedirs(OUT, exist_ok=True); out.write_csv(OUT + "/matching_results.tsv", separator="\t", quote_style="never")
print("base pairs", b.height, "dropped", b.height - kept.height, "of", drops.height, "| added", add_ok.height, "of", adds.height, "->", ex.height)
