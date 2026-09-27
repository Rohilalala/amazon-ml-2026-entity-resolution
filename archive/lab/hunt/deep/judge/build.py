"""Judge builds on top of the best file: groupe tail adds (generator/out_R1 rule), France different-street drops
(fr_fp/out_street rule), and both combined. Adds/drops are taken as set differences of the lens outputs vs the best file."""
import os, polars as pl
from pairs import pairs
H = "/Users/adityarohilla/Documents/amazon-ml-2026/v8_lab/hunt/"
BEST = H + "final_adapt_all_sister/matching_results.tsv"
base = pl.read_csv(BEST, separator="\t", quote_char=None, infer_schema=False)
best = pairs(BEST)
adds = pairs(H + "deep/generator/out_R1/matching_results.tsv").join(best, on=["s1", "m"], how="anti")
drops = best.join(pairs(H + "deep/fr_fp/out_street/matching_results.tsv"), on=["s1", "m"], how="anti")
assert adds.join(drops, on=["s1", "m"]).height == 0
def write(name, ex):
    assert ex["m"].n_unique() == ex.height  # no j kept by two S1
    out = base.select("source1_entity_id").join(ex.group_by("s1", maintain_order=True).agg(pl.col("m").str.join(",")).rename({"s1": "source1_entity_id", "m": "matched_entity_ids"}), on="source1_entity_id", how="left").with_columns(pl.col("matched_entity_ids").fill_null(""))
    assert out.height == base.height and (out["source1_entity_id"] == base["source1_entity_id"]).all()
    os.makedirs(name, exist_ok=True); out.write_csv(name + "/matching_results.tsv", separator="\t", quote_style="never")
    print(name, "pairs", best.height, "->", ex.height)
write("groupe_add", pl.concat([best, adds]))
write("street_drop", best.join(drops, on=["s1", "m"], how="anti"))
write("combo_groupe_street", pl.concat([best.join(drops, on=["s1", "m"], how="anti"), adds]))
print("adds", adds.height, "on S1", adds["s1"].n_unique(), "| drops", drops.height, "on S1", drops["s1"].n_unique())
