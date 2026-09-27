"""Apply the judge's final lists to ANY base matching_results.tsv.
Drops (final_drops.parquet) are removed where present. An add (final_adds.parquet) is appended only if its S1 row exists in the
base and its m is kept by no S1 in the base. India copy adds (listed in final_adds_src.parquet) also need the base to still keep,
for that S1, a record the add copies. Row order of the base is preserved.
usage: apply_final.py BASE_TSV OUT_DIR"""
import os, sys, polars as pl
J = os.path.dirname(os.path.abspath(__file__)) + "/"
BASE, OUT = sys.argv[1], sys.argv[2]
base = pl.read_csv(BASE, separator="\t", quote_char=None, infer_schema=False)
b = (base.with_columns(pl.col("matched_entity_ids").fill_null("").str.split(",")).explode("matched_entity_ids")
     .filter(pl.col("matched_entity_ids") != "").rename({"source1_entity_id": "s1", "matched_entity_ids": "m"}).select("s1", "m"))
assert b["m"].is_unique().all(), "base keeps an m twice"
drops = pl.read_parquet(J + "final_drops.parquet").select("s1", "m")
adds = pl.read_parquet(J + "final_adds.parquet").select("s1", "m")
assert adds.join(drops, on=["s1", "m"]).height == 0
adds = adds.filter(pl.col("s1").is_in(base["source1_entity_id"].implode()) & ~pl.col("m").is_in(b["m"].implode()))
src = pl.read_parquet(J + "final_adds_src.parquet").explode("src")
held = src.join(b.rename({"m": "src"}), on=["s1", "src"]).select("s1", "m").unique()
adds = pl.concat([adds.join(src.select("s1", "m").unique(), on=["s1", "m"], how="anti"), adds.join(held, on=["s1", "m"])])
kept = b.join(drops, on=["s1", "m"], how="anti")
ex = pl.concat([kept, adds]); assert ex["m"].is_unique().all(), "an m is kept by two S1"
out = (base.select("source1_entity_id").join(ex.group_by("s1", maintain_order=True).agg(pl.col("m").str.join(","))
       .rename({"s1": "source1_entity_id", "m": "matched_entity_ids"}), on="source1_entity_id", how="left", maintain_order="left")
       .with_columns(pl.col("matched_entity_ids").fill_null("")))
assert out.height == base.height and (out["source1_entity_id"] == base["source1_entity_id"]).all()
os.makedirs(OUT, exist_ok=True); out.write_csv(OUT + "/matching_results.tsv", separator="\t", quote_style="never")
print("base pairs", b.height, "| dropped", b.height - kept.height, "| added", adds.height, "->", ex.height)
