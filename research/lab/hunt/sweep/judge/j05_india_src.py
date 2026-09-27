"""For each India copy add (s1, m) in consistency/add_copy_india.parquet: the base record(s) the S1 keeps that m copies (same k_core).
apply_final.py adds (s1, m) only while the base still keeps one of these sources for that S1 (the rule depends on the base's keeps).
usage: j05_india_src.py <base matching_results.tsv>"""
import sys, polars as pl
from pathlib import Path
R = Path("/Users/adityarohilla/Documents/amazon-ml-2026"); D = Path(__file__).parent
rec = (pl.scan_parquet(R / "cache/3ee42f05d06efbcb/recs.parquet")
       .select("entity_id", "src", key=pl.concat_str("core", "addr", separator="|").hash(), e=pl.col("addr").str.len_chars() == 0)
       .filter((pl.col("src") > 1) & ~pl.col("e")).select("entity_id", "key").collect(engine="streaming"))
A = pl.read_parquet(R / "v8_lab/hunt/sweep/consistency/add_copy_india.parquet")
B = (pl.read_csv(sys.argv[1], separator="\t", quote_char=None, infer_schema=False)
     .with_columns(pl.col("matched_entity_ids").fill_null("").str.split(",")).explode("matched_entity_ids")
     .filter(pl.col("matched_entity_ids") != "").rename({"source1_entity_id": "s1", "matched_entity_ids": "src"}))
S = (A.join(rec.rename({"entity_id": "m"}), on="m").join(B, on="s1").join(rec.rename({"entity_id": "src", "key": "k2"}), on="src")
     .filter(pl.col("key") == pl.col("k2")).group_by("s1", "m").agg(pl.col("src").unique().sort()))
assert S.height == A.height, (S.height, A.height)
S.write_parquet(D / "final_adds_src.parquet"); print("india adds with sources", S.height, "sources", S["src"].list.len().sum())
