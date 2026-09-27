"""TEST: base pairs whose exact copy (same name|addr, non-empty address, S2/S3) is kept by a DIFFERENT S1 (count only)."""
import polars as pl
from pathlib import Path
R = Path("/Users/adityarohilla/Documents/amazon-ml-2026")
rec = (pl.scan_parquet(R / "cache/3ee42f05d06efbcb/recs.parquet")
       .select("row", "entity_id", "src", "country_key", k=pl.concat_str("name", "addr", separator="|").hash(), e=pl.col("addr").str.len_chars() == 0)
       .collect(engine="streaming"))
B = (pl.read_csv(R / "v8_lab/hunt/final16b/matching_results.tsv", separator="\t", quote_char=None, infer_schema=False)
     .with_columns(pl.col("matched_entity_ids").str.split(",")).explode("matched_entity_ids").rename({"source1_entity_id": "s1", "matched_entity_ids": "m"})
     .filter(pl.col("m").is_not_null() & (pl.col("m") != "")))
B = B.join(rec.select(s1="entity_id", i="row", country="country_key"), on="s1").join(rec.filter(~pl.col("e")).select(m="entity_id", j="row", k="k"), on="m")
G = B.group_by("k").agg(n_s1=pl.col("i").n_unique(), n=pl.len(), country=pl.col("country").first()).filter(pl.col("n_s1") > 1)
print("copy groups kept by >1 S1:", G.height, "pairs", G["n"].sum(), G.group_by("country").len().rows())
