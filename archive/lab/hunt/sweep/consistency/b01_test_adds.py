"""TEST: pool-copy adds. For each base pair (s1, j): S2/S3 records j2 with the same key as j (non-empty address), j2 kept by
no S1 in the base, (s1, j2) never scored (not a blocked pair, not a recall-head candidate), j2 claimed by one S1 only.
usage: b01_test_adds.py <base matching_results.tsv> <key k_core|k_norm> <out.parquet> [country, e.g. india]"""
import sys, polars as pl
from pathlib import Path
R = Path("/Users/adityarohilla/Documents/amazon-ml-2026")
base_tsv, key, out = sys.argv[1], sys.argv[2], Path(sys.argv[3])
rec = (pl.scan_parquet(R / "cache/3ee42f05d06efbcb/recs.parquet")
       .select("row", "entity_id", "src", "country_key", k_norm=pl.concat_str("name", "addr", separator="|").hash(),
               k_core=pl.concat_str("core", "addr", separator="|").hash(), e=pl.col("addr").str.len_chars() == 0)
       .collect(engine="streaming"))
ids = rec.select("row", "entity_id")
B = (pl.read_csv(base_tsv, separator="\t", quote_char=None, infer_schema=False)
     .with_columns(pl.col("matched_entity_ids").str.split(",")).explode("matched_entity_ids")
     .filter(pl.col("matched_entity_ids").is_not_null() & (pl.col("matched_entity_ids") != ""))
     .rename({"source1_entity_id": "s1", "matched_entity_ids": "m"}))
B = B.join(ids.rename({"entity_id": "s1", "row": "i"}), on="s1").join(ids.rename({"entity_id": "m", "row": "j"}), on="m")
kj = rec.filter((pl.col("src") > 1) & ~pl.col("e")).select(j="row", key=key)
cop = (B.select("i", "j").join(kj, on="j").join(kj.rename({"j": "j2"}), on="key").filter(pl.col("j2") != pl.col("j"))
       .select("i", "j2").unique())
cop = cop.filter(~pl.col("j2").is_in(B["j"].implode()))                       # kept by no S1 in the base
tp = (pl.scan_parquet(R / "v8_lab/ce/setup/out/test_pairs.parquet").select("i", pl.col("j").alias("j2"))
      .filter(pl.col("i").is_in(cop["i"].unique().implode())).collect())
rh = pl.read_parquet(R / "v8_lab/recall/head/out/test_head_prob.parquet", columns=["s1_row", "j_row"]).rename({"s1_row": "i", "j_row": "j2"})
cop = cop.join(pl.concat([tp, rh]).unique(), on=["i", "j2"], how="anti")       # never scored for this S1
cop = cop.filter(pl.col("j2").is_unique())                                        # one claimant S1 only
cop = cop.join(rec.select(i="row", country="country_key"), on="i")
A = cop.join(ids.rename({"row": "i", "entity_id": "s1"}), on="i").join(ids.rename({"row": "j2", "entity_id": "m"}), on="j2")
if len(sys.argv) > 4: A = A.filter(pl.col("country") == sys.argv[4])
A.select("s1", "m").write_parquet(out)
print(key, "adds", A.height, A.group_by("country").len().sort("len", descending=True).rows())
