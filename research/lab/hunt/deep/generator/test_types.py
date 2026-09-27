import polars as pl, sys, time
sys.path.insert(0, "/Users/adityarohilla/Documents/amazon-ml-2026/v8_lab/hunt/deep/generator")
from rel import ntype, atype
R = "/Users/adityarohilla/Documents/amazon-ml-2026/"; O = R + "v8_lab/hunt/deep/generator/"
t0 = time.time()
recs = pl.scan_parquet(R + "cache/3ee42f05d06efbcb/recs.parquet").select("row", "entity_id", "src", "country_key", "core", "addr", "house")
s1 = recs.filter(pl.col("src") == 1).select("row", "entity_id", "country_key").collect()
# France all S1, plus a 60k US / 40k India sample for the analog check
fr = s1.filter(pl.col("country_key") == "france")
us = s1.filter(pl.col("country_key") == "us").sample(60000, seed=1)
ind = s1.filter(pl.col("country_key") == "india").sample(40000, seed=1)
sel = pl.concat([fr, us, ind])
print("S1 selected", sel.group_by("country_key").len(), flush=True)
tp = pl.scan_parquet(R + "v8_lab/ce/setup/out/test_pairs.parquet").select("i", "j", "active").join(sel.lazy().select(pl.col("row").alias("i")), on="i").collect()
p4 = pl.scan_parquet(R + "v8_lab/hunt/plan/s4ce/out_adapt/test_ACNRST_3s_adapt_prob.parquet").join(sel.lazy().select(pl.col("row").alias("i")), on="i").collect()
tp = tp.join(p4, on=["i", "j"], how="left")
fin = pl.read_csv(R + "v8_lab/hunt/final_adapt_all_sister/matching_results.tsv", separator="\t", quote_char=None, infer_schema=False)
fin = fin.join(sel.select(pl.col("entity_id").alias("source1_entity_id"), pl.col("row").alias("i")), on="source1_entity_id") \
         .with_columns(pl.col("matched_entity_ids").str.split(",")).explode("matched_entity_ids").filter(pl.col("matched_entity_ids").str.len_chars() > 0)
eid = recs.select("row", "entity_id").filter(pl.col("entity_id").is_in(fin["matched_entity_ids"].implode())).collect()
fin = fin.join(eid.rename({"entity_id": "matched_entity_ids", "row": "j"}), on="matched_entity_ids").select("i", pl.col("j").cast(pl.UInt32), keep=pl.lit(True))
tp = tp.join(fin, on=["i", "j"], how="full", coalesce=True).with_columns(pl.col("keep").fill_null(False), pl.col("active").fill_null(False), cand=pl.col("i").is_not_null())
print("pairs", tp.height, "kept", tp["keep"].sum(), time.time() - t0, flush=True)
need = pl.concat([tp["i"], tp["j"]]).unique()
rr = recs.filter(pl.col("row").is_in(need.implode())).collect()
tp = tp.join(rr.select(pl.col("row").alias("i"), pl.col("country_key").alias("country"), pl.col("core").alias("c1"), pl.col("addr").alias("a1"), pl.col("house").alias("h1")), on="i") \
       .join(rr.select(pl.col("row").alias("j"), pl.col("src").alias("jsrc"), pl.col("core").alias("c2"), pl.col("addr").alias("a2"), pl.col("house").alias("h2")), on="j")
print("joined", tp.height, time.time() - t0, flush=True)
nt, nd, at = [], [], []
for c1, c2, h1, h2, a2 in zip(tp["c1"].to_list(), tp["c2"].to_list(), tp["h1"].to_list(), tp["h2"].to_list(), tp["a2"].to_list()):
    t, d = ntype((c1 or "").split(), (c2 or "").split()); nt.append(t); nd.append(d); at.append(atype(h1, h2, a2))
tp = tp.with_columns(nt=pl.Series(nt), nd=pl.Series(nd), at=pl.Series(at))
tp.write_parquet(O + "test_typed.parquet")
print("done", time.time() - t0)
