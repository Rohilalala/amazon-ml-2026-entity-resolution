import polars as pl, sys, time
sys.path.insert(0, "/Users/adityarohilla/Documents/amazon-ml-2026/v8_lab/hunt/deep/generator")
from rel import ntype, atype
R = "/Users/adityarohilla/Documents/amazon-ml-2026/"; O = R + "v8_lab/hunt/deep/generator/"
t0 = time.time()
hp = pl.read_parquet(R + "v8_lab/ce/setup/out/holdout_pairs.parquet", columns=["i", "j", "y", "country", "active"])
hk = pl.read_parquet(R + "v8_lab/ce/v2_build/out/holdout_keeps.parquet", columns=["i", "j", "ACNRST_3s_v2ce"]).rename({"ACNRST_3s_v2ce": "keep"})
hp = hp.join(hk, on=["i", "j"], how="left").with_columns(pl.lit(True).alias("cand"))
recs = pl.scan_parquet(R + "cache/58f0a60c5c264fdd/recs.parquet").select("row", "entity_id", "src", "core", "addr", "house")
s1rows = hp.select("i", "country").unique()
s1 = recs.join(s1rows.lazy().rename({"i": "row"}), on="row").select("row", "entity_id", "country").collect()
gt = pl.read_csv(R + "student_resource/dataset/train/train_ground_truth.tsv", separator="\t", quote_char=None, infer_schema=False)
gt = gt.join(s1.rename({"entity_id": "source1_entity_id"}), on="source1_entity_id").with_columns(pl.col("matched_entity_ids").str.split(",")).explode("matched_entity_ids").filter(pl.col("matched_entity_ids").str.len_chars() > 0)
eid = recs.select("row", "entity_id").filter(pl.col("entity_id").is_in(gt["matched_entity_ids"].implode())).collect()
gt = gt.join(eid.rename({"entity_id": "matched_entity_ids", "row": "j"}), on="matched_entity_ids").select(pl.col("row").alias("i"), pl.col("j").cast(pl.UInt32), "country")
miss = gt.join(hp.select("i", "j"), on=["i", "j"], how="anti").with_columns(y=pl.lit(1, pl.Int8), active=pl.lit(False), keep=pl.lit(False), cand=pl.lit(False))
print("truth pairs", gt.height, "missing from candidates", miss.height, flush=True)
hp = pl.concat([hp, miss.select(hp.columns)])
need = pl.concat([hp["i"], hp["j"]]).unique()
rr = recs.filter(pl.col("row").is_in(need.implode())).collect()
hp = hp.join(rr.select(pl.col("row").alias("i"), pl.col("core").alias("c1"), pl.col("addr").alias("a1"), pl.col("house").alias("h1")), on="i") \
       .join(rr.select(pl.col("row").alias("j"), pl.col("src").alias("jsrc"), pl.col("core").alias("c2"), pl.col("addr").alias("a2"), pl.col("house").alias("h2")), on="j")
print("joined", hp.height, time.time() - t0, flush=True)
nt, nd, at = [], [], []
for c1, c2, h1, h2, a2 in zip(hp["c1"].to_list(), hp["c2"].to_list(), hp["h1"].to_list(), hp["h2"].to_list(), hp["a2"].to_list()):
    t, d = ntype((c1 or "").split(), (c2 or "").split()); nt.append(t); nd.append(d); at.append(atype(h1, h2, a2))
hp = hp.with_columns(nt=pl.Series(nt), nd=pl.Series(nd), at=pl.Series(at))
hp.write_parquet(O + "holdout_typed.parquet")
print("done", time.time() - t0)
