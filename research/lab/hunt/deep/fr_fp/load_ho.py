import polars as pl
R = "/Users/adityarohilla/Documents/amazon-ml-2026/"
hp = pl.read_parquet(R+"v8_lab/ce/setup/out/holdout_pairs.parquet", columns=["i","j","y"])
hk = pl.read_parquet(R+"v8_lab/ce/v2_build/out/holdout_keeps.parquet", columns=["ACNRST_3s_v2ce"])
h = hp.with_columns(hk["ACNRST_3s_v2ce"].alias("keep"), pl.col("y").cast(pl.Boolean))
print(h.select(pl.len(), pl.col("i").n_unique(), pl.col("keep").sum(), pl.col("y").sum(), (pl.col("y")&pl.col("keep")).sum().alias("tp")))
rows = pl.concat([h["i"], h["j"]]).unique()
recs = pl.read_parquet(R+"cache/58f0a60c5c264fdd/recs.parquet", columns=["row","entity_id","country_key"]).filter(pl.col("row").is_in(rows.implode()))
ids = recs["entity_id"]
raw = pl.concat([pl.scan_csv(f"{R}student_resource/dataset/train/train_source{s}.tsv", separator="\t", quote_char=None, infer_schema=False).with_columns(pl.col("entity_id").str.strip_chars()).filter(pl.col("entity_id").is_in(ids.implode())).select("entity_id","business_name","business_address") for s in (1,2,3)]).collect()
m = recs.join(raw, on="entity_id", how="left")
h = h.join(m.select(pl.col("row").alias("i"), pl.col("entity_id").alias("a"), pl.col("country_key").alias("cty"), pl.col("business_name").alias("n1"), pl.col("business_address").alias("ad1")), on="i", how="left") \
     .join(m.select(pl.col("row").alias("j"), pl.col("entity_id").alias("b"), pl.col("business_name").alias("n2"), pl.col("business_address").alias("ad2")), on="j", how="left")
h.write_parquet("ho_pairs.parquet"); print(h.head(3)); print(h.group_by("cty").len())
