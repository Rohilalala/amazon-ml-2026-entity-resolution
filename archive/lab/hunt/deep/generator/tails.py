import polars as pl
pl.Config.set_tbl_rows(80); pl.Config.set_tbl_width_chars(250); pl.Config.set_float_precision(3)
R = "/Users/adityarohilla/Documents/amazon-ml-2026/"; O = R + "v8_lab/hunt/deep/generator/"
t = pl.read_parquet(O + "test_typed.parquet").filter(pl.col("country") == "france")
recs = pl.scan_parquet(R + "cache/3ee42f05d06efbcb/recs.parquet").filter(pl.col("country_key") == "france").select("row", "entity_id").collect()
def kept(path, name):
    f = pl.read_csv(path, separator="\t", quote_char=None, infer_schema=False).with_columns(pl.col("matched_entity_ids").str.split(",")).explode("matched_entity_ids")
    f = f.join(recs.rename({"entity_id": "source1_entity_id", "row": "i"}), on="source1_entity_id").join(recs.rename({"entity_id": "matched_entity_ids", "row": "j"}), on="matched_entity_ids")
    return f.select("i", pl.col("j").cast(pl.UInt32), pl.lit(True).alias(name))
t = t.join(kept(R + "v8_lab/day1b/stack/r4/matching_results.tsv", "k_r4"), on=["i", "j"], how="left").join(kept(R + "v8_lab/hunt/base_U9/matching_results.tsv", "k_U9"), on=["i", "j"], how="left").with_columns(pl.col("k_r4", "k_U9").fill_null(False))
TRUE = ["fils", "cie", "services", "developpement", "groupe", "france", "associes"]
DEC = ["participations", "holding", "international", "distribution", "ei"]
x = t.filter(pl.col("nt").is_in(["swaplast", "append", "droplast_add"])).with_columns(w=pl.col("nd").str.split(">").list.last().str.split(" ").list.last()).filter(pl.col("w").is_in(TRUE + DEC))
x.write_parquet(O + "fr_tail_pairs.parquet")
print(x.group_by("w", "nt").agg(n=pl.len(), same=(pl.col("at") == "same").mean(), nohouse=(pl.col("at") == "nohouse").mean(), noaddr=(pl.col("at") == "noaddr").mean(), pm2=(pl.col("at") == "pm2").mean(), diff=(pl.col("at") == "diff").mean()).sort("nt", "w"))
print(x.filter(pl.col("at") == "same").group_by("w", "nt").agg(n=pl.len(), keep=pl.col("keep").mean(), k_r4=pl.col("k_r4").mean(), k_U9=pl.col("k_U9").mean(), p4hi=(pl.col("p4") > 0.5).mean(), p4null=pl.col("p4").is_null().mean()).sort("nt", "w"))
