import polars as pl
R = "/Users/adityarohilla/Documents/amazon-ml-2026/"
def rd(split, s):
    return pl.read_csv(f"{R}student_resource/dataset/{split}/{split}_source{s}.tsv", separator="\t", quote_char=None, infer_schema=False).with_columns(pl.col("entity_id").str.strip_chars())
def kept_test(path=R+"v8_lab/hunt/final_adapt_all_sister/matching_results.tsv"):
    b = pl.read_csv(path, separator="\t", quote_char=None, infer_schema=False)
    b = b.with_columns(pl.col("matched_entity_ids").fill_null("").str.split(",")).explode("matched_entity_ids").rename({"source1_entity_id":"a","matched_entity_ids":"b"})
    s1 = rd("test",1).select(pl.col("entity_id").alias("a"), pl.col("business_name").alias("n1"), pl.col("business_address").alias("ad1"), pl.col("country").str.strip_chars().str.to_lowercase().alias("cty"))
    pool = pl.concat([rd("test",2), rd("test",3)]).select(pl.col("entity_id").alias("b"), pl.col("business_name").alias("n2"), pl.col("business_address").alias("ad2"))
    return b.join(s1, on="a", how="left").join(pool, on="b", how="left")
if __name__ == "__main__":
    k = kept_test(); k.write_parquet("kept_test.parquet")
    print(k.group_by("cty").agg(pl.len(), pl.col("a").n_unique(), (pl.col("b")=="").sum().alias("empty")))
