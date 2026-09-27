import polars as pl
def pairs(f):
    b = pl.read_csv(f, separator="\t", quote_char=None, infer_schema=False)
    return b.with_columns(pl.col("matched_entity_ids").fill_null("").str.split(",")).explode("matched_entity_ids").filter(pl.col("matched_entity_ids") != "").rename({"source1_entity_id": "s1", "matched_entity_ids": "m"})
