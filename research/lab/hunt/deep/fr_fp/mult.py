import polars as pl
from cls import core
from load import rd
def key(n): return " ".join(sorted(core(n)))
def mult(split):
    s = rd(split, 1).select("entity_id", "business_name", pl.col("country").str.strip_chars().str.to_lowercase().alias("cty"))
    s = s.with_columns(pl.col("business_name").map_elements(key, return_dtype=pl.String).alias("key"))
    return s.with_columns(pl.len().over("cty", "key").alias("M")).select(pl.col("entity_id").alias("a"), "key", "M")
if __name__ == "__main__":
    mult("test").write_parquet("mult_test.parquet"); mult("train").write_parquet("mult_train.parquet")
