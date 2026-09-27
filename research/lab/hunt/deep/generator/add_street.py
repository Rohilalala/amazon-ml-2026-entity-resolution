import polars as pl, re, sys
from rapidfuzz import fuzz
O = "/Users/adityarohilla/Documents/amazon-ml-2026/v8_lab/hunt/deep/generator/"
D = re.compile(r"\d+\w*")
for f in ["holdout_typed", "test_typed"]:
    x = pl.read_parquet(O + f + ".parquet")
    s = [int(fuzz.token_set_ratio(D.sub(" ", a or ""), D.sub(" ", b or ""))) if b else -1 for a, b in zip(x["a1"].to_list(), x["a2"].to_list())]
    x = x.with_columns(st_sim=pl.Series(s, dtype=pl.Int16)).with_columns(st=pl.when(pl.col("st_sim") < 0).then(pl.lit("na")).when(pl.col("st_sim") >= 85).then(pl.lit("eq")).otherwise(pl.lit("ne")))
    x.write_parquet(O + f + "_st.parquet"); print(f, x.height, flush=True)
