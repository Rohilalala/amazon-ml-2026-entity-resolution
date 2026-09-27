import polars as pl, re
from rapidfuzz import fuzz
pl.Config.set_tbl_rows(100); pl.Config.set_tbl_width_chars(250); pl.Config.set_float_precision(3)
R = "/Users/adityarohilla/Documents/amazon-ml-2026/"; O = R + "v8_lab/hunt/deep/generator/"
x = pl.read_parquet(O + "fr_tail_pairs.parquet")
recs = pl.scan_parquet(R + "cache/3ee42f05d06efbcb/recs.parquet").filter(pl.col("country_key") == "france").select("row", "name").collect()
x = x.join(recs.rename({"row": "i", "name": "n1"}), on="i").join(recs.rename({"row": "j", "name": "n2"}), on="j")
st = lambda a: re.sub(r"\d+\w*", " ", a or "")
x = x.with_columns(st_sim=pl.Series([fuzz.token_set_ratio(st(a), st(b)) for a, b in zip(x["a1"].to_list(), x["a2"].to_list())]),
                   tail_end=pl.Series([(b or "").endswith(" " + w) for b, w in zip(x["n2"].to_list(), x["w"].to_list())]))
x = x.with_columns(st=pl.when(pl.col("st_sim") >= 85).then(pl.lit("st_eq")).otherwise(pl.lit("st_ne")))
x.write_parquet(O + "fr_tail_pairs2.parquet")
print(x.filter(pl.col("at") == "same").group_by("nt", "w", "st", "tail_end").agg(n=pl.len(), keep=pl.col("keep").mean(), k_r4=pl.col("k_r4").mean()).filter(pl.col("n") >= 40).sort("nt", "w", "st", "tail_end"))
