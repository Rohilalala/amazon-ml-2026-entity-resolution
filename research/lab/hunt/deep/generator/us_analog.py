import polars as pl, re
from rapidfuzz import fuzz
pl.Config.set_tbl_rows(60); pl.Config.set_tbl_width_chars(220); pl.Config.set_float_precision(3)
R = "/Users/adityarohilla/Documents/amazon-ml-2026/"; O = R + "v8_lab/hunt/deep/generator/"
h = pl.read_parquet(O + "holdout_typed.parquet").with_columns(pl.col("keep").fill_null(False))
x = h.filter(pl.col("nt").is_in(["swaplast", "append"])).with_columns(w=pl.col("nd").str.split(">").list.last().str.split(" ").list.last())
x = x.filter(pl.col("w").is_in(["center", "services", "service", "partners", "council", "federation", "group", "enterprises", "associates", "industries", "holdings", "trading", "ventures"]) & (pl.col("at") == "same"))
recs = pl.scan_parquet(R + "cache/58f0a60c5c264fdd/recs.parquet").select("row", "name").filter(pl.col("row").is_in(pl.concat([x["i"], x["j"]]).implode())).collect()
x = x.join(recs.rename({"row": "i", "name": "n1"}), on="i").join(recs.rename({"row": "j", "name": "n2"}), on="j")
st = lambda a: re.sub(r"\d+\w*", " ", a or "")
x = x.with_columns(st_sim=pl.Series([fuzz.token_set_ratio(st(a), st(b)) for a, b in zip(x["a1"].to_list(), x["a2"].to_list())]),
                   tail_end=pl.Series([(b or "").endswith(" " + w) for b, w in zip(x["n2"].to_list(), x["w"].to_list())]))
x = x.with_columns(st=pl.when(pl.col("st_sim") >= 85).then(pl.lit("st_eq")).otherwise(pl.lit("st_ne")))
print(x.group_by("w", "nt", "st", "tail_end").agg(n=pl.len(), ytrue=pl.col("y").mean(), keep=pl.col("keep").mean(), unk=(~pl.col("keep")).sum(), y_unk=pl.col("y").filter(~pl.col("keep")).mean()).filter(pl.col("n") >= 30).sort("w", "nt", "st", "tail_end"))
