"""Classify ALL labelled holdout pairs (kept or not) by copy type, with swap-word frequencies, for analog precision checks."""
import polars as pl, collections, difflib
from ctype import ctype
R = "/Users/adityarohilla/Documents/amazon-ml-2026/"
hp = pl.read_parquet(R + "v8_lab/ce/setup/out/holdout_pairs.parquet", columns=["i", "j", "prob", "country", "y"]).join(
     pl.read_parquet(R + "v8_lab/ce/v2_build/out/holdout_keeps.parquet", columns=["i", "j", "ACNRST_3s_v2ce"]), on=["i", "j"]).rename({"ACNRST_3s_v2ce": "kept"})
recs = pl.read_parquet(R + "cache/58f0a60c5c264fdd/recs.parquet", columns=["row", "country_key", "name", "core", "house"]).with_columns(pl.col("name", "core", "house").fill_null(""))
FREQ = {c: collections.Counter(t for n in recs.filter(pl.col("country_key") == c)["core"].to_list() for t in n.split()) for c in ["us", "india"]}
hp = hp.join(recs.select(pl.col("row").alias("i"), pl.col("name").alias("n1"), pl.col("core").alias("c1"), pl.col("house").alias("h1")), on="i") \
       .join(recs.select(pl.col("row").alias("j"), pl.col("name").alias("n2"), pl.col("core").alias("c2"), pl.col("house").alias("h2")), on="j")
rows = []
for n1, c1, n2, c2, co in zip(hp["n1"], hp["c1"], hp["n2"], hp["c2"], hp["country"].cast(pl.Utf8)):
    t = ctype(n1, c1, n2, c2); a = b = None; fa = fb = sm = None
    x, y = c1.split(), c2.split()
    if len(x) == len(y):
        d = [(p, q) for p, q in zip(x, y) if p != q]
        if len(d) == 1: a, b = d[0]; fa, fb = FREQ[co][a], FREQ[co][b]; sm = difflib.SequenceMatcher(None, a, b).ratio()
    rows.append((t, a, b, fa, fb, sm))
hp = hp.with_columns(pl.DataFrame(rows, schema=["t", "wa", "wb", "fa", "fb", "sm"], orient="row"))
hp.write_parquet("ho_all.parquet")
print(hp.group_by("t").agg(pl.len(), pl.col("y").mean().round(3), pl.col("kept").mean().round(3)).sort("t"))
