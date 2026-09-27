import polars as pl
R = "/Users/adityarohilla/Documents/amazon-ml-2026/"; O = R + "v8_lab/hunt/deep/usin/out/"
D = pl.read_parquet(O + "missexact_kept.parquet")
pl.Config.set_tbl_rows(60); pl.Config.set_tbl_width_chars(200)
print(D.filter(pl.col("side") == "holdout").group_by("country", "h1miss", "a2empty", "ns1b").agg(n=pl.len(), fp=(pl.col("y") == 0).sum()).sort("country", "n", descending=True))
ns = {("holdout", "us"): 59714, ("holdout", "india"): 40286, ("test", "us"): 663106, ("test", "india"): 809986}
# per S1: how many exact addressless copies kept, for ns1==1
g = D.filter(pl.col("a2empty") & (pl.col("ns1") == 1)).group_by("side", "country", "i").agg(k=pl.len(), fp=(pl.col("y") == 0).sum(), npool=pl.col("npool").first())
x = g.group_by("side", "country", pl.col("k").clip(1, 4)).agg(n=pl.len(), fp=pl.col("fp").sum())
x = x.with_columns(ps=pl.struct("side", "country", "n").map_elements(lambda s: s["n"] / ns[(s["side"], s["country"])] * 1000, return_dtype=pl.Float64))
print(x.sort("country", "k", "side"))
