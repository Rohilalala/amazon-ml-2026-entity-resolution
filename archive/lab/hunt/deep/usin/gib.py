"""Gibberish-name candidates (one invented token, no overlap with the S1 name) at a different house number: holdout precision vs test keep rate."""
import polars as pl, re
R = "/Users/adityarohilla/Documents/amazon-ml-2026/"; O = R + "v8_lab/hunt/deep/usin/out/"
def gib(c1, c2):
    t2 = (c2 or "").split()
    if len(t2) != 1: return False
    w = t2[0]
    return w.isalpha() and len(w) >= 6 and w not in set((c1 or "").split()) and not any(len(x) >= 3 and x in w for x in (c1 or "").split())
ns = {("holdout", "us"): 59714, ("holdout", "india"): 40286, ("test", "us"): 663106, ("test", "india"): 809986}
res = []
for side in ["holdout", "test"]:
    d = pl.read_parquet(O + side + "_diff.parquet")
    d = d.with_columns(pl.struct("c1", "c2").map_elements(lambda s: gib(s["c1"], s["c2"]), return_dtype=pl.Boolean).alias("gib")).filter(pl.col("gib"))
    if side == "test": d = d.with_columns(pl.lit(None).cast(pl.Int8).alias("y"))
    d.write_parquet(O + side + "_gib.parquet")
    res.append(d.filter(pl.col("keep")).group_by("country", "hs").agg(n=pl.len(), fp=(pl.col("y") == 0).sum(), p4lo=(pl.col("p4") < 0.99).sum() if side == "test" else pl.lit(0).cast(pl.UInt32)).with_columns(pl.lit(side).alias("side")))
    if side == "holdout":
        print("holdout all active gib by hs:", d.filter(pl.col("active")).group_by("country", "hs").agg(n=pl.len(), rate=pl.col("y").mean(), kept=pl.col("keep").sum()).sort("country", "hs"))
G = pl.concat(res, how="diagonal").with_columns(ps=pl.struct("side", "country", "n").map_elements(lambda s: s["n"] / ns[(s["side"], s["country"])] * 1000, return_dtype=pl.Float64))
W = G.filter(pl.col("side") == "holdout").drop("side", "p4lo").join(G.filter(pl.col("side") == "test").drop("side", "fp"), on=["country", "hs"], how="full", coalesce=True, suffix="_t")
pl.Config.set_tbl_rows(40); pl.Config.set_tbl_width_chars(200)
print(W.with_columns(ratio=pl.col("ps_t") / pl.col("ps"), exp_t=pl.col("ps") * pl.struct("country").map_elements(lambda s: ns[("test", s["country"])] / 1000, return_dtype=pl.Float64)).sort("country", "hs").with_columns(pl.col(pl.Float64).round(3)))
