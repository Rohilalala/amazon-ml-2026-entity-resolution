import polars as pl, re
from cls import *
def sub(n1, n2):
    c1, c2 = core(n1), core(n2)
    ini = "".join(t[0] for t in norm(n1).split())
    ini2 = "".join(t[0] for t in c1)
    j = norm(n2).replace(" ", "")
    if j in (ini, ini2) or (len(c2) == 1 and len(j) <= 5 and j == ini2[:len(j)]): return "acronym"
    if not set(c1) & set(c2) and len(c2) <= 2: return "brand"
    return "other_rest"
def add_sub(df): return df.with_columns(pl.when(pl.col("k")=="other").then(pl.struct("n1","n2").map_elements(lambda s: sub(s["n1"], s["n2"]), return_dtype=pl.String)).otherwise(pl.col("k")).alias("k2"))
