import polars as pl
from addr import nums
def hrel(a1, a2):
    n1, n2 = nums(a1), nums(a2 or "")
    if not n1 or not n2 or n1[0] == n2[0]: return None
    x, y = str(n1[0]), str(n2[0])
    if n2[0] == 1: return "one"
    if y in x or x in y: return "sub"
    if n1[0] in n2 or n2[0] in n1: return "else_in"
    return "rand"
def add_hrel(df): return df.with_columns(pl.struct("ad1","ad2").map_elements(lambda s: hrel(s["ad1"], s["ad2"]), return_dtype=pl.String).alias("hr"))
