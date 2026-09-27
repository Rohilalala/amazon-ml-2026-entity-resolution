import polars as pl, re
from cls import norm
def nums(a): return [int(x) for x in re.findall(r"\d+", norm(a))][:3]
def hcls(a1, a2):
    if a2 is None or a2.strip() == "": return "c_noaddr"
    n1, n2 = nums(a1), nums(a2)
    if not n1: return "s1_nonum"
    if not n2: return "c_nonum"
    if n1[0] == n2[0]: return "same"
    if n1[0] in n2: return "same_elsewhere"
    d = abs(n1[0]-n2[0])
    return "diff1" if d <= 2 else "diffbig"
def add_h(df): return df.with_columns(pl.struct("ad1","ad2").map_elements(lambda s: hcls(s["ad1"], s["ad2"]), return_dtype=pl.String).alias("h"))
