import polars as pl, re
from cls import norm
from addr import nums
LFX = {"sarl","sas","sasu","sa","eurl","snc","sci","scop","selarl","ei","inc","llc","ltd","corp","co","corporation","company","incorporated","pllc","pc","lp","llp","pvt","private","limited","l"}
CANON = {"corporation":"corp","company":"co","incorporated":"inc","limited":"ltd","private":"pvt"}
def lfs(n):
    s = " " + norm(n) + " "
    s = s.replace(" s a r l ", " sarl ").replace(" s a s ", " sas ").replace(" l l c ", " llc ").replace(" l l p ", " llp ")
    return {CANON.get(t, t) for t in s.split() if t in LFX and t != "l"}
def lfrel(n1, n2):
    a, b = lfs(n1), lfs(n2)
    if a == b: return "lf_same"
    if not a: return "lf_add"
    if not b: return "lf_drop"
    if a < b: return "lf_add"
    return "lf_conflict"
def nrel(a1, a2):
    n1, n2 = nums(a1), nums(a2 or "")
    if not n1 or not n2 or n1[0] == n2[0]: return None
    x, y = str(n1[0]), str(n2[0])
    if n2[0] == 1: return "one"
    if len(x) == len(y) and sum(p != q for p, q in zip(x, y)) == 1 and abs(n1[0]-n2[0]) > 20: return "digit1"
    if abs(n1[0]-n2[0]) <= 20: return "near"
    if y in x or x in y: return "sub"
    return "far"
def add_rel(df): return df.with_columns(pl.struct("n1","n2").map_elements(lambda s: lfrel(s["n1"], s["n2"]), return_dtype=pl.String).alias("lr"),
                                          pl.struct("ad1","ad2").map_elements(lambda s: nrel(s["ad1"], s["ad2"]), return_dtype=pl.String).alias("nr"))
