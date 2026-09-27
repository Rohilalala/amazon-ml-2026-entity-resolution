import re, difflib, polars as pl
from cls import norm
from street import ST
def swords(a):
    for p in (a or "").split(","):
        if re.search(r"\d", p):
            return [t for t in norm(p).split() if not re.search(r"\d", t) and t not in ST and len(t) >= 4]
    return None
def strict_diff(a1, a2):
    w1, w2 = swords(a1), swords(a2)
    if not w1 or not w2: return None
    return not any(difflib.SequenceMatcher(None, x, y).ratio() >= 0.6 for x in w1 for y in w2)
def add_sd(df): return df.with_columns(pl.struct("ad1","ad2").map_elements(lambda s: strict_diff(s["ad1"], s["ad2"]), return_dtype=pl.Boolean).alias("sd"))
