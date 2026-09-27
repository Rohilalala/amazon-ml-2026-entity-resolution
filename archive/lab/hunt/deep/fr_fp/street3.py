import re, difflib, polars as pl
from cls import norm
from street import ST
REG = {"hauts","france","pays","loire","nouvelle","aquitaine","nord","gironde","atlantique","calais"}
def s1words(a):
    for p in (a or "").split(","):
        if re.search(r"\d", p):
            return [t for t in norm(p).split() if not re.search(r"\d", t) and t not in ST and len(t) >= 4]
    return None
def allwords(a): return [t for t in norm(a).split() if not re.search(r"\d", t) and t not in ST and len(t) >= 4 and t not in REG]
def diff_street(a1, a2):
    """True when none of the S1 street words appears (fuzzy >= 0.6) anywhere in the candidate address."""
    w1, w2 = s1words(a1), allwords(a2 or "")
    if not w1 or not w2: return None
    return not any(difflib.SequenceMatcher(None, x, y).ratio() >= 0.6 for x in w1 for y in w2)
def add_ds(df): return df.with_columns(pl.struct("ad1","ad2").map_elements(lambda s: diff_street(s["ad1"], s["ad2"]), return_dtype=pl.Boolean).alias("ds"))
