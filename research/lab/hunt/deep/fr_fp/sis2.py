"""Position-agnostic sister-swap detector for France kept pairs (extension of fr_sister_drop.py v2)."""
import re, unicodedata, difflib, collections
import polars as pl
from load import rd
LF = {"sarl","sas","sasu","sa","eurl","snc","sci","scop","selarl","ei"}
TAIL = {"fils","cie","compagnie","services","developpement","groupe","france","associes","et","and"}
def norm(s):
    s = unicodedata.normalize("NFKD", s or "").encode("ascii","ignore").decode().lower()
    s = re.sub(r"\bs\.?\s?a\.?\s?r\.?\s?l\b\.?", "sarl", s); s = re.sub(r"\bs\.a\.s\.?(u)?\b", r"sas\1", s)
    return re.sub(r"[^a-z0-9 ]", " ", s)
def core(n): return [t for t in norm(n).split() if t not in LF]
pool = pl.concat([rd("test",2), rd("test",3)]).filter(pl.col("country").str.strip_chars().str.to_lowercase()=="france")
FREQ = collections.Counter(t for n in pool["business_name"].to_list() for t in core(n))
def swap(n1, n2, minf=20):
    c1, c2 = core(n1), core(n2)
    while len(c2) > len(c1) and c2 and c2[-1] in TAIL: c2 = c2[:-1]
    a, b = collections.Counter(c1) - collections.Counter(c2), collections.Counter(c2) - collections.Counter(c1)
    if sum(a.values()) != 1 or sum(b.values()) != 1 or len(c1) < 2: return None
    x, y = next(iter(a)), next(iter(b))
    if not (x.isalpha() and y.isalpha() and len(x) >= 3 and len(y) >= 3): return None
    if difflib.SequenceMatcher(None, x, y).ratio() >= 0.6 or y in TAIL or {x, y} <= {"cie","compagnie"}: return None
    if FREQ[x] < minf or FREQ[y] < minf: return None
    return (x, y)
