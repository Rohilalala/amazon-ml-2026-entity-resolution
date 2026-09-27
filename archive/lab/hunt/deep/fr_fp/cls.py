import re, unicodedata, difflib, collections
import polars as pl
LF = {"sarl","sas","sasu","sa","eurl","snc","sci","scop","selarl","inc","llc","ltd","corp","co","corporation","company","pvt","private","limited","llp","lp","pllc","incorporated","plc","l","c"}
def norm(s): s = unicodedata.normalize("NFKD", s or "").encode("ascii","ignore").decode().lower(); return re.sub(r"[^a-z0-9 ]", " ", s)
def core(n): return [t for t in norm(n).split() if t not in LF]
def lf(n): return {t for t in norm(n).split() if t in LF}
def sim(a,b): return difflib.SequenceMatcher(None,a,b).ratio()
DBA = re.compile(r"doing business as|\baka\b|\bdba\b|\bd/b/a\b|trading as|\bt/a\b|formerly|also known as", re.I)
def klass(n1, n2, tail):
    if DBA.search(n2 or ""): return "dba"
    if re.search(r"\.(com|fr|net|org|in|co)\b|www\.", (n2 or "").lower()): return "web"
    c1, c2 = core(n1), core(n2)
    if c1 == c2: return "exact"
    if sorted(c1) == sorted(c2): return "reorder"
    if "".join(c1) == "".join(c2): return "space"
    if len(c1) == len(c2):
        d = [(a,b) for a,b in zip(c1,c2) if a!=b]
        if all(sim(a,b) >= 0.6 for a,b in d): return "typo"
        if len(d) == 1:
            a,b = d[0]
            if b in tail: return "swap_tail"
            if a.isdigit() or b.isdigit(): return "swap_num"
            return "swap1"
    s1, s2 = collections.Counter(c1), collections.Counter(c2)
    add, miss = s2 - s1, s1 - s2
    # fuzzy cancel typos
    for a in list(miss):
        for b in list(add):
            if add[b] and miss[a] and sim(a,b) >= 0.6: add[b]-=1; miss[a]-=1
    add, miss = +add, +miss
    if not add and not miss: return "typo2"
    if add and set(add) <= tail and not miss: return "tail_add"
    if add and set(add) <= tail and len(miss) == 1 and c1 and c1[-1] in miss: return "tail_droplast"
    if add and set(add) <= tail: return "tail_dropother"
    if not add: return "missing"
    if not miss: return "extra"
    if any(t.isdigit() for t in list(add)+list(miss)): return "num_conflict"
    return "other"
TAIL_FR = {"fils","cie","compagnie","services","developpement","groupe","france","associes","et","and"}
TAIL_US = {"center","centre","services","service","partners","pc","group","associates","associate","and","com"}
TAIL_IN = TAIL_US | {"india","brothers","sons"}
TAILS = {"france": TAIL_FR, "us": TAIL_US, "india": TAIL_IN}
def add_class(df):
    return df.with_columns(pl.struct("n1","n2","cty").map_elements(lambda s: klass(s["n1"], s["n2"], TAILS[s["cty"]]), return_dtype=pl.String).alias("k"))
