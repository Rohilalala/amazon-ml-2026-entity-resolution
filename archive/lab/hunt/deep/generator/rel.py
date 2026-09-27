"""Label-free relation type between an S1 record and a candidate: name transformation x address relation."""
import re
from rapidfuzz.distance import Levenshtein
from rapidfuzz import fuzz
def sim(a, b): return Levenshtein.normalized_similarity(a, b)
def ntype(a, b):
    """a, b: core token lists (S1, candidate). returns (type, detail)"""
    if a == b: return "exact", ""
    if not a or not b: return "empty", ""
    if sorted(a) == sorted(b): return "reorder", ""
    la, lb = len(a), len(b)
    if la == lb:
        d = [(x, y) for x, y in zip(a, b) if x != y]
        dis = [(x, y) for x, y in d if sim(x, y) < 0.6]
        if not dis: return f"typo{min(len(d),2)}", ""
        if len(dis) == 1 and len(d) == 1:
            x, y = dis[0]
            return ("swaplast" if a[-1] != b[-1] else "swapmid"), f"{x}>{y}"
        if len(dis) == 1: return "swap+typo", f"{dis[0][0]}>{dis[0][1]}"
        return "multi", ""
    if lb > la and b[:la] == a: return "append", " ".join(b[la:])
    if lb > la and b[lb-la:] == a: return "prepend", " ".join(b[:lb-la])
    if lb >= la and la >= 2 and b[:la-1] == a[:-1]: return "droplast_add", a[-1] + ">" + " ".join(b[la-1:])
    if lb < la:
        it = iter(a)
        if all(t in it for t in b):
            dr = [t for t in a if t not in b]
            return ("droplast" if b == a[:lb] else "dropmid"), " ".join(dr)
    # typo-tolerant prefix/suffix versions
    if lb > la and all(sim(x, y) >= 0.6 for x, y in zip(a, b[:la])): return "append~", " ".join(b[la:])
    if lb >= la and la >= 2 and all(sim(x, y) >= 0.6 for x, y in zip(a[:-1], b[:la-1])): return "droplast_add~", a[-1] + ">" + " ".join(b[la-1:])
    r = fuzz.token_set_ratio(" ".join(a), " ".join(b))
    return ("fuzzy_hi" if r >= 80 else "other"), ""
def num(h):
    m = re.match(r"\d+", h or "")
    return int(m.group()) if m else None
def atype(h1, h2, addr2):
    if not addr2 or not addr2.strip(): return "noaddr"
    n1, n2 = num(h1), num(h2)
    if n1 is None or n2 is None: return "nohouse"
    if n1 == n2: return "same"
    if abs(n1 - n2) <= 2: return "pm2"
    return "diff"
if __name__ == "__main__":
    assert ntype(["a", "b"], ["a", "b"])[0] == "exact"
    assert ntype(["acme", "center"], ["acme", "services"]) == ("swaplast", "center>services")
    assert ntype(["le", "petit", "bistro"], ["le", "petit", "et", "fils"])[0] == "droplast_add"
    assert ntype(["acme", "tools"], ["acme", "tools", "group"]) == ("append", "group")
    assert atype("12", "14", "x") == "pm2" and atype("12", None, "") == "noaddr"
    print("ok")
