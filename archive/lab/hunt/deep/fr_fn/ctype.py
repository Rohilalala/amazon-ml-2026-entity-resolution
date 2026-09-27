"""Copy-type classifier for (S1 name, candidate name) using the cache's normalized name + core columns."""
import re
from rapidfuzz.fuzz import ratio
WEB = re.compile(r"\b(com|fr|net|org|www|co)\b")
TAILW = {"fils", "cie", "compagnie", "services", "developpement", "groupe", "france", "associes", "sons", "group", "associates", "partners", "center", "services"}

def abbrev(c1, c2):
    ini = {"".join(w[0] for w in c1[a:b]) for a in range(len(c1)) for b in range(a + 2, len(c1) + 1)}
    return any(len(t) >= 2 and t in ini and t not in c1 for t in c2)

def ctype(n1, c1s, n2, c2s):
    if not n2: return "empty"
    if n1 == n2: return "exact"
    if c1s == c2s: return "legal"
    c1, c2 = c1s.split(), c2s.split()
    if WEB.search(n2) and not WEB.search(n1): return "website"
    if "doing business as" in n2 or " dba " in f" {n2} " or "faisant affaire" in n2: return "dba"
    if sorted(c1) == sorted(c2): return "reorder"
    k = len(c1) - 1
    if k >= 1 and c2[:k] == c1[:k] and len(c2) > k and c1[-1] not in c2[k:] and ratio(c1[-1], c2[k]) < 60: return "tail" if set(c2[k:]) <= TAILW else "lastswap"
    if len(c1) == len(c2) and len(c1) >= 2:
        d = [(a, b) for a, b in zip(c1, c2) if a != b]
        if len(d) == 1 and ratio(*d[0]) < 60: return "swap1"
    s1, s2 = set(c1), set(c2)
    if s2 and s2 < s1: return "drop"
    if s1 and s1 < s2: return "add"
    if abbrev(c1, c2): return "abbrev"
    if ratio(" ".join(sorted(c1)), " ".join(sorted(c2))) >= 80 or ratio(c1s, c2s) >= 80: return "typo"
    if s1 & s2: return "partial"
    return "other"

if __name__ == "__main__":
    T = [("beacon biotechnologies llc", "beacon biotechnologies", "beaconbiotechnologies com", "beaconbiotechnologies com", "website"),
         ("childrens network", "childrens network", "childrens center", "childrens center", "tail"),
         ("a b c", "a b c", "c a b", "c a b", "reorder"),
         ("x ecole y", "x ecole y", "x amicale y", "x amicale y", "swap1"),
         ("jerusalem musique", "jerusalem musique", "jm", "jm", "abbrev"),
         ("meor superior llc", "meor superior", "meor suerior llc", "meor suerior", "typo"),
         ("a b", "a b", "", "", "empty")]
    for n1, c1, n2, c2, want in T: assert ctype(n1, c1, n2, c2) == want, (n1, n2, ctype(n1, c1, n2, c2), want)
    print("ok")
