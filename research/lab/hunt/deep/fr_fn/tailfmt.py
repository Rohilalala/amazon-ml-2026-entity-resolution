"""Tail-copy format: where the appended tail sits relative to the legal form in the candidate's name."""
STOP = {"and", "et", "de", "du", "the", "of", "la", "le", "les", "des", "d", "l"}
def tail_fmt(n1, c1, n2, c2):
    a, b, n = c1.split(), c2.split(), n2.split()
    if len(a) < 1 or len(b) < len(a): return None, None
    tw = b[len(a) - 1:]
    cs = set(b)
    leg = [p for p, t in enumerate(n) if t not in cs and t not in STOP]
    try: pt = max(p for p, t in enumerate(n) if t == tw[0])
    except ValueError: return " ".join(tw), "none"
    if not leg: return " ".join(tw), "nolegal"
    if any(p > pt for p in leg): return " ".join(tw), "before_legal"
    return " ".join(tw), "after_legal"
if __name__ == "__main__":
    assert tail_fmt("marquisat ecole sas", "marquisat ecole", "marquisat sas groupe", "marquisat groupe") == ("groupe", "after_legal")
    assert tail_fmt("bordeaux gestion sarl", "bordeaux gestion", "bordeaux groupe sas", "bordeaux groupe") == ("groupe", "before_legal")
    assert tail_fmt("lang club", "lang club", "lang et fils", "lang fils") == ("fils", "nolegal")
    print("ok")
