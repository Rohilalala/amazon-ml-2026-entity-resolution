"""Expected LB change of an add file vs base: each added pair TP with prob q (S1 was missing it) else FP (S1 was complete)."""
import sys, polars as pl
from est import f05
def est(base, out, q, total=1732544):
    rd = lambda p: dict(pl.read_csv(p, separator="\t", quote_char=None, infer_schema=False).fill_null("").iter_rows())
    B, O = rd(base), rd(out); s = 0.0
    for a, m in O.items():
        n0 = len([x for x in B[a].split(",") if x]); n1 = len([x for x in m.split(",") if x]); d = n1 - n0
        if d <= 0: continue
        s += q * (1.0 - f05(n0, n0, n1)) + (1 - q) * (f05(n0, n1, n0) - 1.0)
    return s / total
if __name__ == "__main__":
    for q in (0.5, 0.7, 0.8, 0.9): print(q, round(est(sys.argv[1], sys.argv[2], q), 5))
