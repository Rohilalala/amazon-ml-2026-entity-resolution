"""Expected LB change of a drop file vs base under 'dropped pair is FP with prob q, other kept pairs correct'; emptied S1 counted neutral."""
import sys, polars as pl
def f05(tp, npred, ntrue):
    if npred == 0 or ntrue == 0 or tp == 0: return 0.0
    p, r = tp / npred, tp / ntrue; return 1.25 * p * r / (0.25 * p + r)
def est(base, out, q, total=1732544):
    rd = lambda p: dict(pl.read_csv(p, separator="\t", quote_char=None, infer_schema=False).fill_null("").iter_rows())
    B, O = rd(base), rd(out); s = 0.0
    for a, m in B.items():
        n0 = len([x for x in m.split(",") if x]); n1 = len([x for x in O[a].split(",") if x]); d = n0 - n1
        if d == 0 or n1 == 0: continue
        # all d dropped are FP (prob q^d approx q per pair; use independent per-pair expectation)
        gain = f05(n1, n0, n1) ; before_fp = gain; after_fp = 1.0            # truth = the n1 kept
        before_tp, after_tp = 1.0, f05(n1, n1, n0)                            # truth = all n0 kept
        s += q * (after_fp - before_fp) + (1 - q) * (after_tp - before_tp)
    return s / total
if __name__ == "__main__":
    for q in (0.8, 0.9, 0.95): print(q, round(est(sys.argv[1], sys.argv[2], q), 5))
