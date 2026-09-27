"""Exact expected-F0.5 top-k selection (Poisson-binomial over the S1's candidates, optional Poisson(missed) outside)
versus the ratio approximation in src.pipeline.expected_f05_mask. Holdout: same probs, same owner rule, recall adds fixed."""
import sys, json, numpy as np, polars as pl
from math import exp, factorial
sys.path.insert(0, "/Users/adityarohilla/Documents/amazon-ml-2026/v8_lab/ce/stack"); import s01_stage4 as s4
R = "/Users/adityarohilla/Documents/amazon-ml-2026"

def conv(d, p):
    o = np.zeros(len(d) + 1); o[:-1] += d * (1 - p); o[1:] += d * p; return o

def exact_keep(i, p, missed=0.0, lo=0.02, hi=0.98):
    """p: owner-zeroed probs. Returns keep mask. Only S1 with a candidate in [lo, hi) are re-decided; others keep p>=0.5."""
    keep = p >= 0.5
    df = pl.DataFrame({"i": i, "p": p, "idx": np.arange(len(p))}).filter(pl.col("p") > 1e-3)
    unc = df.filter((pl.col("p") >= lo) & (pl.col("p") < hi))["i"].unique()
    df = df.filter(pl.col("i").is_in(unc.implode())).sort(["i", "p"], descending=[False, True])
    pois = np.array([exp(-missed) * missed ** m / factorial(m) for m in range(4)]) if missed > 0 else np.array([1.0])
    for g in df.partition_by("i", maintain_order=True):
        q = g["p"].to_numpy(); n = len(q); ix = g["idx"].to_numpy()
        suf = [None] * (n + 1); d = pois.copy(); suf[n] = d
        for t in range(n - 1, -1, -1): d = conv(d, q[t]); suf[t] = d
        best, bk = suf[0][0], 0  # k=0: F=1 iff nothing true anywhere
        pre = np.array([1.0])
        for k in range(1, n + 1):
            pre = conv(pre, q[k - 1]); rest = suf[k]
            a = np.arange(len(pre))[:, None]; b = np.arange(len(rest))[None, :]
            f = np.where(a > 0, 1.25 * a / (0.25 * (a + b) + k), 0.0)
            e = float(pre @ f @ rest)
            if e > best: best, bk = e, k
        keep[ix] = False; keep[ix[:bk]] = True
    return keep

if __name__ == "__main__":
    K = pl.read_parquet("hold_keeps.parquet"); A = pl.read_parquet("hold_adds.parquet").filter("add")
    S = pl.read_parquet(f"{R}/v8_lab/ce/setup/out/holdout_s1.parquet").sort("i")
    i, j, p, y, k0 = (K[c].to_numpy() for c in ("i", "j", "p", "y", "keep"))
    top = pl.DataFrame({"j": j, "p": p}).select(pl.col("p").max().over("j"))["p"].to_numpy()
    pe = np.where(p >= top, p, 0.0)
    assert (s4.apply_rule({"kind": "expected_f05", "missed": 0.0}, i, pe) & (pe > 0) == k0).all()
    rows = S["i"].to_numpy(); nt = S["n_true"].to_numpy().astype(float); code = np.searchsorted(rows, i); acode = np.searchsorted(rows, A["i"].to_numpy())
    ctry = S["country"].to_numpy()
    def f(keep):
        kept = np.bincount(code, weights=keep, minlength=len(rows)) + np.bincount(acode, minlength=len(rows))
        tp = np.bincount(code, weights=keep & (y == 1), minlength=len(rows)) + np.bincount(acode, weights=A["y"].to_numpy() == 1, minlength=len(rows))
        return s4.f05_counts(kept, nt, tp)
    f0 = f(k0); print("base", f0.mean())
    out = {}
    for m in [float(x) for x in sys.argv[1:]] or [0.0]:
        k1 = exact_keep(i, pe, m) & (pe > 0); f1 = f(k1); d = f1 - f0
        out[m] = {"f": round(float(f1.mean()), 6), "n_add": int((k1 & ~k0).sum()), "n_drop": int((k0 & ~k1).sum()),
                  "add_true": int((k1 & ~k0 & (y == 1)).sum()), "drop_true": int((k0 & ~k1 & (y == 1)).sum()),
                  "all": s4.boot(d), "us": s4.boot(d[ctry == "us"]), "india": s4.boot(d[ctry == "india"])}
        print(m, json.dumps(out[m]), flush=True)
        np.save(f"exact_keep_m{m}.npy", k1)
    json.dump(out, open("a08_exactef.json", "w"), indent=1)
