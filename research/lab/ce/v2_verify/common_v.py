"""Verifier helpers (own numpy code, as verify_build/v01_verify.py): grouped rank/gap, exclusive owner rule,
expected-F0.5 top-k with none_scale, per-S1 F0.5, paired bootstrap."""
import numpy as np


class Groups:
    """Rows sorted by group, then value descending (stable)."""
    def __init__(self, g, v):
        self.o = np.lexsort((-v, g)); gs = g[self.o]; self.v = v[self.o]; n = len(v)
        start = np.r_[True, gs[1:] != gs[:-1]]
        self.gid = np.cumsum(start) - 1; self.gstart = np.flatnonzero(start)
        self.gsize = np.diff(np.r_[self.gstart, n]); self.pos = np.arange(n) - self.gstart[self.gid]

    def unsort(self, x):
        out = np.empty_like(x); out[self.o] = x; return out


def rank_gap(g, v):
    """rank('min', descending) within g; v minus the max of v over the other rows of g (NaN for singletons); group size."""
    G = Groups(g, v); s = G.v; n = len(s); gid = G.gid
    newrun = np.r_[True, s[1:] != s[:-1]] | (G.pos == 0)
    first = np.maximum.accumulate(np.where(newrun, np.arange(n), 0))
    rank = (first - G.gstart[gid] + 1).astype(np.float64)
    top1 = s[G.gstart]; top2 = np.full(len(G.gstart), np.nan); m = G.gsize > 1; top2[m] = s[G.gstart[m] + 1]
    ntop = np.bincount(gid, weights=(s == top1[gid]).astype(np.float64), minlength=len(G.gstart))
    other = np.where((s == top1[gid]) & (ntop[gid] == 1), top2[gid], top1[gid])
    return G.unsort(rank), G.unsort(s - other), G.unsort(G.gsize[gid].astype(np.float64))


def owner(j, p):
    """Exclusive owner rule: a pair keeps its prob only if it is the max prob of its j (ties all kept)."""
    o = np.argsort(j, kind="stable"); js = j[o]; st = np.flatnonzero(np.r_[True, js[1:] != js[:-1]])
    mx = np.maximum.reduceat(p[o], st); top = np.empty_like(p); top[o] = np.repeat(mx, np.diff(np.r_[st, len(js)]))
    return np.where(p >= top, p, 0.0)


class EF:
    """Expected-F0.5 top-k per group: keep top k maximising 1.25*cum/(0.25*(total+missed)+k) if it beats
    prod(1-p)*exp(-missed)*scale."""
    def __init__(self, g, p, missed):
        G = Groups(g, p); s = G.v; gid, gst, gsz = G.gid, G.gstart, G.gsize
        cum = s.copy()
        for L in range(1, int(gsz.max())):  # within-group cumulative sum, same summation order as a per-group cum_sum
            idx = gst[gsz > L] + L
            cum[idx] += cum[idx - 1]
        total = cum[gst + gsz - 1][gid]; k = G.pos + 1
        ef = 1.25 * cum / (0.25 * (total + missed) + k)
        self.none = np.exp(np.add.reduceat(np.log(np.clip(1 - s, 1e-12, None)), gst)) * np.exp(-missed)
        self.best = np.maximum.reduceat(ef, gst)
        hit = np.flatnonzero(ef == self.best[gid]); hg = gid[hit]
        fh = hit[np.r_[True, hg[1:] != hg[:-1]]]
        self.bestk = np.zeros(len(gst), np.int64); self.bestk[gid[fh]] = G.pos[fh] + 1
        self.G, self.p = G, p
        kk = self.bestk; inner = (kk > 0) & (kk < gsz) & (self.best > self.none)
        a = gst[inner] + kk[inner]
        self.tie_cut_groups = int(((s[a] == s[a - 1]) & (s[a] > 0)).sum())

    def mask(self, scale=1.0):
        kk = np.where(self.best > self.none * scale, self.bestk, 0)
        return self.G.unsort(self.G.pos + 1 <= kk[self.G.gid]) & (self.p > 0)


def f05(codes, n_true, keep, y):
    """Per-S1 F0.5 = 1.25 tp / (0.25 n_true + n_pred); 1 when n_true == n_pred == 0, 0 when tp == 0 otherwise."""
    n = len(n_true)
    npred = np.bincount(codes, weights=keep.astype(np.float64), minlength=n)
    tp = np.bincount(codes, weights=(keep & (y == 1)).astype(np.float64), minlength=n)
    nt = n_true.astype(np.float64)
    with np.errstate(divide="ignore", invalid="ignore"):
        f = np.where(tp > 0, 1.25 * tp / (0.25 * nt + npred), 0.0)
    return np.where(nt == 0, (npred == 0).astype(np.float64), f)


def boot(d, reps=2000, seed=20260927):
    rng = np.random.default_rng(seed); n = len(d)
    m = np.concatenate([d[rng.integers(0, n, (50, n))].mean(axis=1) for _ in range(reps // 50)])
    return {"diff": round(float(d.mean()), 6), "ci95": [round(float(np.quantile(m, 0.025)), 6), round(float(np.quantile(m, 0.975)), 6)],
            "p_le_0": round(float((m <= 0).mean()), 4)}


if __name__ == "__main__":  # self-check on a toy example
    g = np.array([0, 0, 0, 1, 1, 2]); p = np.array([0.9, 0.8, 0.1, 0.3, 0.3, 0.05])
    r, gap, n = rank_gap(g, p)
    assert r.tolist() == [1, 2, 3, 1, 1, 1] and np.allclose(gap[:3], [0.1, -0.1, -0.8]) and gap[3] == 0 and np.isnan(gap[5])
    assert owner(np.array([5, 5, 6]), np.array([0.2, 0.7, 0.1])).tolist() == [0.0, 0.7, 0.1]
    k = EF(g, p, 0.0).mask(); assert k.tolist() == [True, True, False, False, False, False], k
    assert f05(np.array([0, 0, 1]), np.array([2, 0]), np.array([True, False, False]), np.array([1, 1, 0])).tolist() == [1.25 / 1.5, 1.0]
    print("ok")
