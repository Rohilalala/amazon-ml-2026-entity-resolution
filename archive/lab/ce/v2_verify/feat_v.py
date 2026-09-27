"""Verifier's own stage-4 feature code for the ACNRST model (57 columns), numpy for groups A, C, N; the v7 library
functions (src.features.pair_features without tables, src.alignment._bag/_numbers) and common.relation for S, T, R,
called on the verifier's own positional record lookup. No import of v2_stacker/feat.py."""
import sys
from pathlib import Path
import numpy as np, polars as pl

ROOT = Path("/Users/adityarohilla/Documents/amazon-ml-2026")
sys.path.insert(0, str(ROOT / "code/business_entity_resolution"))
sys.path.insert(0, str(ROOT / "v8_lab/audit_a"))
from src.features import pair_features, RECORD_COLUMNS  # noqa: E402
from src.alignment import _bag, _numbers  # noqa: E402
from common import relation  # noqa: E402
sys.path.insert(0, str(Path(__file__).parent))
from common_v import Groups, rank_gap  # noqa: E402

HREL = ["no_addr", "no_house", "eq", "substr", "edit1", "off20", "far"]  # training's integer codes
NREL = ["same", "subset", "typo1", "swap1", "disjoint", "multi"]
S_COLS = ["fz_name_ratio", "fz_name_partial", "fz_name_sort", "fz_name_set", "jw_name", "fz_core_ratio",
          "fz_core_partial", "fz_core_set", "jw_core", "fz_skel_ratio", "fz_skel_set", "fz_concat", "fz_addr_ratio",
          "fz_addr_set", "fz_addr_sort", "fz_addr_partial", "jac_core", "jac_addr", "jac_num", "shared_num",
          "house_match", "house_num_eq", "house_lev", "house_logdiff", "house_inside", "n_nums_1", "n_nums_2",
          "acr_match", "len_core_1", "len_core_2", "len_addr_1", "len_addr_2", "is_s3"]


def best_other(g, v):
    """Max of v over the other rows of the same group (NaN for singletons)."""
    G = Groups(g, v); s = G.v; gid = G.gid
    top1 = s[G.gstart]; top2 = np.full(len(G.gstart), np.nan); m = G.gsize > 1; top2[m] = s[G.gstart[m] + 1]
    ntop = np.bincount(gid, weights=(s == top1[gid]).astype(np.float64), minlength=len(G.gstart))
    return G.unsort(np.where((s == top1[gid]) & (ntop[gid] == 1), top2[gid], top1[gid]))


def gmax(g, v):
    """Per-row max of v over its group, ignoring NaN (NaN if all NaN)."""
    o = np.argsort(g, kind="stable"); gs = g[o]; st = np.flatnonzero(np.r_[True, gs[1:] != gs[:-1]])
    mx = np.fmax.reduceat(v[o], st); out = np.empty_like(v); out[o] = np.repeat(mx, np.diff(np.r_[st, len(gs)])); return out


def gsum(g, v):
    u, inv = np.unique(g, return_inverse=True)
    return np.bincount(inv, weights=v)[inv]


def features(i, j, prob, ce, emp, recs):
    """i, j: record rows (numpy); prob (v7/stack prob), ce (encoder logit); emp: bool per record row (empty address);
    recs: DataFrame with row + RECORD_COLUMNS covering every i and j. Returns {name: float64 array}."""
    F = {}
    q = np.clip(prob.astype(np.float64), 1e-6, 1 - 1e-6)
    F["lp"] = np.log(q / (1 - q)); F["ce"] = ce.astype(np.float64); F["b_empty"] = emp[j].astype(np.float64)
    F["ce_rank"], F["ce_gap"], F["n_i"] = rank_gap(i, F["ce"])
    F["lp_rank"], F["lp_gap"], _ = rank_gap(i, F["lp"])
    F["ce_top2"], F["lp_top2"] = gmax(i, F["ce_gap"]), gmax(i, F["lp_gap"])
    F["n_ce_hi"] = gsum(i, (F["ce"] >= 8).astype(np.float64))
    F["sum_p"] = gsum(i, prob.astype(np.float64))
    r = recs.sort("row"); rr = r["row"].to_numpy()
    li, lj = np.searchsorted(rr, i), np.searchsorted(rr, j)
    assert (rr[li] == i).all() and (rr[lj] == j).all()
    addr_j, core_j = r["addr"].gather(lj), r["core"].gather(lj)
    for tag, key in (("sa", addr_j), ("sc", core_j)):
        k = pl.DataFrame({"i": i, "k": key}).select(pl.struct("i", "k").rank("dense")).to_series().to_numpy().astype(np.int64)
        ok = (key != "").to_numpy()
        n = np.bincount(k)[k]
        F[f"{tag}_n"] = np.where(ok, n - 1, 0).astype(np.float64)
        F[f"{tag}_ce"] = np.where(ok, best_other(k, F["ce"]), np.nan)
    rel = relation(pl.DataFrame({"i": i, "j": j}), r.select("row", "core", "addr", "house"))
    assert (rel["i"].to_numpy() == i).all() and (rel["j"].to_numpy() == j).all()
    F["hrel"] = rel["hrel"].replace_strict(HREL, list(range(len(HREL)))).to_numpy().astype(np.float64)
    F["nrel"] = rel["nrel"].replace_strict(NREL, list(range(len(NREL)))).to_numpy().astype(np.float64)
    for s in range(0, len(i), 400_000):
        ci, cj = li[s:s + 400_000], lj[s:s + 400_000]
        pf = pair_features(r, pl.DataFrame({"i": ci, "j": cj}))
        for c in S_COLS:
            F.setdefault(c, []).append(pf[c].cast(pl.Float64).to_numpy())
        a, b = r[ci], r[cj]
        F.setdefault("chr_bag_core", []).append(np.einsum("ij,ij->i", _bag(a["core"]), _bag(b["core"])).astype(np.float64))
        F.setdefault("chr_bag_name", []).append(np.einsum("ij,ij->i", _bag(a["name"]), _bag(b["name"])).astype(np.float64))
        for k2, v in _numbers(a, b).items():
            F.setdefault(k2, []).append(np.asarray(v, dtype=np.float64))
    for c in [*S_COLS, "chr_bag_core", "chr_bag_name", "num_frac_1_in_2", "num_min_logdiff", "num_min_lev", "num_prefix"]:
        F[c] = np.concatenate(F[c])
    return F


def X(F, cols):
    return np.column_stack([F[c] for c in cols]).astype(np.float32)
