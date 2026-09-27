"""Holdout check of the combination: v3 stage 4 (ACNRST_3s_v3, saved models) + recall-head adds re-derived on the v3 keeps,
against the recall file's configuration (A_all_v2 keeps + adds, 0.990257). Paired bootstrap by S1. Evaluation only.
usage: CE_DIR=.../v3_encoder/out CE_TAG=v3 ../../../code/business_entity_resolution/.venv/bin/python c01_combined.py"""
import json, sys
from pathlib import Path
import numpy as np, polars as pl, lightgbm as lgb
HERE = Path("/Users/adityarohilla/Documents/amazon-ml-2026/v8_lab/hunt/plan"); ME = Path(__file__).parent
sys.path.insert(0, str(HERE / "s4ce"))
import feat
from feat import s4
V8 = HERE.parent.parent
sys.path.insert(0, str(V8 / "recall/head"))
import h01_head as h

M = json.load(open(HERE / "s4ce/model/ACNRST_3s_v3.json")); COLS, RULE = M["features"], M["rule"]
OUT = HERE / "s4ce/out_v3"; STR_OUT = feat.V2S / "out"
def load(mode, cols):
    ext = {g: [c for c in cols if c in feat.GROUPS[g]] for g in "ST"}
    f = pl.read_parquet(OUT / f"f_{mode}.parquet", columns=[c for c in cols if not any(c in v for v in ext.values())])
    for g, sc in ext.items():
        if sc: f = f.hstack(pl.read_parquet(STR_OUT / f"{g.lower()}_{mode}.parquet", columns=sc))
    return f
H, SH = s4.load_holdout(np.zeros(1, bool))
FH = load("holdout", ["r", *COLS])
p = H["prob"].to_numpy().astype(np.float64).copy()
p[FH["r"].to_numpy()] = np.mean([lgb.Booster(model_file=str(HERE / "s4ce/model" / fn)).predict(s4.X(FH, COLS)) for fn in M["models"]], axis=0)
keep3 = s4.decide(H["i"].to_numpy(), H["j"].to_numpy(), p, RULE)
K2 = pl.read_parquet(V8 / "ce/v2_build/out/holdout_keeps.parquet", columns=["i", "j", "ACNRST_3s_v2ce"])
assert (K2["i"] == H["i"]).all() and (K2["j"] == H["j"]).all()
keep2 = K2["ACNRST_3s_v2ce"].to_numpy()

S = pl.read_parquet(h.SET / "holdout_s1.parquet").sort("i"); rows = S["i"].to_numpy(); n_true = S["n_true"].to_numpy().astype(float)
ky = H["y"].to_numpy(); kc = np.searchsorted(rows, H["i"].to_numpy()); assert (rows[kc] == H["i"].to_numpy()).all()
A = pl.read_parquet(h.OUT / "holdout_add.parquet")  # s1_row, j_row, y, prob, add (add derived on the v2 keeps)
vcode = np.searchsorted(rows, A["s1_row"].to_numpy()); vy = A["y"].to_numpy(); pv = A["prob"].to_numpy(); vj = A["j_row"].to_numpy()
hm = json.load(open(h.MOD / "head.json")); tk, tn = hm["t_keep"], hm["t_none"]
def combo(keep):
    kept0 = np.bincount(kc, weights=keep, minlength=len(rows)); tp0 = np.bincount(kc, weights=keep & (ky == 1), minlength=len(rows))
    taken = np.isin(vj, H["j"].to_numpy()[keep])
    add = h.add_new(vcode, vj, pv, np.where(kept0 > 0, tk, tn), taken)
    return h.f05_counts(kept0, n_true, tp0), h.macro(n_true, kept0, tp0, vcode, add, vy), add
f2, f2r, add2 = combo(keep2); f3, f3r, add3 = combo(keep3)
assert (add2 == A["add"].to_numpy()).all(), "v2 adds must reproduce the recall build"
ctry = S["country"].to_numpy()
R = {"f05": {k: {"all": round(float(v.mean()), 6), **{c: round(float(v[ctry == c].mean()), 6) for c in ("us", "india")}}
             for k, v in (("v2", f2), ("v2+recall", f2r), ("v3", f3), ("v3+recall", f3r))},
     "diff_v3recall_vs_v2recall": {"all": s4.boot(f3r - f2r), **{c: s4.boot((f3r - f2r)[ctry == c]) for c in ("us", "india")}},
     "adds": {"v2": int(add2.sum()), "v3": int(add3.sum()), "v3_true": int((add3 & (vy == 1)).sum())}}
print(json.dumps(R, indent=1))
pl.DataFrame({"i": H["i"], "j": H["j"], "y": H["y"], "country": H["country"], "p": p, "keep": keep3}).write_parquet(ME / "hold_keeps.parquet")
pl.DataFrame({"i": A["s1_row"], "j": A["j_row"], "y": A["y"], "prob": A["prob"], "add": add3}).write_parquet(ME / "hold_adds.parquet")
print("saved")
