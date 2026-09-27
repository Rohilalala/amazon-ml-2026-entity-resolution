"""Holdout check of the combination (v2 encoder logits + the richer stacker ACNRST_3s retrained OOF on STAGE4 with
v2 logits = v2_stacker model ACNRST_3s_v2ce) against each track alone, paired by S1 (s01_stage4.boot, 2,000 resamples).
Holdout keeps: stage4_A (v1, stack/out/holdout_keeps), stage4_A_v2 (Track A, v2_encoder/out/s01_v2_holdout_keeps),
ACNRST_3s and ACNRST_3s_v2ce recomputed here from the saved boosters on the gate's feature tables (v2_stacker/out,
out_v2ce), asserted equal to the F0.5 in their eval json. The holdout is used for evaluation only.
usage (repo venv): h01_holdout.py -> out/h01.json, out/holdout_keeps.parquet"""
import json, sys, time, resource
from pathlib import Path
import numpy as np, polars as pl, lightgbm as lgb
CE = Path("/Users/adityarohilla/Documents/amazon-ml-2026/v8_lab/ce"); VS = CE / "v2_stacker"; OUT = Path(__file__).parent / "out"
sys.path.insert(0, str(VS))
import feat  # CE_V2 unset: paths below are explicit
from feat import s4
s4._rss_guard()
T0 = time.time(); R = {}

H, SH = s4.load_holdout(np.zeros(1, bool))
ref = pl.read_parquet(CE / "stack/out/holdout_keeps.parquet", columns=["i", "j", "stage4_A"])
r2 = pl.read_parquet(CE / "v2_encoder/out/s01_v2_holdout_keeps.parquet", columns=["i", "j", "stage4_A_v2"])
for r in (ref, r2):
    assert (r["i"] == H["i"]).all() and (r["j"] == H["j"]).all()
keeps = {"stage4_A": ref["stage4_A"].to_numpy(), "stage4_A_v2": r2["stage4_A_v2"].to_numpy()}
for name, fdir in (("ACNRST_3s", VS / "out"), ("ACNRST_3s_v2ce", VS / "out_v2ce")):
    meta = json.load(open(VS / f"model/{name}.json")); cols = meta["features"]
    ext = {g: [c for c in cols if c in feat.GROUPS[g]] for g in "ST"}
    f = pl.read_parquet(fdir / "f_holdout.parquet", columns=["r", *[c for c in cols if not any(c in v for v in ext.values())]])
    for g, sc in ext.items():
        if sc:
            f = f.hstack(pl.read_parquet(VS / f"out/{g.lower()}_holdout.parquet", columns=sc))  # S / T do not depend on the encoder
    x = s4.X(f, cols)
    p = H["prob"].to_numpy().astype(np.float64).copy()
    p[f["r"].to_numpy()] = np.mean([lgb.Booster(model_file=str(VS / "model" / m)).predict(x) for m in meta["models"]], axis=0)
    keeps[name] = s4.decide(H["i"].to_numpy(), H["j"].to_numpy(), p, meta["rule"])
    R[f"rule_{name}"] = meta["rule"]
    del f, x, p
res, f, ctry = s4.score_block(H, SH, keeps)
R["holdout_f05"] = res
for name, fdir in (("ACNRST_3s", VS / "out"), ("ACNRST_3s_v2ce", VS / "out_v2ce")):
    assert res[name] == json.load(open(fdir / f"eval_{name}.json"))["holdout_f05"][name], name  # the gate's number
yy = H["y"].to_numpy()
R["pair_errors"] = {k: {"tp": int((v & (yy == 1)).sum()), "fp": int((v & (yy == 0)).sum()), "fn_in_cand": int((~v & (yy == 1)).sum())} for k, v in keeps.items()}
R["paired"] = {}
for a, b in (("ACNRST_3s_v2ce", "stage4_A"), ("ACNRST_3s_v2ce", "stage4_A_v2"), ("ACNRST_3s_v2ce", "ACNRST_3s"),
             ("stage4_A_v2", "stage4_A"), ("ACNRST_3s", "stage4_A"), ("ACNRST_3s", "stage4_A_v2")):
    d = f[a] - f[b]
    R["paired"][f"{a} - {b}"] = {"all": s4.boot(d), **{c: s4.boot(d[ctry == c]) for c in ("us", "india")},
                                "s1_better_worse": [int((d > 0).sum()), int((d < 0).sum())],
                                "pairs_flipped": int((keeps[a] != keeps[b]).sum())}
    print(a, "-", b, R["paired"][f"{a} - {b}"]["all"], flush=True)
pl.DataFrame({"i": H["i"], "j": H["j"], **keeps}).write_parquet(OUT / "holdout_keeps.parquet")
R["peak_rss_gb"] = round(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1e9, 2); R["seconds"] = round(time.time() - T0)
(OUT / "h01.json").write_text(json.dumps(R, indent=1))
print(json.dumps(res), R["peak_rss_gb"], R["seconds"])
