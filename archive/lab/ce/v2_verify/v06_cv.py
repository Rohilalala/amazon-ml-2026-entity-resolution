"""Verifier (6) the combined stacker from STAGE4 only: own 5-fold S1-grouped CV per seed (GroupKFold shuffle, seed 42,
early stopping 100 on the held fold, v7 LightGBM params), OOF mean over seeds; rule chosen on the OOF prob with own
scoring (thresholds + expected-F0.5 grid, non-active pairs keep v7's OOF prob, no owner rule as src.pipeline.choose_rule);
refit per seed at the mean best iteration on all STAGE4 active pairs and compared with the saved boosters' holdout
predictions (a booster fitted on anything else would not match). Also the holdout F0.5 of the combined model per rule.
usage (repo venv, heavy.sh): v06_cv.py -> out/v06_cv.json"""
import json, sys, time, resource
from pathlib import Path
import numpy as np, polars as pl, lightgbm as lgb
from sklearn.model_selection import GroupKFold

HERE = Path(__file__).parent; sys.path.insert(0, str(HERE))
from common_v import EF, f05, owner
ROOT = Path("/Users/adityarohilla/Documents/amazon-ml-2026"); CE = ROOT / "v8_lab/ce"; SET = CE / "setup/out"; VS = CE / "v2_stacker"; OUT = HERE / "out"
B7 = json.load(open(ROOT / "code/business_entity_resolution/artifacts/v7/bundle.json"))["config"]
THRESHOLDS = np.round(np.arange(0.05, 0.96, 0.01), 2); MISSED = (0.0, 0.05, 0.1, 0.2, 0.3, 0.5, 0.8)  # src.pipeline grids
T0 = time.time(); R = {}
log = lambda *a: print(f"[{time.time() - T0:6.1f}s]", *a, flush=True)
meta = json.load(open(VS / "model/ACNRST_3s_v2ce.json")); cols = meta["features"]
S_T = {"s": [], "t": []}
for g in S_T:
    S_T[g] = [c for c in cols if c in pl.read_parquet_schema(VS / f"out/{g}_stage4.parquet")]


def table(mode):
    f = pl.read_parquet(VS / f"out_v2ce/f_{mode}.parquet", columns=[c for c in cols if c not in S_T["s"] + S_T["t"]] + (["y", "entity"] if mode == "stage4" else ["r"]))
    f = f.hstack(pl.read_parquet(VS / f"out/s_{mode}.parquet", columns=S_T["s"])).hstack(pl.read_parquet(VS / f"out/t_{mode}.parquet", columns=S_T["t"]))
    return f


F = table("stage4")
x = F.select(cols).cast(pl.Float32).to_numpy(); y = F["y"].to_numpy(); g = F["entity"].to_numpy(); del F
folds = np.empty(len(y), np.int8)
for n, (_, va) in enumerate(GroupKFold(n_splits=5, shuffle=True, random_state=B7["seed"]).split(g, groups=g)):
    folds[va] = n
params0 = {**B7["lgb_params"], "num_threads": 8}
oofs, rounds = [], []
for sd in meta["seeds"]:
    P = {**params0, "seed": sd}
    full = lgb.Dataset(x, y, params=P, free_raw_data=False).construct()
    o, br = np.zeros(len(y)), []
    for n in range(5):
        tr, va = np.flatnonzero(folds != n), np.flatnonzero(folds == n)
        b = lgb.train(P, full.subset(tr), B7["num_boost_round"], valid_sets=[full.subset(va)], callbacks=[lgb.early_stopping(B7["early_stopping_rounds"], verbose=False)])
        o[va] = b.predict(x[va], num_iteration=b.best_iteration); br.append(b.best_iteration)
    oofs.append(o); rounds.append(br); log("seed", sd, br)
oof = np.mean(oofs, axis=0)
R["fold_rounds"] = rounds; R["final_rounds"] = [int(np.mean(r)) for r in rounds]
R["final_rounds_eq_saved"] = R["final_rounds"] == meta["rounds"]
# rule on OOF
T = pl.read_parquet(SET / "stage4_train.parquet", columns=["entity", "y", "prob", "active"])
S4 = pl.read_parquet(SET / "stage4_s1.parquet")
act = T["active"].to_numpy(); codes = T["entity"].to_numpy(); yy = T["y"].to_numpy(); nt = S4["n_true"].to_numpy()
p = T["prob"].to_numpy().astype(np.float64); p[act] = oof
trials = [(float(f05(codes, nt, p >= t, yy).mean()), {"kind": "threshold", "threshold": float(t)}) for t in THRESHOLDS]
trials += [(float(f05(codes, nt, EF(codes, p, m).mask(), yy).mean()), {"kind": "expected_f05", "missed": m}) for m in MISSED]
best = max(trials, key=lambda t: t[0])
R["oof_f05"] = round(best[0], 6); R["rule"] = best[1]; R["rule_eq_saved"] = best[1] == meta["rule"]
R["oof_top5"] = [{"f05": round(f, 6), **r} for f, r in sorted(trials, key=lambda t: -t[0])[:5]]
log("rule", R["rule"], R["oof_f05"], R["oof_top5"])
del T, p
# refit and compare with the saved boosters on the holdout table
H = table("holdout"); xh = H.select(cols).cast(pl.Float32).to_numpy()
cmpr = []
for sd, n, fn in zip(meta["seeds"], R["final_rounds"], meta["models"]):
    m = lgb.train({**params0, "seed": sd}, lgb.Dataset(x, y), n)
    a, b = m.predict(xh), lgb.Booster(model_file=str(VS / "model" / fn)).predict(xh)
    cmpr.append({"seed": sd, "max_abs_diff_vs_saved": float(np.abs(a - b).max()), "n_diff_gt_1e-9": int((np.abs(a - b) > 1e-9).sum())})
R["refit_vs_saved_holdout_pred"] = cmpr; log("refit", cmpr)
# holdout F0.5 of the combined model per expected-F0.5 setting (sensitivity; saved boosters)
HP = pl.read_parquet(SET / "holdout_pairs.parquet", columns=["i", "j", "y", "prob"]); HS = pl.read_parquet(SET / "holdout_s1.parquet")
hc = HP.select("i").join(pl.DataFrame({"i": HS["i"], "c": np.arange(HS.height)}), on="i", how="left", maintain_order="left")["c"].to_numpy()
ph = HP["prob"].to_numpy().astype(np.float64)
ph[H["r"].to_numpy()] = np.mean([lgb.Booster(model_file=str(VS / "model" / fn)).predict(xh) for fn in meta["models"]], axis=0)
pe = owner(HP["j"].to_numpy(), ph)
R["holdout_f05_by_missed"] = {str(m): round(float(f05(hc, HS["n_true"].to_numpy(), EF(hc, pe, m).mask(), HP["y"].to_numpy()).mean()), 6) for m in MISSED}
R["peak_rss_gb"] = round(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1e9, 2); R["seconds"] = round(time.time() - T0)
(OUT / "v06_cv.json").write_text(json.dumps(R, indent=1))
log("done", R["holdout_f05_by_missed"], R["peak_rss_gb"])
