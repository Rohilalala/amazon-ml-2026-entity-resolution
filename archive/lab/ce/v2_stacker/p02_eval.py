"""One stage-4 v2 variant: 5-fold S1-grouped CV on STAGE4 active pairs (s01_stage4.cv), rule by choose_rule on the
out-of-fold prob (non-active pairs keep v7's OOF prob), refit at the mean best iteration, then the HOLDOUT decision
(owner rule + rule, v7 prob outside ACTIVE) compared with the current stage 4 (stack/out/holdout_keeps stage4_A),
paired bootstrap by S1 (s01_stage4.boot, 2,000 resamples). The holdout is used for this comparison only.
usage (repo venv): p02_eval.py NAME GROUPS [lgb_param=json ...]   e.g. p02_eval.py AC AC  |  p02_eval.py AC_l63 AC num_leaves=63"""
import json, sys, time
from pathlib import Path
import numpy as np, polars as pl, lightgbm as lgb
import feat
from feat import s4

HERE = Path(__file__).parent; OUT = HERE / "out"; SET = s4.SET
STR_OUT = OUT  # S / T tables do not depend on the encoder
if feat.V2CE:
    OUT = HERE / "out_v2ce"
NAME, GROUPS = sys.argv[1], sys.argv[2]
OVER = {k: json.loads(v) for k, v in (a.split("=", 1) for a in sys.argv[3:])}
COLS = feat.names(GROUPS)
if OVER.pop("mono", None):  # +1 on the two score logits, 0 elsewhere
    OVER["monotone_constraints"] = [1 if c in ("lp", "ce") else 0 for c in COLS]
SEEDS = OVER.pop("seeds", None)  # seeds=[a,b,c]: one CV + final model per LightGBM seed, probabilities averaged
s4.PARAMS.update(OVER)  # s01_stage4.cv reads this dict
SEEDS = SEEDS or [s4.PARAMS["seed"]]
T0 = time.time(); R = {"name": NAME, "features": COLS, "params": s4.PARAMS}
s4._rss_guard()


def load(mode, cols):
    """out/f_{mode}.parquet columns, plus GROUPS S / T columns from the row-aligned out/{s,t}_{mode}.parquet."""
    ext = {g: [c for c in cols if c in feat.GROUPS[g]] for g in "ST"}
    f = pl.read_parquet(OUT / f"f_{mode}.parquet", columns=[c for c in cols if not any(c in v for v in ext.values())])
    for g, sc in ext.items():
        if sc:
            f = f.hstack(pl.read_parquet(STR_OUT / f"{g.lower()}_{mode}.parquet", columns=sc))
    return f


def log(*a):
    print(f"[{time.time() - T0:7.1f}s]", NAME, *a, flush=True)


T = pl.read_parquet(SET / "stage4_train.parquet", columns=["entity", "y", "prob", "active"])
S4 = pl.read_parquet(SET / "stage4_s1.parquet")
F = load("stage4", ["y", "entity", *COLS])
act = T["active"].to_numpy()
assert (F["y"].to_numpy() == T["y"].to_numpy()[act]).all() and (F["entity"].to_numpy() == T["entity"].to_numpy()[act]).all()
y, g, x = F["y"].to_numpy(), F["entity"].to_numpy(), s4.X(F, COLS); del F
oofs, frs = [], []
for sd in SEEDS:
    s4.PARAMS["seed"] = sd
    o, bs = s4.cv(x, y, g); oofs.append(o); frs.append([b.best_iteration for b in bs])
oof = np.mean(oofs, axis=0); del oofs
p_all = T["prob"].to_numpy().astype(np.float64).copy(); p_all[act] = oof
f4, rule, top = s4.choose_rule_chunked(T["entity"].to_numpy(), S4["n_true"].to_numpy().astype(float), p_all, T["y"].to_numpy())
rounds = [int(np.mean(r)) for r in frs]
from sklearn.metrics import roc_auc_score
R.update(seeds=SEEDS, fold_rounds=frs if len(SEEDS) > 1 else frs[0], final_rounds=rounds if len(SEEDS) > 1 else rounds[0], oof_auc_active=round(roc_auc_score(y, oof), 5),
         oof_f05=round(f4, 5), rule=rule, top=top)
log({k: R[k] for k in ("fold_rounds", "oof_auc_active", "oof_f05", "rule")})
del T, p_all, oof
finals = []
for sd, n in zip(SEEDS, rounds):
    s4.PARAMS["seed"] = sd
    finals.append(lgb.train(s4.PARAMS, lgb.Dataset(x, y), n))
del x
R["gain"] = dict(zip(COLS, np.sum([m.feature_importance("gain") for m in finals], axis=0).round(0).tolist()))

H, SH = s4.load_holdout(np.zeros(1, bool))  # empty flag unused here
FH = load("holdout", ["r", *COLS])
p = H["prob"].to_numpy().astype(np.float64).copy()
xh = s4.X(FH, COLS)
p[FH["r"].to_numpy()] = np.mean([m.predict(xh) for m in finals], axis=0)
keep = s4.decide(H["i"].to_numpy(), H["j"].to_numpy(), p, rule)
ref = pl.read_parquet(s4.HERE / "out/holdout_keeps.parquet", columns=["i", "j", "stage4_A"])
assert (ref["i"] == H["i"]).all() and (ref["j"] == H["j"]).all()
bases = {"base_stage4_A": ref["stage4_A"].to_numpy()}
if feat.V2CE:  # the v2 encoder track's stage 4 (set A on v2 logits)
    r2 = pl.read_parquet(feat.V2CE / "s01_v2_holdout_keeps.parquet", columns=["i", "j", "stage4_A_v2"])
    assert (r2["i"] == H["i"]).all() and (r2["j"] == H["j"]).all()
    bases["base_stage4_A_v2"] = r2["stage4_A_v2"].to_numpy()
res, f, ctry = s4.score_block(H, SH, {**bases, NAME: keep})
if feat.V2CE:
    d2 = f[NAME] - f["base_stage4_A_v2"]
    R["diff_vs_stage4_A_v2"] = {"all": s4.boot(d2), **{c: s4.boot(d2[ctry == c]) for c in ("us", "india")}}
d = f[NAME] - f["base_stage4_A"]
R["holdout_f05"] = res
R["diff_vs_stage4_A"] = {"all": s4.boot(d), **{c: s4.boot(d[ctry == c]) for c in ("us", "india")}}
R["s1_better_worse"] = [int((d > 0).sum()), int((d < 0).sum())]
yy = H["y"].to_numpy()
R["holdout_pair_errors"] = {"fp": int((keep & (yy == 0)).sum()), "fn_in_cand": int((~keep & (yy == 1)).sum())}
g0 = R["diff_vs_stage4_A"]["all"]
R["gate_pass"] = bool(g0["diff"] >= 0.0003 and g0["ci95"][0] > 0)
files = [f"{NAME}.txt"] if len(SEEDS) == 1 else [f"{NAME}_s{sd}.txt" for sd in SEEDS]
for m, fn in zip(finals, files):
    m.save_model(str(HERE / "model" / fn))
json.dump({"features": COLS, "rule": rule, "rounds": rounds if len(SEEDS) > 1 else rounds[0], "seeds": SEEDS, "models": files,
           "params": s4.PARAMS}, open(HERE / f"model/{NAME}.json", "w"), indent=1)
import resource
R["peak_rss_gb"] = round(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1e9, 2); R["seconds"] = round(time.time() - T0)
(OUT / f"eval_{NAME}.json").write_text(json.dumps(R, indent=1))
log(res, R["diff_vs_stage4_A"]["all"], "gate", R["gate_pass"])
