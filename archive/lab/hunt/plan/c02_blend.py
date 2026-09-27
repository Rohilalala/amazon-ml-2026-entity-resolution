"""Holdout check: blend of two stage-4 stackers' holdout probabilities (v3 = ACNRST_3s_v3 on v3 logits, adapt = ACNRST_3s_adapt on
the round-1 adapted logits), decided with the same owner rule + rule, vs each alone. Paired bootstrap by S1. Evaluation only."""
import json, os, sys
from pathlib import Path
import numpy as np, polars as pl, lightgbm as lgb
HERE = Path(__file__).parent; sys.path.insert(0, str(HERE / "s4ce"))
R = "/Users/adityarohilla/Documents/amazon-ml-2026/"
def probs(tag, name, cedir):
    os.environ["CE_DIR"], os.environ["CE_TAG"] = cedir, tag
    import importlib, feat; importlib.reload(feat); from feat import s4
    M = json.load(open(HERE / f"s4ce/model/{name}.json")); OUT = HERE / f"s4ce/out_{tag}"; STR = feat.V2S / "out"
    cols = M["features"]; ext = {g: [c for c in cols if c in feat.GROUPS[g]] for g in "ST"}
    f = pl.read_parquet(OUT / "f_holdout.parquet", columns=["r"] + [c for c in cols if not any(c in v for v in ext.values())])
    for g, sc in ext.items():
        if sc: f = f.hstack(pl.read_parquet(STR / f"{g.lower()}_holdout.parquet", columns=sc))
    H, SH = s4.load_holdout(np.zeros(1, bool)); p = H["prob"].to_numpy().astype(np.float64).copy()
    p[f["r"].to_numpy()] = np.mean([lgb.Booster(model_file=str(HERE / "s4ce/model" / fn)).predict(s4.X(f, cols)) for fn in M["models"]], axis=0)
    return s4, H, SH, p, M["rule"]
s4, H, SH, p3, rule3 = probs("v3", "ACNRST_3s_v3", R + "v8_lab/ce/v3_encoder/out")
_, _, _, pa, rulea = probs("adapt", "ACNRST_3s_adapt", R + "v8_lab/adapt/train/out/logits")
i, j = H["i"].to_numpy(), H["j"].to_numpy()
keeps = {"v3": s4.decide(i, j, p3, rule3), "adapt": s4.decide(i, j, pa, rulea)}
for w in (0.5,):
    keeps[f"blend{w}"] = s4.decide(i, j, w * p3 + (1 - w) * pa, rule3)
res, f, ctry = s4.score_block(H, SH, keeps)
print(json.dumps(res))
for k in keeps:
    if k != "v3": print(k, "vs v3", s4.boot(f[k] - f["v3"]), "US", s4.boot((f[k] - f["v3"])[ctry == "us"]), "India", s4.boot((f[k] - f["v3"])[ctry == "india"]))
