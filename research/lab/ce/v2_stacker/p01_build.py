"""Feature tables for STAGE4 and HOLDOUT active pairs (feat.py), with checks against s01_stage4 on the holdout.
usage (repo venv): p01_build.py stage4|holdout -> out/f_{stage4,holdout}.parquet, out/p01_{mode}.json
out/emp_train.npy = s01_stage4.empty_address() saved once (its read peaks at 1.4 GB)."""
import json, sys, time
from pathlib import Path
import numpy as np, polars as pl
import feat
from feat import s4

HERE = Path(__file__).parent; OUT = HERE / "out"; SET, INF = s4.SET, s4.INF
EMP_FILE = OUT / "emp_train.npy"
V2 = feat.V2CE  # CE_V2=1: the v2 encoder's logits (v2_encoder/out) instead of v1's; tables go to out_v2ce/
if V2:
    OUT = HERE / "out_v2ce"; OUT.mkdir(exist_ok=True)
T0 = time.time(); R = {}
s4._rss_guard()


def log(*a):
    print(f"[{time.time() - T0:7.1f}s]", *a, flush=True)


MODE = sys.argv[1]
EMP = np.load(EMP_FILE); assert EMP.sum() == 343428
if MODE == "stage4":
  # STAGE4 active pairs (s01_stage4.load_stage4 without its features)
  A = (pl.read_parquet(SET / "stage4_train.parquet", columns=["i", "j", "y", "country", "entity", "prob", "active"])
       .filter("active").drop("active"))
  C = pl.read_parquet(V2 / "STAGE4_ce.parquet" if V2 else INF / "stage4_ce.parquet")
  assert (A["i"] == C["i"]).all() and (A["j"] == C["j"]).all()
  A = A.with_columns(ce_logit=C["ce_logit"]); del C
  t = time.time(); A = feat.pair_inputs(A, "train"); R["stage4_pair_inputs_s"] = round(time.time() - t, 1)
  t = time.time(); A = feat.features(A, EMP); R["stage4_features_s"] = round(time.time() - t, 1)
  log("stage4", A.height, R)
  A.select("i", "j", "y", "country", "entity", "prob", *feat.names("ACJPNR")).write_parquet(OUT / "f_stage4.parquet")
  del A

else:
  H, SH = s4.load_holdout(EMP)
  if V2:
      H = H.drop("ce_logit").join(pl.read_parquet(V2 / "HOLDOUT_ce.parquet"), on=["i", "j"], how="left", maintain_order="left")
      assert H.filter("active")["ce_logit"].null_count() == 0
  a = H.with_row_index("r").filter("active").select("r", "i", "j", "prob", "ce_logit")
  t = time.time(); F = feat.pair_inputs(a, "train"); R["holdout_pair_inputs_s"] = round(time.time() - t, 1)
  t = time.time(); F = feat.features(F, EMP); R["holdout_features_s"] = round(time.time() - t, 1)
  ref = s4.features(a, EMP)  # s01 definitions (set A + set B columns)
  for c in s4.FEAT_B:
      x, y = F[c].cast(pl.Float64).to_numpy(), ref[c].cast(pl.Float64).to_numpy()
      R[f"eq_s01_{c}"] = bool(np.array_equal(np.isnan(x), np.isnan(y)) and np.array_equal(x[~np.isnan(x)], y[~np.isnan(y)]))
  assert all(v for k, v in R.items() if k.startswith("eq_s01_")), R
  log("holdout", F.height, R)
  F.select("r", "i", "j", "prob", *feat.names("ACJPNR")).write_parquet(OUT / "f_holdout.parquet")
  R["null_share_holdout"] = {c: round(F[c].null_count() / F.height, 4) for c in feat.names("ACJPNR") if F[c].null_count()}
  R["hrel_counts_holdout"] = dict(zip(feat.HREL, np.bincount(F["hrel"].to_numpy(), minlength=len(feat.HREL)).tolist()))
  R["nrel_counts_holdout"] = dict(zip(feat.NREL, np.bincount(F["nrel"].to_numpy(), minlength=len(feat.NREL)).tolist()))
import resource
R["peak_rss_gb"] = round(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1e9, 2)
(OUT / f"p01_{MODE}.json").write_text(json.dumps(R, indent=1))
log("done", R["peak_rss_gb"])
