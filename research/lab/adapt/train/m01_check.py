"""Quick check, adapted encoder (out/eval) vs v3 (ce/v3_encoder/out): AUC / log loss on VAL3, VAL2, CE-VAL (all, us, india) and on the
held-out 10% of the pseudo set (vs pseudo labels; all, per country, per class). Gate: labelled-set AUC drop <= 0.0003 on each of VAL3/VAL2/CE-VAL (all).
usage (ce venv): m01_check.py -> out/m01_check.json"""
import json
from pathlib import Path
import numpy as np, polars as pl
from sklearn.metrics import roc_auc_score, log_loss

HERE = Path(__file__).parent; CE = HERE.parent.parent / "ce"; V3 = CE / "v3_encoder/out"; EV = HERE / "out/eval"


def met(y, z):
    z = z.astype(np.float64); p = np.clip(1 / (1 + np.exp(-z)), 1e-7, 1 - 1e-7)
    r = {"logloss": round(log_loss(y, p, labels=[0, 1]), 5), "acc": round(float(((p >= 0.5) == y).mean()), 5)}
    if 0 < y.mean() < 1: r["auc"] = round(roc_auc_score(y, z), 5)
    return r


def cmp(d):
    y = d["y"].to_numpy(); a, b = met(y, d["v3"].to_numpy()), met(y, d["ad"].to_numpy())
    return {"pairs": d.height, "pos_rate": round(float(y.mean()), 4), "v3": a, "adapt": b,
            "d_auc": round(b["auc"] - a["auc"], 5) if "auc" in a else None, "d_logloss": round(b["logloss"] - a["logloss"], 5)}


R, gate = {}, True
for name, src, flag, tag in (("VAL3", V3 / "d01_pairs.parquet", "is_val3", "VAL3"), ("VAL2", CE / "v2_encoder/out/d01_pairs.parquet", "is_val2", "VAL2"),
                             ("CE-VAL", CE / "setup/out/ce_train.parquet", "is_val", "CEVAL")):
    D = pl.scan_parquet(src).filter(flag).select("i", "j", "y", "country").collect()
    D = D.join(pl.read_parquet(V3 / f"{tag}_ce.parquet").rename({"ce_logit": "v3"}), on=["i", "j"], how="left") \
         .join(pl.read_parquet(EV / f"{tag}_ce.parquet").rename({"ce_logit": "ad"}), on=["i", "j"], how="left")
    assert D["v3"].null_count() == 0 and D["ad"].null_count() == 0
    R[name] = {"all": cmp(D), **{c: cmp(D.filter(pl.col("country") == c)) for c in ("us", "india")}}
    gate &= R[name]["all"]["d_auc"] >= -0.0003
S = pl.read_parquet(HERE / "out/pseudo.parquet").filter("is_ho").rename({"ce": "v3"})
S = S.join(pl.read_parquet(EV / "PSEUDO_HO_ce.parquet").rename({"ce_logit": "ad"}), on=["i", "j"], how="left"); assert S["ad"].null_count() == 0
R["PSEUDO_HO"] = {"all": cmp(S), **{c: cmp(S.filter(pl.col("country") == c)) for c in ("us", "india", "france")},
                  **{f"class_{k}": cmp(S.filter(pl.col("cls") == k)) for k in ("pos", "hard", "easy")},
                  **{f"france_{k}_vs_pos": cmp(S.filter((pl.col("country") == "france") & pl.col("cls").is_in(["pos", k]))) for k in ("hard", "easy")},
                  "sign_agree_v3_adapt": round(float(((S["v3"] > 0) == (S["ad"] > 0)).mean()), 5)}
R["gate_labelled_auc_drop_le_0.0003"] = bool(gate)
(HERE / "out/m01_check.json").write_text(json.dumps(R, indent=1)); print(json.dumps(R, indent=1))
