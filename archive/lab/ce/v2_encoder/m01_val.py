"""Quick gate: VAL2 (10k fresh S1, never trained on by v1 or v2) and v1's CE-VAL (20k S1): AUC / log loss, v1 vs v2.
Segments: all, per country, fit-like (j-disjoint and rank<=6 or true), empty candidate address (s01_stage4 expression).
usage (ce venv): m01_val.py -> out/m01_val.json"""
import json
from pathlib import Path
import numpy as np, polars as pl
from sklearn.metrics import roc_auc_score, log_loss

HERE = Path(__file__).parent; CE = HERE.parent; SET = CE / "setup/out"; OUT = HERE / "out"
e = (pl.scan_parquet(SET / "text_train.parquet")
     .select("row", b=pl.col("text").str.splitn(" | ", 2).struct.field("field_1").fill_null("").str.strip_chars() == "")
     .collect(engine="streaming"))
assert (e["row"].to_numpy() == np.arange(e.height)).all()
EMP = e["b"].to_numpy(); del e


def metrics(d):
    y = d["y"].to_numpy(); r = {"pairs": d.height, "pos_rate": round(float(y.mean()), 4)}
    for m in ("v1", "v2"):
        z = d[m].to_numpy().astype(np.float64); p = np.clip(1 / (1 + np.exp(-z)), 1e-7, 1 - 1e-7)
        r[m] = {"auc": round(roc_auc_score(y, z), 5), "logloss": round(log_loss(y, p, labels=[0, 1]), 5)}
    return r


R = {}
for name, src, flag, v1 in (("VAL2", HERE / "out/d01_pairs.parquet", "is_val2", OUT / "v1/VAL2_ce.parquet"),
                            ("CE-VAL", SET / "ce_train.parquet", "is_val", None)):
    D = pl.scan_parquet(src).filter(flag).collect()
    w1 = pl.read_parquet(v1) if v1 else pl.read_parquet(CE / "train/out/val_scores.parquet", columns=["i", "j", "logit"]).rename({"logit": "ce_logit"})
    w2 = pl.read_parquet(OUT / f"{'VAL2' if name == 'VAL2' else 'CEVAL'}_ce.parquet")
    D = (D.join(w1.rename({"ce_logit": "v1"}), on=["i", "j"], how="left").join(w2.rename({"ce_logit": "v2"}), on=["i", "j"], how="left")
         .with_columns(emp=pl.Series(EMP[D["j"].to_numpy()])))
    assert D["v1"].null_count() == 0 and D["v2"].null_count() == 0
    fitlike = ~pl.col("j_in_stage4_active") & ~pl.col("j_in_holdout_active") & ((pl.col("rank") <= 6) | (pl.col("y") == 1))
    R[name] = {"all": metrics(D), **{c: metrics(D.filter(pl.col("country") == c)) for c in ("us", "india")},
               "fit_like": metrics(D.filter(fitlike)), "empty_address": metrics(D.filter("emp")), "non_empty_address": metrics(D.filter(~pl.col("emp")))}
(OUT / "m01_val.json").write_text(json.dumps(R, indent=1))
print(json.dumps(R, indent=1))
