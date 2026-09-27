"""Stage 4 set A retrained on v2 cross-encoder logits, exactly as stack/s01_stage4.py main (same features, params,
5-fold GroupKFold by S1 on STAGE4 active pairs, rule chosen on STAGE4 OOF, refit at the mean best iteration), with the
CE logits read from out/STAGE4_ce.parquet and out/HOLDOUT_ce.parquet (v2) instead of infer/out (v1).
GATE (HOLDOUT, paired): per-S1 F0.5 of v2 stage 4 minus the current stage 4 (stack/out/holdout_keeps.parquet
stage4_A = set A on v1 logits, 0.98881), bootstrap by S1 (2,000 resamples, s01_stage4.boot); keep only if the
difference is >= +0.0003 and its 95% CI excludes 0.
usage (repo venv, PYTHONDONTWRITEBYTECODE=1): s01_v2.py -> stack_model/stage4_A_v2.{txt,json}, out/s01_v2.json,
out/s01_v2_holdout_keeps.parquet"""
import json, os, sys
from pathlib import Path
import numpy as np, polars as pl, lightgbm as lgb
HERE = Path(__file__).parent; CE = HERE.parent; OUT = HERE / "out"; MOD = HERE / "stack_model"; LOGITS = OUT
if os.environ.get("S4_DRY"):  # dry run on v1 logits: must reproduce stack/out/holdout_keeps.parquet stage4_A exactly
    LOGITS, OUT, MOD = CE / "infer/out", OUT / "dry", OUT / "dry/stack_model"; OUT.mkdir(parents=True, exist_ok=True)
sys.path.insert(0, str(CE / "stack"))
import s01_stage4 as s4
from sklearn.metrics import roc_auc_score

GATE = 0.0003


def load_stage4(emp):
    """s01_stage4.load_stage4 with the v2 STAGE4 logits."""
    T = pl.read_parquet(s4.SET / "stage4_train.parquet", columns=["i", "j", "y", "country", "entity", "prob", "active"]).with_columns(pl.col("country").cast(pl.Categorical))
    S = pl.read_parquet(s4.SET / "stage4_s1.parquet")
    code = T.select("i", "entity").unique().sort("entity")
    assert S.join(code, on="i", how="left")["entity"].to_list() == list(range(S.height))
    C = pl.read_parquet(LOGITS / "STAGE4_ce.parquet")
    A = T.filter("active")
    assert (A["i"] == C["i"]).all() and (A["j"] == C["j"]).all()
    return T, S, s4.features(A.with_columns(ce_logit=C["ce_logit"]), emp)


def load_holdout():
    """s01_stage4.load_holdout with the v2 HOLDOUT logits."""
    H = pl.read_parquet(s4.SET / "holdout_pairs.parquet", columns=["i", "j", "y", "country", "prob", "prob_cap", "kept", "kept_cap", "active", "active_cap"])
    S = pl.read_parquet(s4.SET / "holdout_s1.parquet").with_row_index("code")
    H = H.join(pl.read_parquet(LOGITS / "HOLDOUT_ce.parquet"), on=["i", "j"], how="left", maintain_order="left")
    H = H.join(S.select("i", "code"), on="i", how="left", maintain_order="left")
    assert H["code"].null_count() == 0 and H.filter(pl.col("active") | pl.col("active_cap"))["ce_logit"].null_count() == 0
    return H, S


if __name__ == "__main__":
    s4._rss_guard()
    MOD.mkdir(parents=True, exist_ok=True)
    s4.EMP = EMP = s4.empty_address(); s4.log("b_empty", int(EMP.sum()))
    T, S4, A = load_stage4(EMP); s4.log("stage4 active", A.height)
    H, SH = load_holdout(); s4.log("holdout", H.height)
    y, g = A["y"].to_numpy(), A["entity"].to_numpy()
    codes_all, y_all, nt4, act = T["entity"].to_numpy(), T["y"].to_numpy(), S4["n_true"].to_numpy().astype(float), T["active"].to_numpy()
    names = s4.FEAT_A
    x = s4.X(A, names)
    oof, bs = s4.cv(x, y, g)
    p_all = T["prob"].to_numpy().astype(np.float64).copy(); p_all[act] = oof
    f4, rule4, top4 = s4.choose_rule_chunked(codes_all, nt4, p_all, y_all)
    rounds = int(np.mean([b.best_iteration for b in bs]))
    final = lgb.train(s4.PARAMS, lgb.Dataset(x, y), rounds)
    final.save_model(str(MOD / "stage4_A_v2.txt"))
    json.dump({"features": names, "rule": rule4, "rounds": rounds, "ce": "v2_encoder/model"}, open(MOD / "stage4_A_v2.json", "w"), indent=1)
    R = {"params": s4.PARAMS, "stage4_active_pairs": A.height,
         "stage4_A_v2": {"fold_rounds": [b.best_iteration for b in bs], "final_rounds": rounds, "oof_auc_active": round(roc_auc_score(y, oof), 5),
                         "ce_v2_auc_active": round(roc_auc_score(y, A["ce"].to_numpy()), 5), "oof_f05": round(f4, 5), "rule": rule4, "top": top4,
                         "gain": dict(zip(names, final.feature_importance("gain").round(0).tolist()))},
         "stage4_A_v1_reference": {k: json.load(open(CE / "stack/out/s01_main.json"))["stage4_A"][k] for k in ("oof_auc_active", "oof_f05", "rule", "final_rounds")}}
    s4.log("v2 oof", R["stage4_A_v2"])
    ref = pl.read_parquet(CE / "stack/out/holdout_keeps.parquet")
    assert (ref["i"] == H["i"]).all() and (ref["j"] == H["j"]).all()
    keeps = {"v7_production": ref["v7_production"].to_numpy(), "v7_production_cap": ref["v7_production_cap"].to_numpy(),
             "stage4_A_v1": ref["stage4_A"].to_numpy(), "stage4_A_v1_cap": ref["stage4_A_cap"].to_numpy(),
             "stage4_A_v1_foldavg": ref["stage4_A_foldavg"].to_numpy(),
             "stage4_A_v2": s4.holdout_eval(H, SH, rule4, [final], names),
             "stage4_A_v2_cap": s4.holdout_eval(H, SH, rule4, [final], names, cap=True),
             "stage4_A_v2_foldavg": s4.holdout_eval(H, SH, rule4, bs, names)}
    res, f, ctry = s4.score_block(H, SH, keeps)
    R["holdout_f05"] = res
    R["holdout_diff"] = {}
    for k, base in (("stage4_A_v2", "stage4_A_v1"), ("stage4_A_v2_cap", "stage4_A_v1_cap"), ("stage4_A_v2_foldavg", "stage4_A_v1_foldavg"),
                    ("stage4_A_v2", "v7_production"), ("stage4_A_v1", "v7_production")):
        d = f[k] - f[base]
        R["holdout_diff"][f"{k} - {base}"] = {"all": s4.boot(d), **{c: s4.boot(d[ctry == c]) for c in ("us", "india")},
                                              "s1_better": int((d > 0).sum()), "s1_worse": int((d < 0).sum())}
    g0 = R["holdout_diff"]["stage4_A_v2 - stage4_A_v1"]["all"]
    R["gate"] = {"rule": f"stage4_A_v2 - stage4_A_v1 >= +{GATE} overall and CI95 excludes 0 (paired, bootstrap by S1)",
                 "diff": g0["diff"], "ci95": g0["ci95"], "pass": bool(g0["diff"] >= GATE and g0["ci95"][0] > 0)}
    yy = H["y"].to_numpy()
    R["holdout_pair_errors"] = {k: {"tp": int((v & (yy == 1)).sum()), "fp": int((v & (yy == 0)).sum()), "fn_in_cand": int((~v & (yy == 1)).sum())}
                                for k, v in keeps.items()}
    (OUT / "s01_v2.json").write_text(json.dumps(R, indent=1))
    pl.DataFrame({"i": H["i"], "j": H["j"], **{k: keeps[k] for k in ("stage4_A_v2", "stage4_A_v2_cap", "stage4_A_v2_foldavg")}}).write_parquet(OUT / "s01_v2_holdout_keeps.parquet")
    s4.log("holdout", json.dumps(res), "gate", R["gate"])
