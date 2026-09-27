"""HOLDOUT gate evidence: CE-alone macro F0.5 per country vs v7 production, from out/holdout_ce.parquet.
Pairs outside ACTIVE (never scored by the CE; v7 never keeps them either) get CE p = 0.
CE decisions: global threshold (grid of the train step extended to 0.995; picked ON the holdout, so optimistic),
the CE-VAL-picked thresholds (not tuned on the holdout), top-k + threshold, p>=0.5, expected-F0.5 (missed 0) on raw and
prior-corrected p, and the production decision (exclusive owner rule + expected-F0.5, missed 0) on prior-corrected p.
v7 rows = holdout_pairs 'kept' (uncapped) and 'kept_cap' (the stack's regime), recomputed here.
Prior correction: p_pc = sigmoid(logit + delta_logit) with delta from train/out/t03_metrics.json (fit set 63% vs pool 42% positive).
Run with the repo venv: PYTHONDONTWRITEBYTECODE=1 <repo>/.venv/bin/python i02_holdout_eval.py"""
import json, sys
from pathlib import Path
import numpy as np, polars as pl
HERE = Path(__file__).parent; CE = HERE.parent; SET = CE / "setup/out"
sys.path.insert(0, str(CE.parent.parent / "code/business_entity_resolution"))
from src.pipeline import expected_f05_mask, macro_f05
from sklearn.metrics import roc_auc_score

DELTA = json.load(open(CE / "train/out/t03_metrics.json"))["prior"]["delta_logit"]
H = pl.read_parquet(SET / "holdout_pairs.parquet", columns=["i", "j", "country", "y", "prob", "prob_cap", "kept", "kept_cap", "active", "active_cap"])
C = pl.read_parquet(HERE / "out/holdout_ce.parquet")
assert C.height == H.filter(pl.col("active") | pl.col("active_cap")).height
H = H.join(C, on=["i", "j"], how="left")
assert H.filter(pl.col("active") | pl.col("active_cap"))["ce_logit"].null_count() == 0
emp = (pl.scan_parquet(SET / "text_train.parquet").filter(pl.col("row").is_in(C["j"].unique().implode()))
       .select(j="row", b_empty=pl.col("text").str.splitn(" | ", 2).struct.field("field_1").fill_null("").str.strip_chars() == "").collect())
H = H.join(emp, on="j", how="left").with_columns(
    p=pl.when(pl.col("ce_logit").is_null()).then(0.0).otherwise(1 / (1 + (-pl.col("ce_logit").cast(pl.Float64)).exp())),
    p_pc=pl.when(pl.col("ce_logit").is_null()).then(0.0).otherwise(1 / (1 + (-(pl.col("ce_logit").cast(pl.Float64) + DELTA)).exp())))
H = H.with_columns(blend=(pl.col("prob").cast(pl.Float64).clip(1e-6, 1 - 1e-6).pipe(lambda q: (q / (1 - q)).log()) + pl.col("ce_logit") + DELTA) / 2)
H = H.with_columns(prank=pl.col("p").rank("ordinal", descending=True).over("i"))
S = pl.read_parquet(SET / "holdout_s1.parquet")
VAL_T = {"all": 0.95, "us": 0.9, "india": 0.95}  # CE-VAL best global thresholds (train/out/t03_metrics.json)


def owner_expf(i, j, p):
    top = pl.DataFrame({"j": j, "p": p}).select(pl.col("p").max().over("j"))["p"].to_numpy()
    pe = np.where(p >= top, p, 0.0)
    return expected_f05_mask(i, pe, 0.0) & (pe > 0)


def block(h, s, name):
    idx = {x: k for k, x in enumerate(s["i"].to_list())}
    codes = np.array([idx[x] for x in h["i"].to_list()]); nt = s["n_true"].to_numpy().astype(float)
    y, p, pc, rk, i, j = (h[c].to_numpy() for c in ("y", "p", "p_pc", "prank", "i", "j"))
    F = lambda keep: round(macro_f05(codes, nt, keep, y), 5)
    ts = np.round(np.r_[np.arange(0.05, 0.96, 0.05), 0.97, 0.98, 0.99, 0.995], 3)
    thr = max((F(p >= t), float(t)) for t in ts)
    topk = max((F((p >= t) & (rk <= k)), k, float(t)) for k in (1, 2, 3, 4, 6) for t in ts)
    a = h.filter(pl.col("active") | pl.col("active_cap"))
    auc = lambda d, c: round(roc_auc_score(d["y"].to_numpy(), d[c].to_numpy()), 5) if 0 < d["y"].sum() < d.height else None
    return {"s1": s.height, "pairs": h.height, "active_pairs": a.height,
            "v7_production_nocap": F(h["kept"].to_numpy()), "v7_production_cap": F(h["kept_cap"].to_numpy()),
            "ce_production_owner_expf05_prior_corrected": F(owner_expf(i, j, pc)),
            "ce_expected_f05_missed0_prior_corrected": F(expected_f05_mask(codes, pc, 0.0)),
            "ce_expected_f05_missed0_raw": F(expected_f05_mask(codes, p, 0.0)),
            "ce_best_threshold_on_holdout": {"f05": thr[0], "t": thr[1]},
            "ce_ceval_threshold": {"f05": F(p >= VAL_T[name]), "t": VAL_T[name]},
            "ce_best_topk_threshold_on_holdout": {"f05": topk[0], "k": topk[1], "t": topk[2]},
            "ce_threshold_0.5": F(p >= 0.5), "oracle_in_candidates": F(y == 1),
            "auc_active": {"ce": auc(a, "ce_logit"), "v7_prob": auc(a, "prob"), "v7_prob_cap": auc(a, "prob_cap")},
            "auc_active_b_empty": {"pairs": a.filter("b_empty").height, "pos_rate": round(float(a.filter("b_empty")["y"].mean()), 4),
                                   "ce": auc(a.filter("b_empty"), "ce_logit"), "v7_prob": auc(a.filter("b_empty"), "prob")},
            "auc_active_b_nonempty": {"ce": auc(a.filter(~pl.col("b_empty")), "ce_logit"), "v7_prob": auc(a.filter(~pl.col("b_empty")), "prob")},
            # Complementarity, counting only (nothing fitted): v7's production errors among active pairs and what the CE says there.
            # blend = fixed, untuned average of logit(v7 prob) and the prior-corrected CE logit: indicative only, not a decision.
            "v7_errors_vs_ce": {f"{e}_{seg}": {"n": g.height, "ce_pc_ge_0.5": int((g["p_pc"] >= 0.5).sum())}
                                for e, f in (("fp", pl.col("kept") & (pl.col("y") == 0)), ("fn", ~pl.col("kept") & (pl.col("y") == 1)))
                                for seg, sf in (("b_empty", pl.col("b_empty")), ("b_nonempty", ~pl.col("b_empty"))) for g in [a.filter(f & sf)]},
            "auc_active_fixed_blend": {"all": auc(a, "blend"), "b_empty": auc(a.filter("b_empty"), "blend"), "b_nonempty": auc(a.filter(~pl.col("b_empty")), "blend")}}


R = {"delta_logit": DELTA, "all": block(H, S, "all")}
for c in ("us", "india"):
    R[c] = block(H.filter(pl.col("country") == c), S.filter(pl.col("country") == c), c)
json.dump(R, open(HERE / "out/i02_holdout_eval.json", "w"), indent=1)
print(json.dumps(R, indent=1))
