"""CE-VAL metrics from out/val_scores.parquet: ROC-AUC, log loss, macro F0.5 when deciding by CE probability alone
(global threshold; top-k + threshold; the production expected-F0.5 rule with missed=0), per country, plus per relation type.
Relation type from the text: house = first number in the address part (same / plus2 = differs by exactly 2 / other /
no_house); name = part before ' | ': eq (identical), core_eq (identical after dropping punctuation and legal-form words),
diff; form_conflict = both names carry a legal-form word and the sets differ.
Run with the repo venv (imports the repo's metric and decision code): PYTHONDONTWRITEBYTECODE=1 <repo>/.venv/bin/python t03_metrics.py"""
import json, sys
from pathlib import Path
import numpy as np, polars as pl
HERE = Path(__file__).parent; CE = HERE.parent
sys.path.insert(0, str(CE.parent.parent / "code/business_entity_resolution"))
from src.pipeline import expected_f05_mask, macro_f05
from sklearn.metrics import roc_auc_score, log_loss

SRC = Path(sys.argv[1]) if len(sys.argv) > 1 else HERE / "out/val_scores.parquet"
V = pl.read_parquet(SRC)
S = pl.read_parquet(CE / "setup/out/ce_s1.parquet").filter(pl.col("is_val"))
assert set(V["i"].unique().to_list()) <= set(S["i"].to_list())
# Prior shift: the j-disjoint fit filter dropped mostly negatives, so the fit set is more positive than CE-TRAIN.
# p_pc = sigmoid(logit + DELTA), DELTA = logit(pos rate of all non-val CE-TRAIN pairs) - logit(pos rate of the fit set).
T = pl.read_parquet(CE / "setup/out/ce_train.parquet", columns=["y", "rank", "is_val", "j_in_stage4_active", "j_in_holdout_active"]).filter(~pl.col("is_val"))
lg = lambda r: float(np.log(r / (1 - r)))
PRIOR = {"pool": float(T["y"].mean()), "fit": float(T.filter(~pl.col("j_in_stage4_active") & ~pl.col("j_in_holdout_active") & ((pl.col("rank") <= 6) | (pl.col("y") == 1)))["y"].mean())}
DELTA = lg(PRIOR["pool"]) - lg(PRIOR["fit"])
V = V.with_columns(p_pc=1 / (1 + (-(pl.col("logit").cast(pl.Float64) + DELTA)).exp()))
FORMS = r"\b(inc|incorporated|llc|ltd|limited|corp|corporation|co|company|pvt|private|llp|lp|plc|pc|pllc)\b"
core = lambda c: pl.col(c).str.replace_all(r"[^\w\s]", " ").str.replace_all(FORMS, " ").str.replace_all(r"\s+", " ").str.strip_chars()
forms = lambda c: pl.col(c).str.replace_all(r"[^\w\s]", " ").str.extract_all(FORMS).list.unique().list.sort()
parts = lambda c, k: pl.col(c).str.splitn(" | ", 2).struct.field(k)
V = V.with_columns(na=parts("a", "field_0"), nb=parts("b", "field_0"),
                   ha=parts("a", "field_1").str.extract(r"(\d+)", 1).cast(pl.Int64, strict=False),
                   hb=parts("b", "field_1").str.extract(r"(\d+)", 1).cast(pl.Int64, strict=False))
V = V.with_columns(house=pl.when(pl.col("ha").is_null() | pl.col("hb").is_null()).then(pl.lit("no_house"))
                   .when(pl.col("ha") == pl.col("hb")).then(pl.lit("same"))
                   .when((pl.col("ha") - pl.col("hb")).abs() == 2).then(pl.lit("plus2"))
                   .otherwise(pl.lit("other")),
                   ca=core("na"), cb=core("nb"), fa=forms("na"), fb=forms("nb"))
V = V.with_columns(name=pl.when(pl.col("na") == pl.col("nb")).then(pl.lit("eq")).when(pl.col("ca") == pl.col("cb")).then(pl.lit("core_eq")).otherwise(pl.lit("diff")),
                   form_conflict=(pl.col("fa").list.len() > 0) & (pl.col("fb").list.len() > 0) & (pl.col("fa") != pl.col("fb")),
                   prank=pl.col("p").rank("ordinal", descending=True).over("i"))

def f05_block(v, s):
    idx = {x: k for k, x in enumerate(s["i"].to_list())}
    codes = np.array([idx[x] for x in v["i"].to_list()]); nt = s["n_true"].to_numpy().astype(float)
    y, p, rk = v["y"].to_numpy(), v["p"].to_numpy(), v["prank"].to_numpy()
    F = lambda keep: round(macro_f05(codes, nt, keep, y), 5)
    ts = np.round(np.arange(0.05, 0.96, 0.05), 2)
    thr = max(((F(p >= t), float(t)) for t in ts))
    topk = max(((F((p >= t) & (rk <= k)), k, float(t)) for k in (1, 2, 3, 4, 6) for t in ts))
    return {"s1": s.height, "oracle_in_candidates": F(y == 1), "best_threshold": {"f05": thr[0], "t": thr[1]},
            "best_topk_threshold": {"f05": topk[0], "k": topk[1], "t": topk[2]},
            "expected_f05_missed0": F(expected_f05_mask(codes, p, 0.0)), "threshold_0.5": F(p >= 0.5),
            "expected_f05_missed0_prior_corrected": F(expected_f05_mask(codes, v["p_pc"].to_numpy(), 0.0))}

def cls(v):
    y, p = v["y"].to_numpy(), v["p"].to_numpy()
    r = {"pairs": v.height, "pos_rate": round(float(y.mean()), 4)}
    if 0 < y.sum() < len(y):
        r |= {"auc": round(roc_auc_score(y, v["logit"].to_numpy()), 5), "auc_retrieval_score": round(roc_auc_score(y, v["score"].to_numpy()), 5)}
    r |= {"logloss": round(log_loss(y, np.clip(p, 1e-7, 1 - 1e-7), labels=[0, 1]), 5),
          "logloss_prior_corrected": round(log_loss(y, np.clip(v["p_pc"].to_numpy(), 1e-7, 1 - 1e-7), labels=[0, 1]), 5),
          "fp_rate_at_0.5": round(float((p[y == 0] >= 0.5).mean()), 4) if (y == 0).any() else None,
          "fn_rate_at_0.5": round(float((p[y == 1] < 0.5).mean()), 4) if (y == 1).any() else None}
    return r

R = {"prior": PRIOR | {"delta_logit": round(DELTA, 4)}, "overall": cls(V) | f05_block(V, S)}
for c in ("us", "india"):
    R[c] = cls(V.filter(pl.col("country") == c)) | f05_block(V.filter(pl.col("country") == c), S.filter(pl.col("country") == c))
R["relation"] = {f"{h}/{n}": cls(V.filter((pl.col("house") == h) & (pl.col("name") == n)))
                 for h in ("same", "plus2", "other", "no_house") for n in ("eq", "core_eq", "diff")}
R["form_conflict"] = {h: {str(f): cls(V.filter((pl.col("house") == h) & (pl.col("form_conflict") == f))) for f in (True, False)} for h in ("same", "plus2", "other", "no_house")}
R["relation_by_country"] = {c: {f"{h}/{n}": cls(V.filter((pl.col("country") == c) & (pl.col("house") == h) & (pl.col("name") == n)))
                                for h in ("same", "plus2", "other", "no_house") for n in ("eq", "core_eq", "diff")} for c in ("us", "india")}
# CE-TRAIN fit excluded pairs whose j also sits in STAGE4/HOLDOUT active pairs (j-disjoint) and rank>6 negatives;
# CE-VAL keeps them, so split CE-VAL into the fit-like part and the parts the fit set never showed.
jf = pl.col("j_in_stage4_active") | pl.col("j_in_holdout_active")
R["val_subset"] = {"fit_like": cls(V.filter(~jf & ((pl.col("rank") <= 6) | (pl.col("y") == 1)))),
                   "j_in_stage4_or_holdout_active": cls(V.filter(jf)),
                   "rank_gt6_not_j_flagged": cls(V.filter(~jf & (pl.col("rank") > 6))),
                   "b_address_empty": cls(V.filter(parts("b", "field_1").str.strip_chars() == "")),
                   "b_address_nonempty": cls(V.filter(parts("b", "field_1").str.strip_chars() != ""))}
fp = V.filter((pl.col("p") >= 0.5) & (pl.col("y") == 0))
R["fp_at_0.5"] = {"n": fp.height, "j_flagged": int(fp.select(jf).to_series().sum()), "b_address_empty": int((fp.select(parts("b", "field_1").str.strip_chars() == "").to_series()).sum())}
json.dump(R, open(SRC.parent / "t03_metrics.json", "w"), indent=1)
print(json.dumps(R["val_subset"] | {"fp_at_0.5": R["fp_at_0.5"]}, indent=1))
print(json.dumps({k: R[k] for k in ("prior", "overall", "us", "india")}, indent=1))
for k, r in R["relation"].items(): print(k, r)
for h, d in R["form_conflict"].items(): print("form_conflict", h, d["True"])
