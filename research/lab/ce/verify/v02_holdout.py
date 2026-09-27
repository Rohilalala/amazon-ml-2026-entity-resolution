"""Audit 2+3 (holdout), written independently of stack/s01_stage4.py:
 (2) v7 bundle-report reproduction with src.pipeline._holdout_report (stage 3 'full' and stage 2, no owner rule), and
     v7 production (US stage 2, India stage 3) decided with src.pipeline.decide.
 (3) Stage 4 set A re-scored from the saved model (stack/model/stage4_A.txt) and the saved CE logits
     (infer/out/holdout_ce.parquet) with features re-implemented here in numpy, decided with src.pipeline.decide,
     scored with src.pipeline.macro_f05; compared with s01_main.json and stack/out/holdout_keeps.parquet.
Repo venv, PYTHONDONTWRITEBYTECODE=1. usage: v02_holdout.py -> out/v02_holdout.json"""
import json, os, sys, resource, threading, time
from pathlib import Path
from types import SimpleNamespace
import numpy as np, polars as pl, lightgbm as lgb

ROOT = Path("/Users/adityarohilla/Documents/amazon-ml-2026")
REPO = ROOT / "code/business_entity_resolution"
sys.path.insert(0, str(REPO))
from src import pipeline
CE = ROOT / "v8_lab/ce"; SET = CE / "setup/out"; OUT = Path(__file__).parent / "out"
B = json.load(open(REPO / "artifacts/v7/bundle.json"))
R = {}


def guard(limit=float(os.environ.get("RSS_LIMIT_GB", 3.2)) * 1e9):
    def run():
        while True:
            if resource.getrusage(resource.RUSAGE_SELF).ru_maxrss > limit:
                print("RSS guard tripped", flush=True); os._exit(3)
            time.sleep(0.5)
    threading.Thread(target=run, daemon=True).start()


guard()

H = pl.read_parquet(SET / "holdout_pairs.parquet", columns=["i", "j", "y", "country", "p2", "pf", "prob", "prob_cap", "kept", "kept_cap"])
S = pl.read_parquet(SET / "holdout_s1.parquet").sort("i")  # n_true / country verified against pipeline logic in v01
i, j, y = H["i"].to_numpy(), H["j"].to_numpy(), H["y"].to_numpy()
code = np.searchsorted(S["i"].to_numpy(), i)
assert (S["i"].to_numpy()[code] == i).all()
n_true, s1c = S["n_true"].to_numpy(), S["country"].to_numpy()
pc = s1c[code]
assert (pc == H["country"].to_numpy()).all()


def f05(keep):
    """Macro F0.5 over all 100k holdout S1 (incl. S1 without candidates), overall and by country (pipeline.macro_f05)."""
    out = {"all": pipeline.macro_f05(code, n_true, keep, y)}
    for c in ("us", "india"):
        ent = np.flatnonzero(s1c == c); remap = np.full(len(s1c), -1); remap[ent] = np.arange(len(ent)); m = pc == c
        out[c] = pipeline.macro_f05(remap[code[m]], n_true[ent], keep[m], y[m])
    return {k: round(v, 6) for k, v in out.items()}


def per_s1(keep):
    from src.metric import f05_counts
    n = len(n_true)
    return f05_counts(np.bincount(code, weights=keep, minlength=n), n_true, np.bincount(code, weights=keep & (y == 1), minlength=n))


def decide(prob, rule):
    ns = SimpleNamespace(config=SimpleNamespace(exclusive=True, unseen_none_cap=B["config"]["unseen_none_cap"]), rule=rule,
                         report={"oof_f05_by_country": B["report"]["oof_f05_by_country"]})
    keep, info = pipeline.decide(i, j, prob, pc, s1c, ns)
    assert info == {}
    return keep


# ---- (2) bundle report reproduction: src.pipeline._holdout_report (apply_rule, no owner rule) ----
hold = S.select(pl.col("i").alias("row"), "n_true", pl.col("country").alias("country_key"))
R["bundle_report_reproduced"] = {nm: pipeline._holdout_report(H.select("i", "y"), H[col].to_numpy(), hold, B["rule"])
                                 for nm, col in (("full", "pf"), ("none", "p2"))}
R["bundle_report_original"] = {k: B["report"]["holdout"][k] for k in ("full", "none")}
# ---- v7 production ----
prod = np.where(pc == "us", H["p2"].to_numpy(), H["pf"].to_numpy())
R["prob_col_is_production"] = bool((prod == H["prob"].to_numpy()).all())
k7 = decide(H["prob"].to_numpy(), B["rule"]); k7c = decide(H["prob_cap"].to_numpy(), B["rule"])
R["v7_production"] = f05(k7); R["v7_production_cap"] = f05(k7c)
R["v7_keep_eq_setup_kept"] = [int((k7 != H["kept"].to_numpy()).sum()), int((k7c != H["kept_cap"].to_numpy()).sum())]

# ---- (3) stage 4 set A, own features ----
txt = pl.read_parquet(SET / "text_train.parquet")
assert (txt["row"].to_numpy() == np.arange(txt.height)).all()
emp = txt["text"].str.ends_with(" | ").to_numpy()  # text = name + ' | ' + address, both stripped: ends with ' | ' <=> empty address
emp_s01 = (txt["text"].str.splitn(" | ", 2).struct.field("field_1").fill_null("").str.strip_chars() == "").to_numpy()
R["b_empty"] = {"n": int(emp.sum()), "disagree_with_s01_definition": int((emp != emp_s01).sum())}
del txt
ce = H.select("i", "j").join(pl.read_parquet(CE / "infer/out/holdout_ce.parquet"), on=["i", "j"], how="left", maintain_order="left")["ce_logit"].to_numpy()


def group_feats(g, v):
    """rank ('min', descending) within g, v minus the max of the other members of g (NaN for a singleton), group size."""
    o = np.lexsort((-v, g)); gs, vs = g[o], v[o]
    start = np.r_[True, gs[1:] != gs[:-1]]
    st_idx = np.maximum.accumulate(np.where(start, np.arange(len(g)), 0))
    newv = start | np.r_[True, vs[1:] != vs[:-1]]
    rank = np.maximum.accumulate(np.where(newv, np.arange(len(g)), 0)) - st_idx + 1
    size = np.diff(np.r_[np.flatnonzero(start), len(g)])[np.cumsum(start) - 1]
    nxt = np.r_[vs[1:], np.nan]
    other = np.where(start, np.where(size > 1, nxt, np.nan), vs[st_idx])
    out = [np.empty(len(g)) for _ in range(3)]
    for a, b in zip(out, (rank, vs - other, size)):
        a[o] = b
    return out


booster = lgb.Booster(model_file=str(CE / "stack/model/stage4_A.txt"))
meta = json.load(open(CE / "stack/model/stage4_A.json"))
R["stage4_model"] = {"trees": booster.num_trees(), "features": booster.feature_name(), "rule": meta["rule"]}


def stage4(pcol):
    p = H[pcol].to_numpy().astype(np.float64)
    a = np.flatnonzero(p >= 0.002)
    assert not np.isnan(ce[a]).any()
    q = np.clip(p[a], 1e-6, 1 - 1e-6); lp = np.log(q / (1 - q)); c = ce[a].astype(np.float64)
    ce_rank, ce_gap, n_i = group_feats(i[a], c)
    lp_rank, lp_gap, _ = group_feats(i[a], lp)
    X = np.column_stack([lp, c, emp[j[a]], ce_rank, ce_gap, lp_rank, lp_gap, n_i]).astype(np.float32)
    out = p.copy(); out[a] = booster.predict(X)
    return decide(out, meta["rule"]), len(a)


k4, na = stage4("prob"); k4c, nac = stage4("prob_cap")
R["active"] = [na, nac]
R["stage4_A"] = f05(k4); R["stage4_A_cap"] = f05(k4c)
hk = pl.read_parquet(CE / "stack/out/holdout_keeps.parquet", columns=["i", "j", "stage4_A", "stage4_A_cap"])
assert (hk["i"].to_numpy() == i).all() and (hk["j"].to_numpy() == j).all()
R["stage4_keep_diff_vs_saved"] = [int((k4 != hk["stage4_A"].to_numpy()).sum()), int((k4c != hk["stage4_A_cap"].to_numpy()).sum())]
R["pair_errors"] = {n: {"tp": int((k & (y == 1)).sum()), "fp": int((k & (y == 0)).sum()), "fn_in_cand": int((~k & (y == 1)).sum())}
                    for n, k in (("v7_production", k7), ("stage4_A", k4))}
rng = np.random.default_rng(123)
for nm, a, b in (("stage4_A - v7", k4, k7), ("stage4_A_cap - v7_cap", k4c, k7c)):
    d = per_s1(a) - per_s1(b)
    m = np.concatenate([d[rng.integers(0, len(d), (100, len(d)))].mean(1) for _ in range(20)])
    R[f"diff {nm}"] = {"mean": round(float(d.mean()), 6), "ci95": [round(float(x), 6) for x in np.quantile(m, [0.025, 0.975])],
                       **{c: round(float(d[s1c == c].mean()), 6) for c in ("us", "india")},
                       "s1_better": int((d > 0).sum()), "s1_worse": int((d < 0).sum())}
R["reported_s01"] = {k: json.load(open(CE / "stack/out/s01_main.json"))["holdout_f05"][k] for k in ("v7_production", "v7_production_cap", "stage4_A", "stage4_A_cap", "bundle_report_full_stage3", "bundle_report_none_stage2")}
R["peak_rss_gb"] = round(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1e9, 2)
(OUT / "v02_holdout.json").write_text(json.dumps(R, indent=1))
print(json.dumps(R, indent=1))
