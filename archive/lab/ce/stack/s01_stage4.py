"""Stage 4: LightGBM over v7 production prob + cross-encoder logit + per-record context, then the v7 decision.

Training rows: STAGE4-TRAIN active pairs (v7 prob >= 0.002). v7 prob there is v7's own out-of-fold value; the CE never
saw these S1 (nor their j records). Stage 4 is cross-validated with 5 folds grouped by S1 (sklearn GroupKFold as
src.pipeline.fold_ids, seed 42); the decision rule is chosen by src.pipeline.choose_rule on the out-of-fold stage-4
prob (non-active pairs keep v7's out-of-fold prob). The final model is refit on all STAGE4 active pairs with the
mean best iteration and scores the HOLDOUT, which is used only here, for evaluation.
Decision on the holdout = src.pipeline.decide without the unseen-country part: exclusive owner rule over the holdout
pairs, then the rule; pairs outside ACTIVE keep v7's prob.

Feature sets:
  A (pool-independent): v7 logit, CE logit, empty candidate address, rank and gap to the best other candidate of the
    same S1 for both scores, number of active candidates of the S1.
  B = A + j context: number of active S1 for the same j and the max CE logit of the other S1 for that j. The density
    of competing S1 differs by dataset (STAGE4 500k of 2.2M train S1, HOLDOUT 100k, TEST all 1.73M), so B is
    reported for information only.

usage: python s01_stage4.py main | transfer        (repo venv, PYTHONDONTWRITEBYTECODE=1)"""
import json, sys, time
from pathlib import Path
import numpy as np, polars as pl, lightgbm as lgb
HERE = Path(__file__).parent; CE = HERE.parent; SET = CE / "setup/out"; INF = CE / "infer/out"; OUT = HERE / "out"
REPO = CE.parent.parent / "code/business_entity_resolution"
sys.path.insert(0, str(REPO))
from src.pipeline import choose_rule, apply_rule, macro_f05, THRESHOLDS, MISSED_GRID
from src.metric import f05_counts
from sklearn.model_selection import GroupKFold

CFG = json.load(open(REPO / "artifacts/v7/bundle.json"))["config"]
PARAMS = {**CFG["lgb_params"], "seed": CFG["seed"]}
V7_RULE = json.load(open(REPO / "artifacts/v7/bundle.json"))["rule"]
FOLDS, SEED = 5, CFG["seed"]
FEAT_A = ["lp", "ce", "b_empty", "ce_rank", "ce_gap", "lp_rank", "lp_gap", "n_i"]
FEAT_B = FEAT_A + ["n_j", "j_ce_other"]
T0 = time.time()
PARAMS["num_threads"] = 8  # share the CPU with the other workflow's jobs


def _rss_guard(limit=6e9):
    """Abort before the peak RSS passes the limit (6 GB by default)."""
    import os, resource, threading
    def run():
        while True:
            if resource.getrusage(resource.RUSAGE_SELF).ru_maxrss > limit:  # bytes on macOS
                print(f"RSS guard: peak above {limit / 1e9:.1f} GB, aborting (rerun under heavy.sh)", flush=True); os._exit(3)
            time.sleep(0.5)
    threading.Thread(target=run, daemon=True).start()


def log(*a):
    print(f"[{time.time() - T0:7.1f}s]", *a, flush=True)


def best_other(col, by):
    """Max of `col` over the other rows of the same `by` group (null when the group has one row)."""
    c, top = pl.col(col), pl.col(col).max().over(by)
    n_top = (c == top).sum().over(by)
    second = pl.when(pl.len().over(by) > 1).then(c.top_k(2).min().over(by))
    return pl.when(c < top).then(top).when(n_top > 1).then(top).otherwise(second)


def features(df: pl.DataFrame, emp: np.ndarray) -> pl.DataFrame:
    """df: i, j, prob (v7), ce_logit -> feature columns (active pairs only)."""
    q = pl.col("prob").cast(pl.Float64).clip(1e-6, 1 - 1e-6)
    df = df.with_columns(lp=(q / (1 - q)).log(), ce=pl.col("ce_logit").cast(pl.Float64),
                         b_empty=pl.Series(emp[df["j"].to_numpy()]).cast(pl.Int8))
    return df.with_columns(
        ce_rank=pl.col("ce").rank("min", descending=True).over("i"), ce_gap=pl.col("ce") - best_other("ce", "i"),
        lp_rank=pl.col("lp").rank("min", descending=True).over("i"), lp_gap=pl.col("lp") - best_other("lp", "i"),
        n_i=pl.len().over("i"), n_j=pl.len().over("j"), j_ce_other=best_other("ce", "j"))


def X(df, names):
    return df.select(names).cast(pl.Float32).to_numpy()


def cv(x, y, groups):
    """Out-of-fold prob over 5 folds grouped by S1, early-stopped on the held fold (as src.pipeline.cross_validate)."""
    folds = np.empty(len(y), np.int8)
    for n, (_, va) in enumerate(GroupKFold(n_splits=FOLDS, shuffle=True, random_state=SEED).split(groups, groups=groups)):
        folds[va] = n
    full = lgb.Dataset(x, y, params=PARAMS, free_raw_data=False).construct()
    oof, boosters = np.zeros(len(y)), []
    for n in range(FOLDS):
        tr, va = np.flatnonzero(folds != n), np.flatnonzero(folds == n)
        b = lgb.train(PARAMS, full.subset(tr), CFG["num_boost_round"], valid_sets=[full.subset(va)],
                      callbacks=[lgb.early_stopping(CFG["early_stopping_rounds"], verbose=False)])
        oof[va] = b.predict(x[va], num_iteration=b.best_iteration)
        boosters.append(b)
        log(f"  fold {n + 1}/{FOLDS}: {b.best_iteration} rounds")
    return oof, boosters


def choose_rule_chunked(codes, n_true, prob, y, parts=4):
    """src.pipeline.choose_rule scored over interleaved S1 chunks: per-S1 F0.5 is independent across S1, so the macro
    mean is the same; peak memory is ~1/parts. Same trial grid, same first-best tie order."""
    rules = ([{"kind": "threshold", "threshold": float(t)} for t in THRESHOLDS] +
             [{"kind": "expected_f05", "missed": m} for m in MISSED_GRID])
    tot = np.zeros(len(rules))
    for q in range(parts):
        m = codes % parts == q
        c, p, yy, nt = codes[m] // parts, prob[m], y[m], n_true[q::parts]
        for k, r in enumerate(rules):
            tot[k] += macro_f05(c, nt, apply_rule(r, c, p), yy) * len(nt)
    f = tot / len(n_true)
    best = max(range(len(rules)), key=lambda k: f[k])
    top = sorted(range(len(rules)), key=lambda k: -f[k])[:5]
    return float(f[best]), rules[best], [{"f05": round(float(f[k]), 5), **rules[k]} for k in top]


def decide(i, j, p, rule):
    """src.pipeline.decide for seen countries: exclusive owner rule, then the rule."""
    top = pl.DataFrame({"j": j, "p": p}).select(pl.col("p").max().over("j"))["p"].to_numpy()
    pe = np.where(p >= top, p, 0.0)
    return apply_rule(rule, i, pe) & (pe > 0)


def per_s1_f05(codes, n_true, keep, y):
    n = len(n_true)
    return f05_counts(np.bincount(codes, weights=keep, minlength=n), n_true,
                      np.bincount(codes, weights=keep & (y == 1), minlength=n))


def boot(d, reps=2000, seed=0):
    """Percentile 95% CI of mean(d) resampling S1 with replacement."""
    rng, n, means = np.random.default_rng(seed), len(d), []
    for _ in range(reps // 100):
        means.append(d[rng.integers(0, n, (100, n))].mean(axis=1))
    m = np.concatenate(means)
    return {"diff": round(float(d.mean()), 5), "ci95": [round(float(np.quantile(m, 0.025)), 5), round(float(np.quantile(m, 0.975)), 5)],
            "p_le_0": round(float((m <= 0).mean()), 4)}


def load_stage4(emp):
    T = pl.read_parquet(SET / "stage4_train.parquet", columns=["i", "j", "y", "country", "entity", "prob", "active"]).with_columns(pl.col("country").cast(pl.Categorical))
    S = pl.read_parquet(SET / "stage4_s1.parquet")
    code = T.select("i", "entity").unique().sort("entity")
    assert S.join(code, on="i", how="left")["entity"].to_list() == list(range(S.height))  # stage4_s1 row k <-> entity k
    C = pl.read_parquet(INF / "stage4_ce.parquet")
    A = T.filter("active")
    assert (A["i"] == C["i"]).all() and (A["j"] == C["j"]).all()
    A = features(A.with_columns(ce_logit=C["ce_logit"]), emp)
    return T, S, A


def load_holdout(emp):
    H = pl.read_parquet(SET / "holdout_pairs.parquet", columns=["i", "j", "y", "country", "prob", "prob_cap", "kept", "kept_cap", "active", "active_cap"])
    S = pl.read_parquet(SET / "holdout_s1.parquet").with_row_index("code")
    H = H.join(pl.read_parquet(INF / "holdout_ce.parquet"), on=["i", "j"], how="left", maintain_order="left")
    H = H.join(S.select("i", "code"), on="i", how="left", maintain_order="left")
    assert H["code"].null_count() == 0 and H.filter(pl.col("active") | pl.col("active_cap"))["ce_logit"].null_count() == 0
    return H, S


def empty_address():
    e = (pl.scan_parquet(SET / "text_train.parquet")
         .select("row", b=pl.col("text").str.splitn(" | ", 2).struct.field("field_1").fill_null("").str.strip_chars() == "")
         .collect(engine="streaming"))
    assert (e["row"].to_numpy() == np.arange(e.height)).all()
    return e["b"].to_numpy()


def holdout_eval(H, S, rule, models, names, cap=False):
    """Stage-4 prob on active pairs of the regime (uncapped: prob / active, capped: prob_cap / active_cap),
    v7 prob elsewhere; returns the keep mask."""
    pc, ac = ("prob_cap", "active_cap") if cap else ("prob", "active")
    a = H.with_row_index("r").filter(ac).select("r", "i", "j", prob=pc, ce_logit="ce_logit")
    x = X(features(a, EMP), names)
    p = H[pc].to_numpy().astype(np.float64).copy()
    p[a["r"].to_numpy()] = np.mean([m.predict(x, num_iteration=m.best_iteration or None) for m in models], axis=0)
    return decide(H["i"].to_numpy(), H["j"].to_numpy(), p, rule)


def score_block(H, S, keeps):
    """Macro F0.5 (all/us/india) per decision + per-S1 vectors."""
    codes, y, nt = H["code"].to_numpy(), H["y"].to_numpy(), S["n_true"].to_numpy().astype(float)
    f = {k: per_s1_f05(codes, nt, v, y) for k, v in keeps.items()}
    ctry = S["country"].to_numpy()
    res = {k: {"all": round(float(v.mean()), 5), **{c: round(float(v[ctry == c].mean()), 5) for c in ("us", "india")}} for k, v in f.items()}
    return res, f, ctry


if __name__ == "__main__":
    mode = sys.argv[1]
    _rss_guard()
    EMP = empty_address(); log("b_empty", int(EMP.sum()))
    T, S4, A = load_stage4(EMP); log("stage4 active", A.height)
    H, SH = load_holdout(EMP); log("holdout", H.height)
    y, g = A["y"].to_numpy(), A["entity"].to_numpy()
    codes_all, y_all, nt4 = T["entity"].to_numpy(), T["y"].to_numpy(), S4["n_true"].to_numpy().astype(float)
    act = T["active"].to_numpy()
    R = {"params": PARAMS, "folds": FOLDS, "stage4_active_pairs": A.height, "stage4_active_pos_rate": round(float(y.mean()), 5)}
    if mode == "main":
        f7, rule7, top7 = choose_rule_chunked(codes_all, nt4, T["prob"].to_numpy(), y_all)
        R["v7_production_oof"] = {"f05": round(f7, 5), "rule": rule7, "top": top7}; log("v7 oof", R["v7_production_oof"])
        keeps, models_by = {"v7_production": H["kept"].to_numpy(), "v7_production_cap": H["kept_cap"].to_numpy()}, {}
        # Reproduction of the bundle's holdout report (apply_rule, no owner rule, stage 2 / stage 3 'full' for every S1).
        Hp = pl.read_parquet(SET / "holdout_pairs.parquet", columns=["p2", "pf"])
        for nm, col in (("bundle_report_none_stage2", "p2"), ("bundle_report_full_stage3", "pf")):
            keeps[nm] = apply_rule(V7_RULE, H["i"].to_numpy(), Hp[col].to_numpy())
        for tag, names in (("A", FEAT_A), ("B", FEAT_B)):
            log(f"CV {tag}: {names}")
            x = X(A, names)
            oof, bs = cv(x, y, g)
            p_all = T["prob"].to_numpy().astype(np.float64).copy(); p_all[act] = oof
            f4, rule4, top4 = choose_rule_chunked(codes_all, nt4, p_all, y_all)
            rounds = int(np.mean([b.best_iteration for b in bs]))
            final = lgb.train(PARAMS, lgb.Dataset(x, y), rounds)
            final.save_model(str(HERE / f"model/stage4_{tag}.txt"))
            gain = dict(zip(names, final.feature_importance("gain").round(0).tolist()))
            from sklearn.metrics import roc_auc_score
            R[f"stage4_{tag}"] = {"features": names, "fold_rounds": [b.best_iteration for b in bs], "final_rounds": rounds,
                                  "oof_auc_active": round(roc_auc_score(y, oof), 5), "v7_oof_auc_active": round(roc_auc_score(y, A["prob"].to_numpy()), 5),
                                  "oof_f05": round(f4, 5), "rule": rule4, "top": top4, "gain": gain}
            log(tag, {k: R[f'stage4_{tag}'][k] for k in ('fold_rounds', 'oof_auc_active', 'oof_f05', 'rule')})
            keeps[f"stage4_{tag}"] = holdout_eval(H, SH, rule4, [final], names)
            keeps[f"stage4_{tag}_cap"] = holdout_eval(H, SH, rule4, [final], names, cap=True)
            keeps[f"stage4_{tag}_foldavg"] = holdout_eval(H, SH, rule4, bs, names)
            json.dump({"features": names, "rule": rule4, "rounds": rounds}, open(HERE / f"model/stage4_{tag}.json", "w"), indent=1)
        res, f, ctry = score_block(H, SH, keeps)
        R["holdout_f05"] = res
        R["holdout_diff_vs_v7"] = {}
        for k, base in (("stage4_A", "v7_production"), ("stage4_A_cap", "v7_production_cap"), ("stage4_B", "v7_production"),
                        ("stage4_B_cap", "v7_production_cap"), ("stage4_A_foldavg", "v7_production")):
            d = f[k] - f[base]
            R["holdout_diff_vs_v7"][f"{k} - {base}"] = {"all": boot(d), **{c: boot(d[ctry == c]) for c in ("us", "india")}}
        g0 = R["holdout_diff_vs_v7"]["stage4_A - v7_production"]["all"]
        R["gate"] = {"rule": "stage4_A - v7_production >= +0.0020 overall and CI95 excludes 0",
                     "pass": bool(g0["diff"] >= 0.002 and g0["ci95"][0] > 0)}
        # Pair-level error counts on the holdout.
        yy = H["y"].to_numpy()
        R["holdout_pair_errors"] = {k: {"tp": int((v & (yy == 1)).sum()), "fp": int((v & (yy == 0)).sum()), "fn_in_cand": int((~v & (yy == 1)).sum())}
                                    for k, v in keeps.items()}
        json.dump(R, open(OUT / "s01_main.json", "w"), indent=1)
        pl.DataFrame({"i": H["i"], "j": H["j"], **{k: v for k, v in keeps.items()}}).write_parquet(OUT / "holdout_keeps.parquet")
    else:  # transfer: stage 4 (set A) trained on one country's STAGE4 rows, evaluated on the other country's HOLDOUT
        ctry4 = S4["country"].to_numpy()
        R["transfer"] = {}
        for src, dst in (("us", "india"), ("india", "us")):
            log(f"transfer {src} -> {dst}")
            m = (A["country"] == src).to_numpy()
            x = X(A.filter(m), FEAT_A)
            oof, bs = cv(x, y[m], g[m])
            ent = np.flatnonzero(ctry4 == src); remap = np.full(len(ctry4), -1); remap[ent] = np.arange(len(ent))
            mt = (T["country"] == src).to_numpy()
            p_all = T["prob"].to_numpy().astype(np.float64).copy(); p_all[act & (T["country"] == src).to_numpy()] = oof
            f_src, rule, _ = choose_rule_chunked(remap[codes_all[mt]], nt4[ent], p_all[mt], y_all[mt])
            final = lgb.train(PARAMS, lgb.Dataset(x, y[m]), int(np.mean([b.best_iteration for b in bs])))
            Hd = H.filter(pl.col("country") == dst); Sd = SH.filter(pl.col("country") == dst)
            Hd = Hd.drop("code").join(Sd.drop("code").with_row_index("code"), on="i", how="left", maintain_order="left").drop("country_right", "n_true")
            keeps = {"v7_production": Hd["kept"].to_numpy(), "v7_production_cap": Hd["kept_cap"].to_numpy(),
                     "stage4_A_from_other_country": holdout_eval(Hd, Sd, rule, [final], FEAT_A),
                     "stage4_A_from_other_country_cap": holdout_eval(Hd, Sd, rule, [final], FEAT_A, cap=True)}
            codes, yy, nt = Hd["code"].to_numpy(), Hd["y"].to_numpy(), Sd["n_true"].to_numpy().astype(float)
            fs = {k: per_s1_f05(codes, nt, v, yy) for k, v in keeps.items()}
            R["transfer"][f"{src}->{dst}"] = {"train_oof_f05_src": round(f_src, 5), "rule_from_src": rule,
                                               "fold_rounds": [b.best_iteration for b in bs],
                                               "holdout_f05_dst": {k: round(float(v.mean()), 5) for k, v in fs.items()},
                                               "diff_vs_v7": boot(fs["stage4_A_from_other_country"] - fs["v7_production"]),
                                               "diff_vs_v7_cap": boot(fs["stage4_A_from_other_country_cap"] - fs["v7_production_cap"])}
            log(R["transfer"][f"{src}->{dst}"])
        json.dump(R, open(OUT / "s01_transfer.json", "w"), indent=1)
    log("done")
