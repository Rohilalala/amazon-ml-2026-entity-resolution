"""Recall head: decide the dense-retrieval NEW pairs (outside the sparse candidate set) with a small LightGBM.

Train: HEAD new pairs (200k train S1 outside the v7 500k sample and the 100k holdout), labels from truth_train.
  5-fold GroupKFold by S1 -> OOF prob; AUC and precision at thresholds on OOF; refit on all HEAD at the mean best round.
Decision: existing keeps stay; a new pair is added when prob >= t (t_keep for S1 with >= 1 existing keep, t_none for
  S1 with none), its j is not kept by any S1 of the existing decision, and it has the highest prob among the added
  pairs of that j. (t_keep, t_none) is chosen on HEAD OOF by macro F0.5 of the combined decision, where the existing
  keeps of a HEAD S1 are proxied by its true pairs inside the sparse candidates (a perfect in-candidate matcher).
Evaluate: HOLDOUT once, on top of the best file's holdout keeps (ACNRST_3s_v2ce, 0.989291), paired bootstrap by S1.
features() is the only feature code; it runs unchanged on TEST (text_test) given the same pair-list columns + ce_logit.
usage (repo venv): h01_head.py -> out/h01.json, model/head.txt, model/head.json, out/{head_oof,holdout_add}.parquet"""
import json, resource, sys, time
from pathlib import Path
import numpy as np, polars as pl, lightgbm as lgb
from rapidfuzz import fuzz
from rapidfuzz.process import cpdist
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import GroupKFold

ROOT = Path("/Users/adityarohilla/Documents/amazon-ml-2026"); V8 = ROOT / "v8_lab"; SET = V8 / "ce/setup/out"
HERE = Path(__file__).parent; OUT = HERE / "out"; MOD = HERE / "model"; PAIRS = V8 / "recall/retrieve/out/pairs"
sys.path.insert(0, str(V8 / "ce/stack"))
import s01_stage4 as s4  # best_other, boot, _rss_guard; src.metric via its repo path
from src.metric import f05_counts

FEATS = ["ce", "ce_rank", "ce_gap", "cos", "cos_gap", "drank", "nrank", "rrank", "rev_only", "src", "n_new",
         "e_name_i", "e_addr_i", "e_name_j", "e_addr_j", "name_ratio", "name_tset", "addr_ratio", "addr_tset"]
PARAMS = {"objective": "binary", "learning_rate": 0.05, "num_leaves": 31, "min_data_in_leaf": 100, "feature_fraction": 0.8,
          "bagging_fraction": 0.8, "bagging_freq": 1, "lambda_l2": 1.0, "deterministic": True, "force_row_wise": True,
          "verbose": -1, "seed": 42, "num_threads": 6}
GRID = [round(x, 2) for x in np.arange(0.20, 0.96, 0.05)] + [0.97, 0.98, 0.99, 1.01]  # 1.01 = add nothing
T0 = time.time()


def log(*a):
    print(f"[{time.time() - T0:6.1f}s]", *a, flush=True)


def split_text(col):
    s = pl.col(col).fill_null("").str.splitn(" | ", 2)
    return s.struct.field("field_0").fill_null("").str.strip_chars(), s.struct.field("field_1").fill_null("").str.strip_chars()


def features(df: pl.DataFrame, text_parquet) -> pl.DataFrame:
    """df: s1_row, j_row, dense_cos, dense_rank, new_rank, rev_rank, source, ce_logit (one S1's whole new-pair list
    present) -> df + FEATS. Text = lower(name) + ' | ' + lower(address) from setup/out/text_{train,test}.parquet."""
    need = pl.concat([df["s1_row"], df["j_row"]]).unique()
    T = pl.scan_parquet(text_parquet).filter(pl.col("row").is_in(need.implode())).collect()
    df = (df.join(T.rename({"row": "s1_row", "text": "ti"}), on="s1_row", how="left", maintain_order="left")
            .join(T.rename({"row": "j_row", "text": "tj"}), on="j_row", how="left", maintain_order="left"))
    ni, ai = split_text("ti"); nj, aj = split_text("tj")
    df = df.with_columns(ni=ni, ai=ai, nj=nj, aj=aj).drop("ti", "tj")
    sim = {k: cpdist(df[a].to_list(), df[b].to_list(), scorer=sc, workers=6).astype(np.float32)
           for k, a, b, sc in (("name_ratio", "ni", "nj", fuzz.ratio), ("name_tset", "ni", "nj", fuzz.token_set_ratio),
                               ("addr_ratio", "ai", "aj", fuzz.ratio), ("addr_tset", "ai", "aj", fuzz.token_set_ratio))}
    df = df.with_columns(ce=pl.col("ce_logit").cast(pl.Float64), cos=pl.col("dense_cos").cast(pl.Float64),
                         drank=pl.col("dense_rank"), nrank=pl.col("new_rank"), rrank=pl.col("rev_rank"),
                         rev_only=pl.col("dense_rank").is_null().cast(pl.Int8), src=pl.col("source"),
                         e_name_i=(pl.col("ni") == "").cast(pl.Int8), e_addr_i=(pl.col("ai") == "").cast(pl.Int8),
                         e_name_j=(pl.col("nj") == "").cast(pl.Int8), e_addr_j=(pl.col("aj") == "").cast(pl.Int8),
                         **{k: pl.Series(v) for k, v in sim.items()}).drop("ni", "ai", "nj", "aj")
    return df.with_columns(ce_rank=pl.col("ce").rank("min", descending=True).over("s1_row"),
                           ce_gap=pl.col("ce") - s4.best_other("ce", "s1_row"),
                           cos_gap=pl.col("cos") - s4.best_other("cos", "s1_row"), n_new=pl.len().over("s1_row"))


def X(df):
    return df.select(FEATS).cast(pl.Float32).to_numpy()


def load(name):
    P = pl.read_parquet(PAIRS / f"{name}_m1_r1_cap10.parquet")
    C = pl.read_parquet(OUT / f"{name.upper()}_ce.parquet")
    assert (P["s1_row"] == C["i"]).all() and (P["j_row"] == C["j"]).all() and C["ce_logit"].is_finite().all()
    return features(P.with_columns(ce_logit=C["ce_logit"]), SET / "text_train.parquet")


def add_new(code, j, p, thr_by_s1, j_taken):
    """Mask of added new pairs: p >= the S1's threshold, j not kept by the existing decision, top prob per j."""
    cand = (p >= thr_by_s1[code]) & ~j_taken
    idx = np.flatnonzero(cand)
    idx = idx[np.lexsort((idx, -p[idx]))]  # prob desc, then row order (deterministic ties)
    _, first = np.unique(j[idx], return_index=True)
    m = np.zeros(len(p), bool); m[idx[first]] = True
    return m


def macro(n_true, kept0, tp0, code, add, y):
    n = len(n_true)
    return f05_counts(kept0 + np.bincount(code, weights=add, minlength=n), n_true,
                      tp0 + np.bincount(code, weights=add & (y == 1), minlength=n))


def choose(n_true, kept0, tp0, code, j, p, y, j_taken, has_keep):
    base = f05_counts(kept0, n_true, tp0).mean(); best = (-1, None); single = (-1, None); table = []
    for tk in GRID:
        for tn in GRID:
            thr = np.where(has_keep, tk, tn)
            f = macro(n_true, kept0, tp0, code, add_new(code, j, p, thr, j_taken), y).mean()
            table.append((tk, tn, f))
            if f > best[0]: best = (f, (tk, tn))
            if tk == tn and f > single[0]: single = (f, tk)
    return base, best, single, table


def main():
    s4._rss_guard()
    R = {"params": PARAMS, "feats": FEATS}
    # ---------- HEAD ----------
    head_rows = np.load(V8 / "recall/retrieve/out/sets/head_rows.npy")
    H = load("head"); log("head pairs", H.height, "pos", int(H["y"].sum()))
    truth = pl.read_parquet(SET / "truth_train.parquet")
    th = truth.filter(pl.col("i").is_in(pl.Series(head_rows).implode()))
    cand = (pl.scan_parquet(ROOT / "cache/58f0a60c5c264fdd/pairs.parquet").select("i", "j")
            .filter(pl.col("i").is_in(pl.Series(head_rows).implode())).collect())
    th = th.join(cand.with_columns(inc=pl.lit(True)), on=["i", "j"], how="left").with_columns(pl.col("inc").fill_null(False))
    assert H.join(cand, left_on=["s1_row", "j_row"], right_on=["i", "j"], how="semi").height == 0
    nh = len(head_rows); tcode = np.searchsorted(head_rows, th["i"].to_numpy())
    n_true_h = np.bincount(tcode, minlength=nh).astype(float)
    kept0_h = np.bincount(tcode, weights=th["inc"].to_numpy(), minlength=nh)
    # new-pair trues must be exactly the missed trues' subset
    assert H.filter(pl.col("y") == 1).join(th.filter(~pl.col("inc")), left_on=["s1_row", "j_row"], right_on=["i", "j"], how="anti").height == 0
    ce_s1 = pl.concat([pl.read_parquet(SET / "ce_s1.parquet", columns=["i"]), pl.read_parquet(V8 / "ce/v2_encoder/out/d01_s1.parquet", columns=["i"])])["i"]
    seen = H["s1_row"].is_in(ce_s1.implode()).to_numpy()
    x, y = X(H), H["y"].to_numpy().astype(np.int8); g = H["s1_row"].to_numpy()
    R["head"] = {"s1": nh, "pairs": H.height, "pos": int(y.sum()), "n_true": int(n_true_h.sum()), "missed": int((~th["inc"]).sum()),
                 "s1_ce_trained_share": round(float(np.isin(head_rows, ce_s1.to_numpy()).mean()), 4),
                 "ce_auc_ce_trained_s1": round(roc_auc_score(y[seen], H["ce"].to_numpy()[seen]), 5),
                 "ce_auc_other_s1": round(roc_auc_score(y[~seen], H["ce"].to_numpy()[~seen]), 5),
                 "pos_rate_ce_trained_s1": round(float(y[seen].mean()), 5), "pos_rate_other_s1": round(float(y[~seen].mean()), 5)}
    log(R["head"])
    # ---------- CV ----------
    folds = np.empty(len(y), np.int8)
    for n, (_, va) in enumerate(GroupKFold(n_splits=5, shuffle=True, random_state=42).split(x, groups=g)):
        folds[va] = n
    ds = lgb.Dataset(x, y, feature_name=FEATS, params=PARAMS, free_raw_data=False).construct()
    oof, rounds = np.zeros(len(y)), []
    for n in range(5):
        tr, va = np.flatnonzero(folds != n), np.flatnonzero(folds == n)
        b = lgb.train(PARAMS, ds.subset(tr), 3000, valid_sets=[ds.subset(va)], callbacks=[lgb.early_stopping(100, verbose=False)])
        oof[va] = b.predict(x[va], num_iteration=b.best_iteration); rounds.append(b.best_iteration)
        log(f"fold {n}: {b.best_iteration} rounds, auc {roc_auc_score(y[va], oof[va]):.5f}")
    R["cv"] = {"rounds": rounds, "oof_auc": round(roc_auc_score(y, oof), 5), "ce_only_auc": round(roc_auc_score(y, H["ce"].to_numpy()), 5),
               "oof_auc_ce_trained_s1": round(roc_auc_score(y[seen], oof[seen]), 5), "oof_auc_other_s1": round(roc_auc_score(y[~seen], oof[~seen]), 5),
               "oof_auc_by_country": {c: round(roc_auc_score(y[m], oof[m]), 5) for c in ("us", "india") for m in [(H["country"] == c).to_numpy()]},
               "prec_at": [{"t": t, "kept": int((oof >= t).sum()), "tp": int(((oof >= t) & (y == 1)).sum()),
                            "precision": round(float(y[oof >= t].mean()), 4) if (oof >= t).any() else None,
                            "recall_of_new_trues": round(float(((oof >= t) & (y == 1)).sum() / y.sum()), 4)}
                           for t in (0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 0.95)]}
    log(R["cv"])
    # ---------- threshold on HEAD OOF ----------
    hcode = np.searchsorted(head_rows, g)
    j_taken_h = H["j_row"].is_in(th.filter("inc")["j"].implode()).to_numpy()
    base, best, single, table = choose(n_true_h, kept0_h, kept0_h, hcode, H["j_row"].to_numpy(), oof, y, j_taken_h, kept0_h > 0)
    tk, tn = best[1]
    thr_h = np.where(kept0_h > 0, tk, tn); add_h = add_new(hcode, H["j_row"].to_numpy(), oof, thr_h, j_taken_h)
    R["head_decision"] = {"proxy_existing": "HEAD S1's true pairs inside the sparse candidates (perfect in-candidate matcher)",
                          "base_f05": round(base, 6), "best": {"t_keep": tk, "t_none": tn, "f05": round(best[0], 6), "gain": round(best[0] - base, 6)},
                          "single_t": {"t": single[1], "f05": round(single[0], 6), "gain": round(single[0] - base, 6)},
                          "ceiling_gain_all_new_trues": round(float(macro(n_true_h, kept0_h, kept0_h, hcode, (y == 1) & ~j_taken_h, y).mean() - base), 6),
                          "added": int(add_h.sum()), "added_true": int((add_h & (y == 1)).sum()), "j_taken_new_pairs": int(j_taken_h.sum()),
                          "top10": sorted(({"t_keep": a, "t_none": b, "f05": round(f, 6)} for a, b, f in table), key=lambda r: -r["f05"])[:10]}
    log(R["head_decision"])
    pl.DataFrame({"s1_row": g, "j_row": H["j_row"], "y": y, "oof": oof, "fold": folds, "ce_trained_s1": seen}).write_parquet(OUT / "head_oof.parquet")
    # ---------- refit ----------
    nr = int(round(np.mean(rounds)))
    model = lgb.train(PARAMS, ds, nr)
    MOD.mkdir(exist_ok=True); model.save_model(str(MOD / "head.txt"))
    json.dump({"features": FEATS, "rounds": nr, "params": PARAMS, "t_keep": tk, "t_none": tn,
               "decision": "add new pair if prob >= (t_keep if the S1 keeps >= 1 existing pair else t_none), its j is kept by no S1 "
                           "in the existing decision, and it has the highest prob among the added pairs of that j",
               "pair_list": "recall/retrieve/out/pairs/*_m1_r1_cap10.parquet", "ce": "ce/v2_encoder/model via c01_score.py"},
              open(MOD / "head.json", "w"), indent=1)
    imp = dict(zip(FEATS, model.feature_importance("gain").round(1).tolist()))
    R["refit"] = {"rounds": nr, "gain": dict(sorted(imp.items(), key=lambda kv: -kv[1]))}
    # ---------- HOLDOUT (evaluation only) ----------
    V = load("holdout"); log("holdout pairs", V.height, "pos", int(V["y"].sum()))
    S = pl.read_parquet(SET / "holdout_s1.parquet").sort("i"); rows = S["i"].to_numpy(); assert (np.diff(rows.astype(np.int64)) > 0).all()
    K = pl.read_parquet(V8 / "ce/v2_build/out/holdout_keeps.parquet", columns=["i", "j", "ACNRST_3s_v2ce"])
    Hp = pl.read_parquet(SET / "holdout_pairs.parquet", columns=["i", "j", "y"])
    assert (K["i"] == Hp["i"]).all() and (K["j"] == Hp["j"]).all()
    kc, keep, ky = np.searchsorted(rows, K["i"].to_numpy()), K["ACNRST_3s_v2ce"].to_numpy(), Hp["y"].to_numpy()
    n_true = S["n_true"].to_numpy().astype(float)
    kept0 = np.bincount(kc, weights=keep, minlength=len(rows)); tp0 = np.bincount(kc, weights=keep & (ky == 1), minlength=len(rows))
    f0 = f05_counts(kept0, n_true, tp0)
    assert round(float(f0.mean()), 6) == 0.989291, f0.mean()
    vcode = np.searchsorted(rows, V["s1_row"].to_numpy()); vy = V["y"].to_numpy()
    assert (rows[vcode] == V["s1_row"].to_numpy()).all()
    pv = model.predict(X(V))
    j_taken_v = V["j_row"].is_in(K.filter("ACNRST_3s_v2ce")["j"].implode()).to_numpy()
    add_v = add_new(vcode, V["j_row"].to_numpy(), pv, np.where(kept0 > 0, tk, tn), j_taken_v)
    f1 = macro(n_true, kept0, tp0, vcode, add_v, vy)
    ctry = S["country"].to_numpy(); d = f1 - f0
    fceil = macro(n_true, kept0, tp0, vcode, (vy == 1) & ~j_taken_v, vy)
    R["holdout"] = {"base_f05": {"all": round(float(f0.mean()), 6), **{c: round(float(f0[ctry == c].mean()), 6) for c in ("us", "india")}},
                    "new_f05": {"all": round(float(f1.mean()), 6), **{c: round(float(f1[ctry == c].mean()), 6) for c in ("us", "india")}},
                    "diff": {"all": s4.boot(d), **{c: s4.boot(d[ctry == c]) for c in ("us", "india")}},
                    "s1_better_worse": [int((d > 0).sum()), int((d < 0).sum())],
                    "added": int(add_v.sum()), "added_true": int((add_v & (vy == 1)).sum()),
                    "added_by_path": {k: [int((add_v & m).sum()), int((add_v & m & (vy == 1)).sum())] for k, m in
                                      (("reverse_only", V["rev_only"].to_numpy() == 1), ("forward", V["rev_only"].to_numpy() == 0))},
                    "added_to_s1_with_keeps": int((add_v & (kept0[vcode] > 0)).sum()),
                    "auc_new_pairs": round(roc_auc_score(vy, pv), 5), "ce_only_auc": round(roc_auc_score(vy, V["ce"].to_numpy()), 5),
                    "j_taken_new_pairs": int(j_taken_v.sum()), "new_trues_with_j_taken": int((j_taken_v & (vy == 1)).sum()),
                    "ceiling_gain_all_new_trues_j_free": round(float(fceil.mean() - f0.mean()), 6)}
    R["gate"] = {"need": 0.0005, "pass": bool(R["holdout"]["diff"]["all"]["diff"] >= 0.0005 and R["holdout"]["diff"]["all"]["ci95"][0] > 0)}
    log(R["holdout"], R["gate"])
    pl.DataFrame({"s1_row": V["s1_row"], "j_row": V["j_row"], "y": vy, "prob": pv, "add": add_v}).write_parquet(OUT / "holdout_add.parquet")
    R["peak_rss_gb"] = round(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1e9, 2); R["seconds"] = round(time.time() - T0)
    (OUT / "h01.json").write_text(json.dumps(R, indent=1))


if __name__ == "__main__":
    main()
