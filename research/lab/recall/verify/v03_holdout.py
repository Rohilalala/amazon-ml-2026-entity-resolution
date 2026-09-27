"""Recompute the holdout gate from saved scores with own code.
Base = best file's holdout keeps (ce/v2_build/out/holdout_keeps.parquet, ACNRST_3s_v2ce). New = base + adds decided from
the saved holdout head probs (recall/head/out/holdout_add.parquet) with an own replay of the rule. Labels re-derived from
truth_train. Macro F0.5 over the 100k holdout S1 (singletons: 1 if nothing kept, else 0). Paired bootstrap by S1,
own seed. Also: holdout probs re-predicted with model/head.txt; threshold sensitivity (reported only). -> out/v03.json"""
import json, resource, sys, time
import numpy as np, polars as pl, lightgbm as lgb

R = "/Users/adityarohilla/Documents/amazon-ml-2026"; V8 = f"{R}/v8_lab"; SET = f"{V8}/ce/setup/out"; O = f"{V8}/recall/verify/out"
T0 = time.time(); SEED, NB = 918273, 2000


def f05(npred, ntrue, tp):
    out = np.zeros(len(ntrue))
    z = ntrue == 0; out[z] = (npred[z] == 0)
    m = ~z & (tp > 0); p, r = tp[m] / npred[m], tp[m] / ntrue[m]
    out[m] = 1.25 * p * r / (0.25 * p + r)
    return out


def boot(d, rng):
    n = len(d); s = np.empty(NB)
    for b in range(NB): s[b] = d[rng.integers(0, n, n)].mean()
    return {"diff": round(float(d.mean()), 6), "ci95": [round(float(np.quantile(s, 0.025)), 6), round(float(np.quantile(s, 0.975)), 6)],
            "p_le_0": float((s <= 0).mean())}


S = pl.read_parquet(f"{SET}/holdout_s1.parquet").sort("i"); rows = S["i"].to_numpy(); n = len(rows)
truth = pl.read_parquet(f"{SET}/truth_train.parquet").filter(pl.col("i").is_in(S["i"].implode())).with_columns(y=pl.lit(1, pl.Int8))
nt = np.bincount(np.searchsorted(rows, truth["i"].to_numpy()), minlength=n).astype(float)
chk = {"n_true_matches_truth": bool((nt == S["n_true"].to_numpy()).all())}
K = pl.read_parquet(f"{V8}/ce/v2_build/out/holdout_keeps.parquet", columns=["i", "j", "ACNRST_3s_v2ce"]).filter("ACNRST_3s_v2ce").select("i", "j")
K = K.join(truth, on=["i", "j"], how="left").with_columns(pl.col("y").fill_null(0))
kc = np.searchsorted(rows, K["i"].to_numpy()); assert (rows[kc] == K["i"].to_numpy()).all()
kept0 = np.bincount(kc, minlength=n).astype(float); tp0 = np.bincount(kc, weights=K["y"].to_numpy(), minlength=n)
f0 = f05(kept0, nt, tp0)
A = pl.read_parquet(f"{V8}/recall/head/out/holdout_add.parquet").with_row_index("idx")
A = A.join(truth.rename({"i": "s1_row", "j": "j_row", "y": "yt"}), on=["s1_row", "j_row"], how="left").with_columns(pl.col("yt").fill_null(0))
chk["add_labels_match_truth"] = bool((A["y"] == A["yt"]).all())
cache = (pl.scan_parquet(f"{R}/cache/58f0a60c5c264fdd/pairs.parquet").select("i", "j").filter(pl.col("i").is_in(S["i"].implode())).collect())
chk["new_pairs_in_sparse_candidates"] = A.join(cache, left_on=["s1_row", "j_row"], right_on=["i", "j"], how="semi").height
chk["keeps_outside_sparse_candidates"] = K.join(cache, on=["i", "j"], how="anti").height
chk["new_pairs_s1_outside_holdout"] = int((~A["s1_row"].is_in(S["i"].implode())).sum())
del cache
code = np.searchsorted(rows, A["s1_row"].to_numpy()); y = A["yt"].to_numpy(); p = A["prob"].to_numpy()
taken = A["j_row"].is_in(K["j"].implode()).to_numpy()


def decide(tk, tn):
    thr = np.where(kept0[code] > 0, tk, tn)
    d = A.with_columns(ok=pl.Series((p >= thr) & ~taken)).filter("ok").sort(["j_row", "prob", "idx"], descending=[False, True, False])
    m = np.zeros(A.height, bool); m[d.unique("j_row", keep="first")["idx"].to_numpy()] = True
    return m


meta = json.load(open(f"{V8}/recall/head/model/head.json")); TK, TN = meta["t_keep"], meta["t_none"]
add = decide(TK, TN)
chk["replayed_adds_equal_saved_add"] = bool((add == A["add"].to_numpy()).all())
f1 = f05(kept0 + np.bincount(code, weights=add, minlength=n), nt, tp0 + np.bincount(code, weights=add & (y == 1), minlength=n))
d = f1 - f0; ct = S["country"].to_numpy(); rng = np.random.default_rng(SEED)
out = {"seed": SEED, "resamples": NB, "checks": chk, "t_keep": TK, "t_none": TN,
       "base": {"all": round(float(f0.mean()), 6), **{c: round(float(f0[ct == c].mean()), 6) for c in ("us", "india")}},
       "new": {"all": round(float(f1.mean()), 6), **{c: round(float(f1[ct == c].mean()), 6) for c in ("us", "india")}},
       "diff": {"all": boot(d, rng), **{c: boot(d[ct == c], rng) for c in ("us", "india")}},
       "s1_better_worse": [int((d > 0).sum()), int((d < 0).sum())],
       "added": int(add.sum()), "added_true": int((add & (y == 1)).sum()),
       "added_by_country": {c: [int((add & (ct[code] == c)).sum()), int((add & (ct[code] == c) & (y == 1)).sum())] for c in ("us", "india")}}
out["gate_pass"] = out["diff"]["all"]["diff"] >= 0.0005 and out["diff"]["all"]["ci95"][0] > 0
print(json.dumps(out), flush=True)
# sensitivity (reported only; the rule was fixed before this)
sens = {}
for tk, tn in ((0.5, 0.5), (0.6, 0.6), (0.7, 0.7), (0.75, 0.75), (0.8, 0.75), (0.8, 0.8), (0.85, 0.85), (0.9, 0.9), (0.95, 0.95)):
    m = decide(tk, tn)
    g = f05(kept0 + np.bincount(code, weights=m, minlength=n), nt, tp0 + np.bincount(code, weights=m & (y == 1), minlength=n)).mean() - f0.mean()
    sens[f"{tk}/{tn}"] = round(float(g), 6)
out["holdout_gain_by_threshold"] = sens
# holdout probs vs the shipped model (features() = the head's own feature code)
sys.path.insert(0, f"{V8}/recall/head"); import h01_head as h
PLs = pl.read_parquet(f"{V8}/recall/retrieve/out/pairs/holdout_m1_r1_cap10.parquet")
C = pl.read_parquet(f"{V8}/recall/head/out/HOLDOUT_ce.parquet")
assert (C["i"] == PLs["s1_row"]).all() and (C["j"] == PLs["j_row"]).all()
assert (PLs["s1_row"] == A["s1_row"]).all() and (PLs["j_row"] == A["j_row"]).all()
Fx = h.features(PLs.with_columns(ce_logit=C["ce_logit"]), f"{SET}/text_train.parquet")
pr = lgb.Booster(model_file=f"{V8}/recall/head/model/head.txt").predict(h.X(Fx))
out["holdout_prob_vs_model_max_abs_diff"] = float(np.abs(pr - p).max())
out["seconds"] = round(time.time() - T0); out["peak_rss_gb"] = round(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1e9, 2)
out["PASS"] = (out["gate_pass"] and all((v is True) if isinstance(v, bool) else v == 0 for v in chk.values()) and out["holdout_prob_vs_model_max_abs_diff"] < 1e-9)
json.dump(out, open(f"{O}/v03.json", "w"), indent=1); print(json.dumps(out, indent=1))
