"""Leakage checks + independent re-choice of (t_keep, t_none) on HEAD OOF only.
Sets: v7 500k sample = S1 of code/.../artifacts/v7/oof.parquet; holdout = ce/setup/out/holdout_s1.parquet.
Head training S1 = S1 of recall/retrieve/out/pairs/head_m1_r1_cap10.parquet, HEAD_ce.parquet and head_oof.parquet.
The CE that feeds the head (ce/v2_encoder, continued from ce/train) was trained on ce_s1 (v1) + d01_s1 (v2): must avoid the holdout.
Threshold: own grid search over the h01 grid on head_oof (existing keeps proxied by in-candidate trues, as documented).
-> out/v04.json"""
import json, resource, time
import numpy as np, polars as pl

R = "/Users/adityarohilla/Documents/amazon-ml-2026"; V8 = f"{R}/v8_lab"; SET = f"{V8}/ce/setup/out"; O = f"{V8}/recall/verify/out"
T0 = time.time()
sample = pl.scan_parquet(f"{R}/code/business_entity_resolution/artifacts/v7/oof.parquet").select("i").unique().collect()["i"]
hold = pl.read_parquet(f"{SET}/holdout_s1.parquet")["i"]
sp = pl.read_parquet(f"{V8}/day1b/r3_regime/out/regime/split.parquet", columns=["row", "holdout", "sample"])
S, Hd = set(sample.to_list()), set(hold.to_list())
head_rows = np.load(f"{V8}/recall/retrieve/out/sets/head_rows.npy")
head_s1 = {
    "head_rows.npy": set(head_rows.tolist()),
    "pair_list": set(pl.read_parquet(f"{V8}/recall/retrieve/out/pairs/head_m1_r1_cap10.parquet", columns=["s1_row"])["s1_row"].unique().to_list()),
    "HEAD_ce": set(pl.read_parquet(f"{V8}/recall/head/out/HEAD_ce.parquet", columns=["i"])["i"].unique().to_list()),
    "head_oof": set(pl.read_parquet(f"{V8}/recall/head/out/head_oof.parquet", columns=["s1_row"])["s1_row"].unique().to_list())}
ce1 = set(pl.read_parquet(f"{SET}/ce_s1.parquet", columns=["i"])["i"].to_list())
ce2 = set(pl.read_parquet(f"{V8}/ce/v2_encoder/out/d01_s1.parquet", columns=["i"])["i"].to_list())
out = {"sample_s1": len(S), "holdout_s1": len(Hd),
       "sample_equals_split_flag": S == set(sp.filter("sample")["row"].to_list()),
       "holdout_equals_split_flag": Hd == set(sp.filter("holdout")["row"].to_list()),
       "sample_holdout_overlap": len(S & Hd),
       "head_sets": {k: {"s1": len(v), "subset_of_head_rows": v <= head_s1["head_rows.npy"], "overlap_sample": len(v & S), "overlap_holdout": len(v & Hd)}
                     for k, v in head_s1.items()},
       "ce_v1_train_s1_overlap_holdout": len(ce1 & Hd), "ce_v2_train_s1_overlap_holdout": len(ce2 & Hd),
       "ce_train_s1_overlap_sample": len((ce1 | ce2) & S)}
# ---- re-choose thresholds on HEAD OOF (no holdout input)
oof = pl.read_parquet(f"{V8}/recall/head/out/head_oof.parquet").with_row_index("idx")
truth = pl.read_parquet(f"{SET}/truth_train.parquet").filter(pl.col("i").is_in(pl.Series(head_rows).implode()))
cand = (pl.scan_parquet(f"{R}/cache/58f0a60c5c264fdd/pairs.parquet").select("i", "j")
        .filter(pl.col("i").is_in(pl.Series(head_rows).implode())).collect())
truth = truth.join(cand.with_columns(inc=pl.lit(True)), on=["i", "j"], how="left").with_columns(pl.col("inc").fill_null(False))
out["head_new_pairs_in_sparse_candidates"] = oof.join(cand, left_on=["s1_row", "j_row"], right_on=["i", "j"], how="semi").height
del cand
yt = oof.join(truth.select(pl.col("i").alias("s1_row"), pl.col("j").alias("j_row"), pl.lit(1, pl.Int8).alias("yt")), on=["s1_row", "j_row"], how="left")["yt"].fill_null(0)
out["head_labels_match_truth"] = bool((yt == oof["y"]).all())
n = len(head_rows); tc = np.searchsorted(head_rows, truth["i"].to_numpy())
nt = np.bincount(tc, minlength=n).astype(float); k0 = np.bincount(tc, weights=truth["inc"].to_numpy(), minlength=n)
code = np.searchsorted(head_rows, oof["s1_row"].to_numpy()); y = oof["y"].to_numpy(); p = oof["oof"].to_numpy()
taken = oof["j_row"].is_in(truth.filter("inc")["j"].implode()).to_numpy()


def f05(npred, ntrue, tp):
    o = np.zeros(len(ntrue)); z = ntrue == 0; o[z] = npred[z] == 0
    m = ~z & (tp > 0); pr, rc = tp[m] / npred[m], tp[m] / ntrue[m]; o[m] = 1.25 * pr * rc / (0.25 * pr + rc); return o


jr = oof["j_row"].to_numpy(); order = np.lexsort((np.arange(len(p)), -p, jr))  # j asc, prob desc, row order
base = f05(k0, nt, k0).mean()
GRID = [round(x, 2) for x in np.arange(0.20, 0.96, 0.05)] + [0.97, 0.98, 0.99, 1.01]
res = []
for tk in GRID:
    for tn in GRID:
        ok = (p >= np.where(k0[code] > 0, tk, tn)) & ~taken
        o = order[ok[order]]; first = np.ones(len(o), bool); first[1:] = jr[o][1:] != jr[o][:-1]
        m = np.zeros(len(p), bool); m[o[first]] = True
        f = f05(k0 + np.bincount(code, weights=m, minlength=n), nt, k0 + np.bincount(code, weights=m & (y == 1), minlength=n)).mean()
        res.append((round(float(f), 6), tk, tn))
res.sort(key=lambda r: (-r[0], r[1], r[2]))
h = json.load(open(f"{V8}/recall/head/out/h01.json"))["head_decision"]; meta = json.load(open(f"{V8}/recall/head/model/head.json"))
out["t_rechoice_on_head_oof"] = {"base_f05": round(float(base), 6), "top5": res[:5], "argmax": [res[0][1], res[0][2]],
                                 "shipped": [meta["t_keep"], meta["t_none"]], "h01_best": h["best"], "h01_base": h["base_f05"]}
out["t_chosen_on_head"] = [res[0][1], res[0][2]] == [meta["t_keep"], meta["t_none"]] and abs(res[0][0] - h["best"]["f05"]) < 1e-6
# code-order evidence: in h01_head.main the holdout is first read after the threshold choice and the model save
src = open(f"{V8}/recall/head/h01_head.py").read(); body = src[src.index("def main"):]
out["h01_holdout_first_read_after_choose_and_save"] = body.index("load(\"holdout\")") > body.index("choose(") and body.index("load(\"holdout\")") > body.index("save_model")
hs = out["head_sets"].values()
out["PASS"] = (out["sample_equals_split_flag"] and out["holdout_equals_split_flag"] and out["sample_holdout_overlap"] == 0
               and all(v["subset_of_head_rows"] and v["overlap_sample"] == 0 and v["overlap_holdout"] == 0 for v in hs)
               and out["ce_v1_train_s1_overlap_holdout"] == 0 and out["ce_v2_train_s1_overlap_holdout"] == 0
               and out["head_new_pairs_in_sparse_candidates"] == 0 and out["head_labels_match_truth"] and out["t_chosen_on_head"]
               and out["h01_holdout_first_read_after_choose_and_save"])
out["seconds"] = round(time.time() - T0); out["peak_rss_gb"] = round(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1e9, 2)
json.dump(out, open(f"{O}/v04.json", "w"), indent=1, default=str); print(json.dumps(out, indent=1, default=str))
