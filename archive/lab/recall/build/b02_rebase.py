"""Upload files = best file (ce/v2_build/A_all_v2, LB 0.985) + recall-head adds on the dense-retrieval new pairs.
Decision = recall/head/model/head.json, applied with h01_head.add_new unchanged: a new pair is added when prob >= 0.80
(S1 already keeps >= 1 id in the best file) or >= 0.75 (S1 keeps none), its j is kept by no S1 of the best file, and it
has the highest prob among that j's candidates. Existing match lists stay as they are; added ids are appended.
  ALL : adds in every country.   USIN: adds in US and India only (France rows identical to the best file).
candidate_pairs.tsv = output_v7_final candidates + the new pairs that were scored for that file's scope.
Probabilities: recall/head/out/test_head_prob.parquet (p01_test.py); re-derived here for the S1 that get an add plus a
random 10k S1 with h01_head.features + model/head.txt (must match).
usage (repo venv, under day1b/heavy.sh): b01_build.py -> ALL/, USIN/, out/b01.json
Rebase copy: RECALL_BASE=<best matching_results.tsv> RECALL_OUT=<dir> b02_rebase.py -> <dir>/ALL, <dir>/USIN, <dir>/out/b01.json"""
import json, os, resource, sys, time
from pathlib import Path
import numpy as np, polars as pl, lightgbm as lgb

ROOT = Path("/Users/adityarohilla/Documents/amazon-ml-2026"); V8 = ROOT / "v8_lab"; D = Path(__file__).parent
sys.path.insert(0, str(V8 / "recall/head"))
import h01_head as h

OUTD = Path(os.environ.get("RECALL_OUT", Path(__file__).parent))
T0 = time.time()
def log(*a): print(f"[{time.time() - T0:6.1f}s]", *a, flush=True)
def q(s, ps=(0.1, 0.5, 0.9)): return {f"p{int(p * 100)}": round(float(s.quantile(p)), 4) for p in ps} if len(s) else {}
TSV = dict(separator="\t", quote_char=None, infer_schema=False)
BEST = Path(os.environ.get("RECALL_BASE", V8 / "ce/v2_build/A_all_v2/matching_results.tsv")); CAND = ROOT / "output_v7_final/candidate_pairs.tsv"
meta = json.load(open(h.MOD / "head.json")); T_KEEP, T_NONE = meta["t_keep"], meta["t_none"]

# ---- best file, ids <-> test cache rows
best = pl.read_csv(BEST, **TSV).select(pl.col("source1_entity_id").alias("s1"), pl.col("matched_entity_ids").fill_null("").alias("m"))
recs = pl.read_parquet(ROOT / "cache/3ee42f05d06efbcb/recs.parquet", columns=["row", "entity_id", "country_key", "src"]).with_columns(
    pl.col("entity_id").str.strip_chars())
best = best.join(recs.select(pl.col("entity_id").alias("s1"), pl.col("row").alias("s1_row"), pl.col("country_key").alias("c"), "src"),
                 on="s1", how="left", maintain_order="left")
assert best["s1_row"].null_count() == 0 and (best["src"] == 1).all() and best["s1"].n_unique() == best.height
best = best.drop("src").with_row_index("code")
kept = best.select("s1", pl.col("m").str.split(",")).explode("m").filter(pl.col("m") != "")
kept_ids = kept["m"].unique()
log("best rows", best.height, "kept ids", kept.height, "unique j kept", kept_ids.len())

# ---- new pairs + head prob (row-aligned with the pair list and the CE scores)
P = pl.read_parquet(h.PAIRS / "test_m1_r1_cap10.parquet")
C = pl.read_parquet(h.OUT / "TEST_ce.parquet"); HP = pl.read_parquet(h.OUT / "test_head_prob.parquet")
assert (P["s1_row"] == C["i"]).all() and (P["j_row"] == C["j"]).all() and (P["s1_row"] == HP["s1_row"]).all() and (P["j_row"] == HP["j_row"]).all()
P = P.with_columns(ce_logit=C["ce_logit"], prob=HP["prob"]); del C, HP
jid = recs.select(pl.col("row").alias("j_row"), pl.col("entity_id").alias("jid"), pl.col("country_key").alias("jc"), pl.col("src").alias("jsrc"))
P = (P.join(best.select("s1_row", "code", "s1", "c", has_keep=pl.col("m") != ""), on="s1_row", how="left", maintain_order="left")
      .join(jid, on="j_row", how="left", maintain_order="left"))
assert P["code"].null_count() == 0 and P["jid"].null_count() == 0
assert (P["c"] == P["country"]).all() and (P["jc"] == P["country"]).all() and (P["jsrc"].cast(pl.UInt8) == P["source"]).all()
P = P.with_columns(j_taken=pl.col("jid").is_in(kept_ids.implode()), thr=pl.when("has_keep").then(T_KEEP).otherwise(T_NONE),
                   rev_only=pl.col("dense_rank").is_null()).drop("jc", "jsrc")
thr_by_s1 = np.where(best["m"].to_numpy() != "", T_KEEP, T_NONE)
code, j, p, taken = P["code"].to_numpy(), P["j_row"].to_numpy(), P["prob"].to_numpy(), P["j_taken"].to_numpy()
add_all = h.add_new(code, j, p, thr_by_s1, taken)
usin = P["country"].is_in(["us", "india"]).to_numpy()
iu = np.flatnonzero(usin)
add_usin = np.zeros(len(p), bool); add_usin[iu] = h.add_new(code[iu], j[iu], p[iu], thr_by_s1, taken[iu])
assert (add_usin == (add_all & usin)).all(), "USIN adds must equal ALL adds restricted to US+India (j is country-local)"
P = P.with_columns(add=pl.Series(add_all))
A = P.filter("add")
assert A["jid"].n_unique() == A.height and not A["j_taken"].any()
log("adds ALL", A.height, "USIN", int(add_usin.sum()))

# ---- re-derive probs for the S1 that get an add + a random 10k S1 (same features() and model as p01_test.py)
rng = np.random.default_rng(0)
chk_s1 = np.union1d(A["s1_row"].unique().to_numpy(), rng.choice(P["s1_row"].unique().to_numpy(), 10000, replace=False))
F = h.features(P.filter(pl.col("s1_row").is_in(chk_s1)).select("s1_row", "j_row", "dense_cos", "dense_rank", "new_rank", "rev_rank",
                                                                "source", "ce_logit", "prob"), h.SET / "text_test.parquet")
model = lgb.Booster(model_file=str(h.MOD / "head.txt"))
dmax = float(np.abs(model.predict(h.X(F)) - F["prob"].to_numpy()).max())
n_chk = F.height; log("prob re-derivation on", n_chk, "pairs: max |diff|", dmax); assert dmax < 1e-9
A = A.join(F.select("s1_row", "j_row", "name_ratio", "addr_ratio", "e_addr_i", "e_addr_j"), on=["s1_row", "j_row"], how="left")
del F

# ---- write the two files
cand = pl.read_csv(CAND, **TSV).select(pl.col("source1_entity_id").alias("s1"), pl.col("candidate_entity_ids").fill_null("").alias("cm"))
assert cand["s1"].equals(best["s1"])
cx = cand.select("s1", pl.col("cm").str.split(",")).explode("cm").filter(pl.col("cm") != "")
n_cand0 = cx.height
assert P.select("s1", pl.col("jid").alias("cm")).join(cx, on=["s1", "cm"], how="semi").height == 0, "new pair already a v7 candidate"
assert kept.join(cx.rename({"cm": "m"}), on=["s1", "m"], how="anti").height == 0, "best-file match outside v7 candidates"
del cx
log("v7 candidate ids", n_cand0)


def append(base, col, extra):
    e = extra.group_by("s1").agg(pl.col("jid").sort().str.join(",").alias("x"))
    return (base.join(e, on="s1", how="left", maintain_order="left")
                .with_columns(pl.when(pl.col("x").is_null()).then(pl.col(col))
                              .when(pl.col(col) == "").then(pl.col("x"))
                              .otherwise(pl.col(col) + "," + pl.col("x")).alias(col)).drop("x"))


R = {"decision": {"t_keep": T_KEEP, "t_none": T_NONE, "rule": meta["decision"]}}
for name, scope in (("ALL", ("us", "india", "france")), ("USIN", ("us", "india"))):
    out = OUTD / name; out.mkdir(parents=True, exist_ok=True)
    inscope = P.filter(pl.col("country").is_in(scope))
    res = append(best.select("s1", "m"), "m", inscope.filter("add"))
    cres = append(cand, "cm", inscope)
    assert res["s1"].equals(best["s1"]) and cres["s1"].equals(best["s1"])
    same = (res["m"] == best["m"]).to_numpy()
    assert (same | best["c"].is_in(scope).to_numpy()).all()
    ex = res.select("s1", pl.col("m").str.split(",")).explode("m").filter(pl.col("m") != "")
    assert not ex["m"].str.starts_with("S1-").any() and not ex.select(pl.struct("s1", "m").is_duplicated().any()).item()
    res.select(pl.col("s1").alias("source1_entity_id"), pl.col("m").alias("matched_entity_ids")).write_csv(
        out / "matching_results.tsv", separator="\t", quote_style="never")
    cres.select(pl.col("s1").alias("source1_entity_id"), pl.col("cm").alias("candidate_entity_ids")).write_csv(
        out / "candidate_pairs.tsv", separator="\t", quote_style="never")
    R[name] = {"rows": res.height, "rows_changed": int((~same).sum()), "matched_ids": ex.height, "matched_ids_best": kept.height,
               "candidate_ids": n_cand0 + inscope.height, "candidate_ids_v7": n_cand0, "new_candidate_ids": inscope.height}
    log(name, R[name]); del res, cres, ex

# ---- per-country stats (ALL scope; USIN = the same numbers for US and India, France 0)
hs1 = pl.read_parquet(h.SET / "holdout_s1.parquet", columns=["i", "country"])
hk = pl.read_parquet(V8 / "ce/v2_build/out/holdout_keeps.parquet").filter("ACNRST_3s_v2ce")  # the best file's holdout keeps
hadd = (pl.read_parquet(h.OUT / "holdout_add.parquet").filter("add").join(hs1.rename({"i": "s1_row"}), on="s1_row", how="left")
        .with_columns(has_keep=pl.col("s1_row").is_in(hk["i"].unique().implode())))
tx = pl.read_parquet(h.SET / "text_test.parquet").filter(pl.col("row").is_in(pl.concat([A["s1_row"], A["j_row"]]).unique().implode()))
R["countries"] = {}
for c in ("us", "india", "france"):
    Pc, Ac, n = P.filter(pl.col("country") == c), A.filter(pl.col("country") == c), int((best["c"] == c).sum())
    above = Pc.filter(pl.col("prob") >= pl.col("thr"))
    r = {"s1": n, "s1_keeping_none_in_best": int(best.filter(pl.col("c") == c)["m"].eq("").sum()), "new_pairs": Pc.height,
         "above_threshold": above.height, "blocked_j_kept_in_best": int(above["j_taken"].sum()),
         "dropped_one_s1_per_j": above.height - int(above["j_taken"].sum()) - Ac.height,
         "added_pairs": Ac.height, "s1_with_add": Ac["s1"].n_unique(),
         "added_per_s1": round(Ac.height / n, 5), "s1_add_share": round(Ac["s1"].n_unique() / n, 5),
         "adds_to_s1_with_keep": int(Ac["has_keep"].sum()), "adds_to_s1_keeping_none": int((~Ac["has_keep"]).sum()),
         "adds_reverse_only_share": round(float(Ac["rev_only"].mean()), 4), "adds_source2_share": round(float((Ac["source"] == 2).mean()), 4),
         "adds_prob": q(Ac["prob"]), "adds_ce_logit": q(Ac["ce_logit"]), "adds_dense_cos": q(Ac["dense_cos"].cast(pl.Float64)),
         "adds_name_ratio": q(Ac["name_ratio"]), "adds_addr_ratio": q(Ac["addr_ratio"]),
         "adds_empty_addr_s1_share": round(float(Ac["e_addr_i"].mean()), 4), "adds_empty_addr_j_share": round(float(Ac["e_addr_j"].mean()), 4)}
    if c != "france":
        hc, nh = hadd.filter(pl.col("country") == c), int((hs1["country"] == c).sum())
        r["holdout"] = {"s1": nh, "added_pairs": hc.height, "true": int(hc["y"].sum()), "precision": round(float(hc["y"].mean()), 4),
                        "s1_with_add": hc["s1_row"].n_unique(), "added_per_s1": round(hc.height / nh, 5),
                        "s1_add_share": round(hc["s1_row"].n_unique() / nh, 5), "adds_to_s1_with_keep": int(hc["has_keep"].sum()),
                        "adds_prob": q(hc["prob"])}
        r["test_vs_holdout_added_per_s1"] = round(r["added_per_s1"] / r["holdout"]["added_per_s1"], 3)
    else:
        smp = Ac.sample(min(15, Ac.height), seed=0).join(tx.rename({"row": "s1_row", "text": "s1_text"}), on="s1_row", how="left").join(
            tx.rename({"row": "j_row", "text": "j_text"}), on="j_row", how="left")
        r["sample_adds"] = smp.select("s1", "jid", "s1_text", "j_text", pl.col("prob").round(4), pl.col("ce_logit").round(3), "rev_only").to_dicts()
    R["countries"][c] = r
    log(c, json.dumps({k: v for k, v in r.items() if k != "sample_adds"}))
R["prob_rederivation"] = {"s1": int(len(chk_s1)), "pairs": n_chk, "max_abs_diff": dmax}
R["seconds"] = round(time.time() - T0); R["peak_rss_gb"] = round(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1e9, 2)
(OUTD / "out").mkdir(parents=True, exist_ok=True); (OUTD / "out/b01.json").write_text(json.dumps(R, indent=1, default=str, ensure_ascii=False))
log("done", R["peak_rss_gb"])
