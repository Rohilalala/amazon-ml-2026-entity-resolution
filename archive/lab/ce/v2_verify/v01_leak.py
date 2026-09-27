"""Verifier (1) leakage of the v2 encoder + combined stacker, own code.
 - v2 encoder fit S1 (d01 + t01's filter re-applied) vs v7 500k sample (oof.parquet S1), 100k holdout, v1 CE-TRAIN S1
   (ce_train.parquet, incl. CE-VAL), VAL2; labels vs ground truth (src.data.load_truth).
 - j-disjointness: fit j vs STAGE4 active (prob >= 0.002 recomputed) and HOLDOUT active in either regime.
 - stacker inputs: out_v2ce/f_stage4 prob == stage4_train.prob (v7 out-of-fold, == oof.parquet), ce == v2 STAGE4 logits,
   row order of out_v2ce/f_* == out/f_* (the S / T tables are hstacked by position).
usage (repo venv, PYTHONDONTWRITEBYTECODE=1): v01_leak.py -> out/v01_leak.json"""
import json, os, resource, sys, threading, time
from pathlib import Path
import numpy as np, polars as pl

ROOT = Path("/Users/adityarohilla/Documents/amazon-ml-2026"); CE = ROOT / "v8_lab/ce"; SET = CE / "setup/out"
REPO = ROOT / "code/business_entity_resolution"; V2 = CE / "v2_encoder/out"; VS = CE / "v2_stacker"
OUT = Path(__file__).parent / "out"
sys.path.insert(0, str(REPO))
from src import data


def guard(limit=3.2e9):
    def run():
        while True:
            if resource.getrusage(resource.RUSAGE_SELF).ru_maxrss > limit:
                print("RSS guard tripped", flush=True); os._exit(3)
            time.sleep(0.5)
    threading.Thread(target=run, daemon=True).start()


guard(float(os.environ.get("RSS_LIMIT_GB", 3.2)) * 1e9); R = {}
iset = lambda s: set(s.unique().to_list())
# ---- S1 sets
sample = iset(pl.read_parquet(REPO / "artifacts/v7/oof.parquet", columns=["i"])["i"])
hold = iset(pl.read_parquet(SET / "holdout_s1.parquet")["i"])
v1_ce = iset(pl.read_parquet(SET / "ce_train.parquet", columns=["i"])["i"])
v1_s1 = iset(pl.read_parquet(SET / "ce_s1.parquet")["i"])
D = pl.read_parquet(V2 / "d01_pairs.parquet")
S1 = pl.read_parquet(V2 / "d01_s1.parquet")
fit = D.filter(~pl.col("is_val2") & ~pl.col("j_in_stage4_active") & ~pl.col("j_in_holdout_active") & ((pl.col("rank") <= 6) | (pl.col("y") == 1)))
fit_i, val2_i, d_i = iset(fit["i"]), iset(S1.filter("is_val2")["i"]), iset(D["i"])
R["s1"] = {"sample": len(sample), "holdout": len(hold), "v1_ce_train_s1": len(v1_ce), "v1_ce_s1_file_eq": v1_ce == v1_s1,
           "fresh_s1": S1.height, "d01_pair_s1": len(d_i), "d01_pair_s1_subset_of_fresh": d_i <= iset(S1["i"]),
           "fit_pairs": fit.height, "fit_pos": int(fit["y"].sum()), "fit_s1": len(fit_i),
           "fit_s1_in_sample": len(fit_i & sample), "fit_s1_in_holdout": len(fit_i & hold), "fit_s1_in_v1_ce_train": len(fit_i & v1_ce),
           "fit_s1_in_val2": len(fit_i & val2_i), "val2_s1_in_v1_or_sample_or_holdout": len(val2_i & (v1_ce | sample | hold)),
           "sample_holdout_overlap": len(sample & hold)}
tl = [json.loads(l) for l in open(V2 / "t01_train.log")]
ev = {e["ev"]: e for e in tl if e["ev"] in ("data", "leak_check", "done")}
R["t01_log"] = {"data_events": sum(e["ev"] == "data" for e in tl), "resume_events": sum(e["ev"] == "resume" for e in tl),
                "fit_pairs": ev["data"]["fit_pairs"], "fit_pos": ev["data"]["fit_pos"], "fit_s1": ev["data"]["fit_s1"],
                "pairs_seen": ev["done"]["pairs_seen"], "epoch_frac": ev["done"]["epoch_frac"],
                "match_refilter": ev["data"]["fit_pairs"] == fit.height and ev["data"]["fit_pos"] == int(fit["y"].sum()) and ev["data"]["fit_s1"] == len(fit_i)}
# ---- labels and country (train cache)
recs = pl.read_parquet(ROOT / "cache/58f0a60c5c264fdd/recs.parquet", columns=["row", "entity_id", "src", "country_key"])
assert (recs["row"].to_numpy() == np.arange(recs.height)).all()
truth = data.load_truth(ROOT / "student_resource/dataset/train/train_ground_truth.tsv", recs)
src = recs["src"].to_numpy(); ck = recs["country_key"].to_numpy()
lab = D.select("i", "j", "y").join(truth.with_columns(pl.lit(1, pl.Int8).alias("t")), on=["i", "j"], how="left")
R["labels"] = {"d01_y_mismatch_vs_truth": int((lab["y"] != lab["t"].fill_null(0)).sum()),
               "fit_i_all_src1": bool((src[fit["i"].to_numpy()] == 1).all()), "fit_j_src": np.unique(src[fit["j"].to_numpy()]).tolist(),
               "fit_countries": sorted(set(ck[fit["i"].to_numpy()].tolist())), "fit_pair_same_country": bool((ck[fit["i"].to_numpy()] == ck[fit["j"].to_numpy()]).all())}
del recs, truth, lab, src, ck
# ---- j-disjointness (active recomputed from prob)
s4 = pl.read_parquet(SET / "stage4_train.parquet", columns=["i", "j", "y", "entity", "prob", "active", "country"])
ho = pl.read_parquet(SET / "holdout_pairs.parquet", columns=["i", "j", "prob", "prob_cap", "active", "active_cap"])
fj = fit["j"].unique().implode()
v1fit = pl.read_parquet(SET / "ce_train.parquet").filter(~pl.col("is_val") & ~pl.col("j_in_stage4_active") & ~pl.col("j_in_holdout_active") & ((pl.col("rank") <= 6) | (pl.col("y") == 1)))
v1j = v1fit["j"].unique().implode()
R["j_disjoint"] = {
    "v2_fit_j_in_stage4_active_pairs": int(s4.filter(pl.col("prob") >= 0.002)["j"].is_in(fj).sum()),
    "v2_fit_j_in_holdout_active_pairs": int(ho.filter(pl.col("prob") >= 0.002)["j"].is_in(fj).sum()),
    "v2_fit_j_in_holdout_active_cap_pairs": int(ho.filter(pl.col("prob_cap") >= 0.002)["j"].is_in(fj).sum()),
    "v2_fit_j_in_any_holdout_pair": int(ho["j"].is_in(fj).sum()),
    "v1_fit_j_in_holdout_active_cap_pairs": int(ho.filter(pl.col("prob_cap") >= 0.002)["j"].is_in(v1j).sum()),
    "v1_fit_j_in_stage4_active_pairs": int(s4.filter(pl.col("prob") >= 0.002)["j"].is_in(v1j).sum()),
    "fit_s1_as_j_in_stage4_or_holdout": int(s4["j"].is_in(fit["i"].unique().implode()).sum() + ho["j"].is_in(fit["i"].unique().implode()).sum()),
    "stage4_active_flag_eq_prob": bool((s4["active"] == (s4["prob"] >= 0.002)).all()),
    "holdout_active_flags_eq_prob": bool((ho["active"] == (ho["prob"] >= 0.002)).all() and (ho["active_cap"] == (ho["prob_cap"] >= 0.002)).all())}
del v1fit, fit, D
# ---- stage-4 prob is v7 out-of-fold (oof.parquet production column)
oof = pl.read_parquet(REPO / "artifacts/v7/oof.parquet", columns=["i", "j", "oof", "oof_stage2"])
m = s4.select("i", "j", "prob", "country").join(oof, on=["i", "j"], how="left")
want = np.where(m["country"].to_numpy() == "us", m["oof_stage2"].to_numpy(), m["oof"].to_numpy())
R["stage4_prob_is_v7_oof"] = {"pairs": s4.height, "unmatched": int(m["oof"].null_count()), "prob_eq": bool((m["prob"].to_numpy() == want).all())}
del oof, m, want
# ---- combined stacker's tables
A = s4.filter("active")
f2 = pl.read_parquet(VS / "out_v2ce/f_stage4.parquet", columns=["i", "j", "y", "entity", "prob", "ce"])
f1 = pl.read_parquet(VS / "out/f_stage4.parquet", columns=["i", "j", "ce"])
c2 = pl.read_parquet(V2 / "STAGE4_ce.parquet"); c1 = pl.read_parquet(CE / "infer/out/stage4_ce.parquet")
R["stage4_table"] = {"rows": f2.height, "rows_eq_active": f2.height == A.height,
                     "ij_eq_stage4_active_order": bool((f2["i"] == A["i"]).all() and (f2["j"] == A["j"]).all()),
                     "ij_eq_v1_table_order": bool((f2["i"] == f1["i"]).all() and (f2["j"] == f1["j"]).all()),
                     "y_eq": bool((f2["y"] == A["y"]).all()), "entity_eq": bool((f2["entity"] == A["entity"]).all()),
                     "prob_eq": bool((f2["prob"] == A["prob"]).all()),
                     "ce_eq_v2_logits": bool((c2["i"] == A["i"]).all() and (f2["ce"].to_numpy() == c2["ce_logit"].to_numpy().astype(np.float64)).all()),
                     "ce_differs_from_v1": int((f2["ce"].to_numpy() != c1["ce_logit"].to_numpy().astype(np.float64)).sum()),
                     "v1_table_ce_eq_v1_logits": bool((f1["ce"].to_numpy() == c1["ce_logit"].to_numpy().astype(np.float64)).all()),
                     "s_rows": pl.scan_parquet(VS / "out/s_stage4.parquet").select(pl.len()).collect().item(),
                     "t_rows": pl.scan_parquet(VS / "out/t_stage4.parquet").select(pl.len()).collect().item()}
# GroupKFold by entity == S1 (entity is one code per S1)
R["stage4_table"]["entity_one_per_s1"] = bool(A.group_by("i").agg(pl.col("entity").n_unique())["entity"].max() == 1
                                              and A.group_by("entity").agg(pl.col("i").n_unique())["i"].max() == 1)
del f2, f1, c2, c1, A, s4
h2 = pl.read_parquet(VS / "out_v2ce/f_holdout.parquet", columns=["r", "i", "j", "prob", "ce"])
h1 = pl.read_parquet(VS / "out/f_holdout.parquet", columns=["r", "i", "j"])
H = pl.read_parquet(SET / "holdout_pairs.parquet", columns=["i", "j", "prob", "active"]).with_row_index("r").filter("active")
ch = H.select("i", "j").join(pl.read_parquet(V2 / "HOLDOUT_ce.parquet"), on=["i", "j"], how="left", maintain_order="left")
R["holdout_table"] = {"rows": h2.height, "rows_eq_active": h2.height == H.height,
                      "r_eq": bool((h2["r"] == H["r"].cast(h2["r"].dtype)).all()), "r_eq_v1_table": bool((h2["r"] == h1["r"]).all()),
                      "prob_eq_uncapped": bool((h2["prob"] == H["prob"]).all()),
                      "ce_eq_v2_logits": bool((h2["ce"].to_numpy() == ch["ce_logit"].to_numpy().astype(np.float64)).all()),
                      "s_rows": pl.scan_parquet(VS / "out/s_holdout.parquet").select(pl.len()).collect().item(),
                      "t_rows": pl.scan_parquet(VS / "out/t_holdout.parquet").select(pl.len()).collect().item()}
R["peak_rss_gb"] = round(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1e9, 2)
(OUT / "v01_leak.json").write_text(json.dumps(R, indent=1))
print(json.dumps(R, indent=1))
