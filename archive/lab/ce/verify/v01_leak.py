"""Audit 1 (leakage), independent of the setup scripts: v7 sample / holdout re-derived with src.pipeline.train()'s own
lines (data.load_truth on the train cache records, n_true_all, sample(seed), head, slice) and compared with v7 oof.parquet,
the setup's holdout_s1 / stage4_s1, the CE fit set (t01's filter, re-applied here) and the stage-4 inputs.
Repo venv, PYTHONDONTWRITEBYTECODE=1. usage: v01_leak.py -> out/v01_leak.json"""
import json, os, resource, sys, threading, time
from pathlib import Path
import numpy as np, polars as pl

ROOT = Path("/Users/adityarohilla/Documents/amazon-ml-2026")
REPO = ROOT / "code/business_entity_resolution"
sys.path.insert(0, str(REPO))
from src import data, pipeline
SET = ROOT / "v8_lab/ce/setup/out"
OUT = Path(__file__).parent / "out"
CFG = json.load(open(REPO / "artifacts/v7/bundle.json"))["config"]
R = {}


def guard(limit=float(os.environ.get("RSS_LIMIT_GB", 3.2)) * 1e9):
    def run():
        while True:
            if resource.getrusage(resource.RUSAGE_SELF).ru_maxrss > limit:
                print("RSS guard tripped", flush=True); os._exit(3)
            time.sleep(0.5)
    threading.Thread(target=run, daemon=True).start()


guard()
recs = pl.read_parquet(ROOT / "cache/58f0a60c5c264fdd/recs.parquet", columns=["row", "entity_id", "src", "country_key"])
assert (recs["row"].to_numpy() == np.arange(recs.height)).all()
truth = data.load_truth(ROOT / "student_resource/dataset/train/train_ground_truth.tsv", recs)
recs = recs.drop("entity_id")
# ---- src/pipeline.py train(), lines 495-504, verbatim logic ----
s1 = recs.filter(pl.col("src") == 1).select("row", "country_key")
n_true_all = s1.join(truth.group_by("i").len("n_true"), left_on="row", right_on="i", how="left") \
    .with_columns(pl.col("n_true").fill_null(0))
shuffled = n_true_all.sample(fraction=1.0, shuffle=True, seed=CFG["seed"])
sample = shuffled.head(min(CFG["train_s1_sample"], n_true_all.height)).sort("row")
holdout = shuffled.slice(sample.height, CFG["holdout_s1"]).sort("row")
S, H = set(sample["row"].to_list()), set(holdout["row"].to_list())
R["pl_version"] = pl.__version__
R["train_s1"], R["sample"], R["holdout"], R["sample_holdout_overlap"] = s1.height, len(S), len(H), len(S & H)

oof = pl.read_parquet(REPO / "artifacts/v7/oof.parquet", columns=["i", "j", "y", "entity", "oof", "oof_stage2"])
R["sample_eq_v7_oof_s1"] = S == set(oof["i"].unique().to_list())
hs = pl.read_parquet(SET / "holdout_s1.parquet")
R["holdout_eq_setup_holdout_s1"] = H == set(hs["i"].to_list())
R["holdout_n_true_country_eq"] = hs.sort("i").select("i", "n_true", "country").equals(
    holdout.select(pl.col("row").alias("i"), pl.col("n_true").cast(hs["n_true"].dtype), pl.col("country_key").alias("country")))
ss = pl.read_parquet(SET / "stage4_s1.parquet")
R["stage4_s1_eq_sample"] = ss.select("i", "n_true").equals(sample.select(pl.col("row").alias("i"), pl.col("n_true").cast(ss["n_true"].dtype)))

# ---- CE fit set: t01_train.py filter re-applied ----
ce = pl.read_parquet(SET / "ce_train.parquet")
fit = ce.filter(~pl.col("is_val") & ~pl.col("j_in_stage4_active") & ~pl.col("j_in_holdout_active") & ((pl.col("rank") <= 6) | (pl.col("y") == 1)))
ce_i = set(ce["i"].unique().to_list())
fit_i = set(fit["i"].unique().to_list())
R["ce_train"] = {"pairs": ce.height, "s1": len(ce_i), "fit_pairs": fit.height, "fit_pos": int(fit["y"].sum()), "fit_s1": len(fit_i),
                 "ce_s1_in_sample": len(ce_i & S), "ce_s1_in_holdout": len(ce_i & H),
                 "fit_s1_in_val": len(fit_i & set(ce.filter("is_val")["i"].unique().to_list()))}
lab = ce.select("i", "j", "y").join(truth.with_columns(pl.lit(1, pl.Int8).alias("t")), on=["i", "j"], how="left")
R["ce_train"]["y_mismatch_vs_truth"] = int((lab["y"] != lab["t"].fill_null(0)).sum())
# j-disjointness recomputed from the stage-4 / holdout pair files (active recomputed as prob >= 0.002)
s4 = pl.read_parquet(SET / "stage4_train.parquet")
ho = pl.read_parquet(SET / "holdout_pairs.parquet", columns=["i", "j", "y", "prob", "prob_cap", "active", "active_cap"])
fj = fit["j"].unique().implode()
R["ce_fit_j_overlap"] = {
    "stage4_active_pairs_with_fit_j": int(s4.filter(pl.col("prob") >= 0.002)["j"].is_in(fj).sum()),
    "holdout_active_pairs_with_fit_j": int(ho.filter(pl.col("prob") >= 0.002)["j"].is_in(fj).sum()),
    "holdout_active_cap_pairs_with_fit_j": int(ho.filter(pl.col("prob_cap") >= 0.002)["j"].is_in(fj).sum()),
    "holdout_active_flag_eq_recomputed": bool((ho["active"] == (ho["prob"] >= 0.002)).all()),
    "stage4_active_flag_eq_recomputed": bool((s4["active"] == (s4["prob"] >= 0.002)).all())}

# ---- stage-4 inputs are v7 out-of-fold values ----
oof = oof.sort("i", "j"); s4 = s4.sort("i", "j")
ctry = recs["country_key"].to_numpy()[s4["i"].to_numpy()]
want = np.where(ctry == "us", oof["oof_stage2"].to_numpy(), oof["oof"].to_numpy())
R["stage4_vs_oof"] = {"pairs_s4": s4.height, "pairs_oof": oof.height,
                      "same_pairs": bool(s4.height == oof.height and (s4["i"] == oof["i"]).all() and (s4["j"] == oof["j"]).all()),
                      "countries": sorted(set(ctry.tolist())),
                      "prob_eq_oof_production": bool((s4["prob"].to_numpy() == want).all()),
                      "prob_full_eq_oof": bool((s4["prob_full"] == oof["oof"]).all()),
                      "prob_stage2_eq_oof_stage2": bool((s4["prob_stage2"] == oof["oof_stage2"]).all()),
                      "entity_eq": bool((s4["entity"] == oof["entity"]).all()), "y_eq": bool((s4["y"] == oof["y"]).all()),
                      "country_col_eq": bool((s4["country"].cast(pl.String).to_numpy() == ctry).all())}
# oof.parquet is the bundle's own out-of-fold vector: train()'s by-country report recomputed from it
B = json.load(open(REPO / "artifacts/v7/bundle.json"))
codes, nt, sc = oof["entity"].to_numpy(), sample["n_true"].to_numpy(), sample["country_key"].to_numpy()
keep = pipeline.apply_rule(B["rule"], codes, oof["oof"].to_numpy())
R["oof_file_vs_bundle_report"] = {"recomputed": round(pipeline.macro_f05(codes, nt, keep, oof["y"].to_numpy()), 5), "report": B["report"]["oof_f05"]}
for c in np.unique(sc):
    ent = np.flatnonzero(sc == c); mk = np.isin(codes, ent); remap = np.full(len(sc), -1); remap[ent] = np.arange(len(ent))
    R["oof_file_vs_bundle_report"][str(c)] = [round(pipeline.macro_f05(remap[codes[mk]], nt[ent], keep[mk], oof["y"].to_numpy()[mk]), 5),
                                              B["report"]["oof_f05_by_country"][str(c)]]
del keep, codes
del oof, want, ctry
# stage-4 entity code k <-> sample row k (pipeline code_of), and holdout S1 not among stage-4 S1
ent = s4.select("i", "entity").unique().sort("entity")
R["stage4_entity_is_pipeline_code"] = ent["i"].to_list() == sample["row"].to_list()
R["stage4_s1_in_holdout"] = len(set(s4["i"].unique().to_list()) & H)
# ---- holdout pairs: complete candidate set of holdout S1 and labels ----
hc = pl.scan_parquet(ROOT / "cache/58f0a60c5c264fdd/pairs.parquet").select("i", "j").filter(pl.col("i").is_in(holdout["row"].implode())).collect()
hy = ho.select("i", "j", "y").join(truth.with_columns(pl.lit(1, pl.Int8).alias("t")), on=["i", "j"], how="left")
R["holdout_pairs"] = {"rows": ho.height, "cache_pairs_of_holdout_s1": hc.height,
                      "same_pair_set": hc.join(ho, on=["i", "j"], how="anti").height == 0 and ho.join(hc, on=["i", "j"], how="anti").height == 0,
                      "y_mismatch_vs_truth": int((hy["y"] != hy["t"].fill_null(0)).sum())}
R["peak_rss_gb"] = round(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1e9, 2)
(OUT / "v01_leak.json").write_text(json.dumps(R, indent=1))
print(json.dumps(R, indent=1))
