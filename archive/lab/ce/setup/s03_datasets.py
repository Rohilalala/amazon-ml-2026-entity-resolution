"""Datasets for the cross-encoder track (pairs as record rows i, j of the split's cache; text in out/text_{split}.parquet).
 a) CE-TRAIN: train S1 outside v7's 500k sample and 100k holdout (both reproduced exactly as pipeline.train() and
    checked against artifacts/v7/oof.parquet and the r3_regime split); N_CE random S1 of that pool; per S1 the top-8
    candidates by cached retrieval score plus every true pair among its candidates; 20k S1 flagged is_val (CE-VAL).
 b) STAGE4-TRAIN: every v7 training-sample pair with its out-of-fold production probability (US oof_stage2, India oof).
 d) TEST: every test pair with the stack's corrected production probability (US a22 ldf-train stage 2, India a24
    ldf-train stage 3, France g01 cap stage 2) and kept = pair in the stack's rows for that country.
ACTIVE = prob >= THR. Also writes holdout_s1.parquet + truth_train.parquet for s04 (HOLDOUT scoring).
usage: s03_datasets.py -> out/{ce_train,stage4_train,test_pairs,holdout_s1,truth_train,*_s1}.parquet, out/s03.json"""
import json
from pathlib import Path
import numpy as np, polars as pl

ROOT = Path("/Users/adityarohilla/Documents/amazon-ml-2026")
V8 = ROOT / "v8_lab"
TRAIN_CACHE, TEST_CACHE = ROOT / "cache/58f0a60c5c264fdd", ROOT / "cache/3ee42f05d06efbcb"
OUT = Path(__file__).parent / "out"
SEED, TRAIN_S1_SAMPLE, HOLDOUT_S1 = 42, 500_000, 100_000  # artifacts/v7/bundle.json config
N_CE, N_VAL, TOPK, THR = 360_000, 20_000, 8, 0.002
THRS = [0.001, 0.002, 0.005, 0.01, 0.02, 0.05]
R = {}


def q(x):
    x = np.asarray(x)
    return {"mean": round(float(x.mean()), 3), **{f"p{k}": float(np.percentile(x, k)) for k in (50, 90, 99)}, "max": int(x.max())}


def coverage(df: pl.DataFrame, n_true_total: int) -> dict:
    """ACTIVE size and true-pair coverage by threshold, plus the tighter rule (rank <= 6 by prob within S1 AND prob >= 0.01)."""
    out = {"pairs": df.height, "true_in_cands": int(df["y"].sum()) if "y" in df.columns else None, "by_thr": {}}
    rk = df.select(pl.col("prob").rank("ordinal", descending=True).over("i"))["prob"].to_numpy()
    rules = {f"prob>={t}": (df["prob"] >= t).to_numpy() for t in THRS}
    rules["top6&prob>=0.01"] = (rk <= 6) & (df["prob"] >= 0.01).to_numpy()
    rules["top6&prob>=0.002"] = (rk <= 6) & (df["prob"] >= 0.002).to_numpy()
    s1 = df["i"].unique()
    for name, a in rules.items():
        per = df.filter(pl.Series(a)).group_by("i").len()
        per_all = np.zeros(len(s1)); per_all[:per.height] = per["len"].to_numpy()
        d = {"active": int(a.sum()), "share_pairs": round(float(a.mean()), 5), "active_per_s1": q(per_all)}
        if "y" in df.columns:
            y = df["y"].to_numpy() == 1
            d["true_covered_of_in_cands"] = round(float((a & y).sum() / max(y.sum(), 1)), 6)
            d["true_covered_of_all_true"] = round(float((a & y).sum() / max(n_true_total, 1)), 6)
            d["true_missed"] = int((~a & y).sum())
        if "kept" in df.columns:
            k = df["kept"].to_numpy()
            d["kept_outside_active"] = int((k & ~a).sum())
        out["by_thr"][name] = d
    return out


# ---------- train split: truth, v7 sample + holdout reproduced exactly ----------
recs = pl.read_parquet(TRAIN_CACHE / "recs.parquet", columns=["row", "entity_id", "src", "country_key"])
gt = pl.read_csv(ROOT / "student_resource/dataset/train/train_ground_truth.tsv", separator="\t", quote_char=None, infer_schema=False)
gt = (gt.select(pl.col("source1_entity_id").str.strip_chars().alias("s1"), pl.col("matched_entity_ids").str.split(",").alias("m"))
      .explode("m").with_columns(pl.col("m").str.strip_chars()).filter(pl.col("m").is_not_null() & (pl.col("m") != "")))
ids = recs.select("entity_id", "row", "src")
truth = (gt.join(ids.filter(pl.col("src") == 1).select(pl.col("entity_id").alias("s1"), pl.col("row").alias("i")), on="s1", how="left")
         .join(ids.filter(pl.col("src") != 1).select(pl.col("entity_id").alias("m"), pl.col("row").alias("j")), on="m", how="left")
         .select("i", "j"))
assert truth.null_count().sum_horizontal().item() == 0
truth.write_parquet(OUT / "truth_train.parquet")
s1 = recs.filter(pl.col("src") == 1).select("row", "country_key")
n_true_all = s1.join(truth.group_by("i").len("n_true"), left_on="row", right_on="i", how="left").with_columns(pl.col("n_true").fill_null(0))
shuffled = n_true_all.sample(fraction=1.0, shuffle=True, seed=SEED)
sample = shuffled.head(TRAIN_S1_SAMPLE).sort("row")
holdout = shuffled.slice(sample.height, HOLDOUT_S1).sort("row")
oof_s1 = pl.scan_parquet(ROOT / "code/business_entity_resolution/artifacts/v7/oof.parquet").select("i").unique().collect()["i"]
sp = pl.read_parquet(V8 / "day1b/r3_regime/out/regime/split.parquet", columns=["row", "holdout", "sample"])
R["check_sample_equals_v7_oof_s1"] = set(sample["row"].to_list()) == set(oof_s1.to_list())
R["check_holdout_equals_r3_regime_split"] = set(holdout["row"].to_list()) == set(sp.filter("holdout")["row"].to_list())
R["check_sample_holdout_disjoint"] = not set(sample["row"].to_list()) & set(holdout["row"].to_list())
print("checks", R, flush=True)
assert R["check_sample_equals_v7_oof_s1"] and R["check_holdout_equals_r3_regime_split"] and R["check_sample_holdout_disjoint"]
holdout.select(pl.col("row").alias("i"), pl.col("country_key").alias("country"), "n_true").write_parquet(OUT / "holdout_s1.parquet")
R["train_s1"] = {"all": s1.height, "sample": sample.height, "holdout": holdout.height,
                 "holdout_by_country": dict(holdout.group_by("country_key").len().iter_rows())}

# ---------- a) CE-TRAIN ----------
used = pl.concat([sample["row"], holdout["row"]])
pool = n_true_all.filter(~pl.col("row").is_in(used.implode()))
ce_s1 = pool.sample(N_CE, seed=7).sort("row")
val_rows = ce_s1.sample(N_VAL, seed=8)["row"]
ce_s1 = ce_s1.with_columns(pl.col("row").is_in(val_rows.implode()).alias("is_val"))
P = (pl.scan_parquet(TRAIN_CACHE / "pairs.parquet").select("i", "j", "score")
     .filter(pl.col("i").is_in(ce_s1["row"].implode())).collect())
P = (P.join(truth.with_columns(pl.lit(1, dtype=pl.Int8).alias("y")), on=["i", "j"], how="left").with_columns(pl.col("y").fill_null(0))
     .with_columns(pl.col("score").rank("ordinal", descending=True).over("i").cast(pl.UInt16).alias("rank")))
ce = (P.filter((pl.col("rank") <= TOPK) | (pl.col("y") == 1))
      .join(ce_s1.select(pl.col("row").alias("i"), pl.col("country_key").alias("country"), "is_val"), on="i", how="left")
      .sort("i", "rank"))
assert not set(ce["i"].unique().to_list()) & set(used.to_list())  # leakage guard: no sample / holdout S1
ce.write_parquet(OUT / "ce_train.parquet")
ce_s1.select(pl.col("row").alias("i"), pl.col("country_key").alias("country"), "n_true", "is_val").write_parquet(OUT / "ce_s1.parquet")
R["ce_train"] = {
    "pool_s1_outside_sample_and_holdout": pool.height, "s1": ce_s1.height, "val_s1": N_VAL,
    "s1_by_country": dict(ce_s1.group_by("country_key").len().iter_rows()),
    "pool_s1_by_country": dict(pool.group_by("country_key").len().iter_rows()),
    "all_cand_pairs_of_these_s1": P.height, "pairs": ce.height, "positives": int(ce["y"].sum()),
    "pairs_by_country": dict(ce.group_by("country").len().iter_rows()),
    "pairs_fit": int((~ce["is_val"]).sum()), "pairs_val": int(ce["is_val"].sum()),
    "true_pairs_total_of_these_s1": int(ce_s1["n_true"].sum()), "true_in_cands": int(P["y"].sum()),
    "true_rank_gt_topk_added": int(((ce["rank"] > TOPK) & (ce["y"] == 1)).sum()),
    "pairs_per_s1": q(ce.group_by("i").len()["len"].to_numpy()),
}
print("ce_train", R["ce_train"], flush=True)
del P, ce, pool

# ---------- b) STAGE4-TRAIN (v7 OOF, production choice per country) ----------
cty = recs.filter(pl.col("src") == 1).select(pl.col("row").alias("i"), pl.col("country_key").alias("country"))
oof = (pl.read_parquet(ROOT / "code/business_entity_resolution/artifacts/v7/oof.parquet", columns=["i", "j", "y", "entity", "oof", "oof_stage2"])
       .join(cty, on="i", how="left")
       .with_columns(pl.when(pl.col("country") == "us").then(pl.col("oof_stage2")).otherwise(pl.col("oof")).alias("prob")))
assert oof["country"].null_count() == 0 and set(oof["country"].unique()) <= {"us", "india"}
oof = oof.with_columns((pl.col("prob") >= THR).alias("active"))
oof.select("i", "j", "y", "country", "entity", "prob", "active", pl.col("oof_stage2").alias("prob_stage2"),
           pl.col("oof").alias("prob_full")).write_parquet(OUT / "stage4_train.parquet")
sample.select(pl.col("row").alias("i"), pl.col("country_key").alias("country"), "n_true").write_parquet(OUT / "stage4_s1.parquet")
R["stage4_train"] = coverage(oof.select("i", "y", "prob"), int(sample["n_true"].sum()))
R["stage4_train"]["by_country_active"] = {c: int(oof.filter((pl.col("country") == c) & pl.col("active")).height) for c in ("us", "india")}
print("stage4_train", json.dumps(R["stage4_train"])[:1500], flush=True)
del oof, recs, truth, gt, ids

# ---------- d) TEST (stack probabilities + stack kept flag) ----------
trecs = pl.read_parquet(TEST_CACHE / "recs.parquet", columns=["row", "entity_id", "src", "country_key"])
src = {"us": (V8 / "day1b/a_us/out/a22_scores_us_ldftrain.parquet", "p2", "day1b/build/us_ldf_train_df"),
       "india": (V8 / "day1b/a_in/out/a24_scores_india_ldftrain.parquet", "prob", "day1b/build/in_ldf_train_tables_on_U2"),
       "france": (V8 / "audit_a/out/g01_scores_france_all_stage2.parquet", "p2", "day1b/build/r3_fr_tv_add")}
parts, R["test"] = [], {}
for c, (f, col, build) in src.items():
    sc = pl.read_parquet(f, columns=["i", "j", col]).rename({col: "prob"})
    m = pl.read_csv(V8 / build / "matching_results.tsv", separator="\t", quote_char=None, infer_schema=False)
    kept = (m.select(pl.col("source1_entity_id").alias("s1"), pl.col("matched_entity_ids").fill_null("").str.split(",").alias("m"))
            .explode("m").filter(pl.col("m") != "")
            .join(trecs.select(pl.col("entity_id").alias("s1"), pl.col("row").alias("i"), "country_key"), on="s1")
            .filter(pl.col("country_key") == c)
            .join(trecs.select(pl.col("entity_id").alias("m"), pl.col("row").alias("j")), on="m").select("i", "j"))
    sc = sc.join(kept.with_columns(pl.lit(True).alias("kept")), on=["i", "j"], how="left").with_columns(pl.col("kept").fill_null(False))
    n_s1 = trecs.filter((pl.col("src") == 1) & (pl.col("country_key") == c)).height
    R["test"][c] = {"s1": n_s1, "stack_kept_pairs": kept.height, "stack_kept_pairs_in_scores": int(sc["kept"].sum()),
                    **coverage(sc.select("i", "prob", "kept"), 0)}
    R["test"][c].pop("true_in_cands")
    print("test", c, json.dumps(R["test"][c])[:1200], flush=True)
    parts.append(sc.with_columns(pl.lit(c).alias("country")))
test = pl.concat(parts).with_columns((pl.col("prob") >= THR).alias("active"), pl.col("country").cast(pl.Categorical))
cache_pairs = pl.scan_parquet(TEST_CACHE / "pairs.parquet").select(pl.len()).collect().item()
R["test_total"] = {"pairs": test.height, "cache_pairs": cache_pairs, "unique_pairs": test.select("i", "j").n_unique(),
                   "active": int(test["active"].sum())}
assert R["test_total"]["pairs"] == cache_pairs == R["test_total"]["unique_pairs"]
test.select("i", "j", "country", "prob", "active", "kept").write_parquet(OUT / "test_pairs.parquet")
trecs.filter(pl.col("src") == 1).select(pl.col("row").alias("i"), pl.col("country_key").alias("country")).write_parquet(OUT / "test_s1.parquet")
(OUT / "s03.json").write_text(json.dumps(R, indent=1))
print("done", json.dumps(R["test_total"]))
