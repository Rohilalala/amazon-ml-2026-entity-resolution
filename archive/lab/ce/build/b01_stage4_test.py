"""Stage 4 (set A) on TEST, decided per country; kept (s1, m) parquets for the leaderboard mixes.

Features: stack/s01_stage4.py features() on the TEST active pairs (prob >= 0.002) with the TEST cross-encoder logits;
the empty-address flag is s01_stage4.empty_address()'s expression applied to text_test.parquet (asserted identical to
s01_stage4.empty_address() on text_train). Model: stack/model/stage4_A.txt, rule from stage4_A.json.
Decision: src.pipeline.decide of the stack-corrections worktree per country (exclusive owner rule over all pairs of
the country, non-active pairs keep the stack prob, then the rule); France is unseen: pinned empty-share target from
the worktree's v8 bundle, then src.unseen_rules.correct on the prob France is decided on, as pipeline.finish does.
Checks: (1) holdout: this path reproduces stack/out/holdout_keeps.parquet stage4_A; (2) test: the same path on the
stack prob with the v8 rule (missed 0) reproduces day1b/stack/r4 per country.
usage (repo venv, under heavy.sh): b01_stage4_test.py -> out/b01.json, out/keep_{us,india,france}.parquet"""
import json, sys, time, resource
from pathlib import Path
from types import SimpleNamespace
import numpy as np, polars as pl, lightgbm as lgb

ROOT = Path("/Users/adityarohilla/Documents/amazon-ml-2026")
WT = Path("/Users/adityarohilla/Documents/amazon-ml-2026-integrate/code/business_entity_resolution")
CE = ROOT / "v8_lab/ce"; SET = CE / "setup/out"; INF = CE / "infer/out"; STACK = CE / "stack"; OUT = CE / "build/out"
sys.path.insert(0, str(WT))
from src import pipeline, unseen_rules  # worktree modules; cached before s01_stage4 puts the main repo on the path
sys.path.insert(0, str(STACK))
import s01_stage4 as s4
sys.path.insert(0, str(ROOT / "v8_lab/audit_a"))
from common import relation
assert pipeline.__file__.startswith(str(WT)) and unseen_rules.__file__.startswith(str(WT)) and s4.apply_rule is pipeline.apply_rule
T0 = time.time(); R = {}


def log(*a):
    print(f"[{time.time() - T0:7.1f}s]", *a, flush=True)


def empty_address(path):
    """s01_stage4.empty_address() expression on any text file."""
    e = (pl.scan_parquet(path)
         .select("row", b=pl.col("text").str.splitn(" | ", 2).struct.field("field_1").fill_null("").str.strip_chars() == "")
         .collect(engine="streaming"))
    assert (e["row"].to_numpy() == np.arange(e.height)).all()
    return e["b"].to_numpy()


def features_a(df, emp):
    """s01_stage4.features() minus its set-B columns (n_j, j_ce_other): set A does not use them and their j windows
    stall on the dense TEST j groups. Asserted equal to s01_stage4.features() on the set-A columns (holdout)."""
    q = pl.col("prob").cast(pl.Float64).clip(1e-6, 1 - 1e-6)
    df = df.with_columns(lp=(q / (1 - q)).log(), ce=pl.col("ce_logit").cast(pl.Float64),
                         b_empty=pl.Series(emp[df["j"].to_numpy()]).cast(pl.Int8))
    return df.with_columns(
        ce_rank=pl.col("ce").rank("min", descending=True).over("i"), ce_gap=pl.col("ce") - s4.best_other("ce", "i"),
        lp_rank=pl.col("lp").rank("min", descending=True).over("i"), lp_gap=pl.col("lp") - s4.best_other("lp", "i"),
        n_i=pl.len().over("i"))


def dist(df):
    """Per-feature mean and quantiles."""
    out = {}
    for f in s4.FEAT_A:
        x = df[f].cast(pl.Float64).to_numpy()
        fin = x[np.isfinite(x)]
        out[f] = {"mean": round(float(fin.mean()), 4), **{f"p{q}": round(float(np.quantile(fin, q / 100)), 4) for q in (1, 10, 50, 90, 99)},
                  "null_or_inf": int(len(x) - len(fin))}
    return out


BUNDLE = json.load(open(WT / "artifacts/v8/bundle.json"))
META = json.load(open(STACK / "model/stage4_A.json"))
RULE4, RULE_STACK = META["rule"], BUNDLE["rule"]
TARGET = BUNDLE["config"]["unseen_empty_target"]
R["rules"] = {"stage4": RULE4, "stack_v8": RULE_STACK, "unseen_empty_target": TARGET, "unseen_none_cap": BUNDLE["config"]["unseen_none_cap"]}


def ns(rule):
    return SimpleNamespace(config=SimpleNamespace(exclusive=True, unseen_none_cap=BUNDLE["config"]["unseen_none_cap"],
                                                  unseen_empty_target=TARGET),
                           rule=rule, report={"oof_f05_by_country": BUNDLE["report"]["oof_f05_by_country"]})


booster = lgb.Booster(model_file=str(STACK / "model/stage4_A.txt"))
assert booster.feature_name() == [f"Column_{k}" for k in range(len(s4.FEAT_A))] or booster.feature_name() == s4.FEAT_A
R["model"] = {"trees": booster.num_trees(), "features": META["features"]}
assert META["features"] == s4.FEAT_A

# ---- empty-address flag: identical expression on train, then test
EMP_TR = s4.empty_address()
assert np.array_equal(empty_address(SET / "text_train.parquet"), EMP_TR)
EMP_TE = empty_address(SET / "text_test.parquet")
R["b_empty_records"] = {"train": int(EMP_TR.sum()), "test": int(EMP_TE.sum())}
log("empty flags", R["b_empty_records"])

# ---- check 1: holdout through this path == the stack run's stage4_A keeps
H, SH = s4.load_holdout(EMP_TR)
a = H.with_row_index("r").filter("active").select("r", "i", "j", "prob", "ce_logit")
FH = s4.features(a, EMP_TR)
assert FH.select(s4.FEAT_A).equals(features_a(a, EMP_TR).select(s4.FEAT_A))
R["features_a_eq_s01_features_on_holdout"] = True
ph = H["prob"].to_numpy().astype(np.float64).copy()
ph[a["r"].to_numpy()] = booster.predict(s4.X(FH, s4.FEAT_A))
kh, info = pipeline.decide(H["i"].to_numpy(), H["j"].to_numpy(), ph, H["country"].cast(pl.String).to_numpy(),
                           SH["country"].cast(pl.String).to_numpy(), ns(RULE4))
assert info == {}
ref = pl.read_parquet(STACK / "out/holdout_keeps.parquet")
assert (ref["i"] == H["i"]).all() and (ref["j"] == H["j"]).all()
R["holdout_repro"] = {"pairs": H.height, "diff_vs_stack_run_stage4_A": int((kh != ref["stage4_A"].to_numpy()).sum())}
hc = H["country"].cast(pl.String).to_numpy()
R["holdout_kept_per_s1"] = {}
for c in ("us", "india"):
    n = int((SH["country"].cast(pl.String) == c).sum()); m = hc == c; hi = H["i"].to_numpy()[m]
    r = {}
    for k in ("v7_production", "stage4_A"):
        v = ref[k].to_numpy()[m]
        r[k] = {"kept_per_s1": round(float(v.sum() / n), 5), "empty_share": round(1 - len(np.unique(hi[v])) / n, 5)}
    r["shift_kept_per_s1"] = round(r["stage4_A"]["kept_per_s1"] - r["v7_production"]["kept_per_s1"], 5)
    r["shift_empty_share"] = round(r["stage4_A"]["empty_share"] - r["v7_production"]["empty_share"], 5)
    r["s1"] = n
    R["holdout_kept_per_s1"][c] = r
R["feature_dist"] = {"holdout": {c: dist(FH.filter(pl.col("i").is_in(H.filter(pl.col("country") == c)["i"].unique().implode()))) for c in ("us", "india")}}
log("holdout", R["holdout_repro"], R["holdout_kept_per_s1"])
del H, SH, a, FH, ph, kh, ref, hc

# ---- STAGE4-TRAIN feature distributions
_, _, A4 = s4.load_stage4(EMP_TR)
R["feature_dist"]["stage4_train"] = {c: dist(A4.filter(pl.col("country") == c)) for c in ("us", "india")}
del A4
log("stage4 train distributions done")

# ---- TEST stage-4 prob
T = pl.read_parquet(SET / "test_pairs.parquet")
C = pl.read_parquet(INF / "test_ce.parquet")
A = T.with_row_index("r").filter("active").select("r", "i", "j", "prob", "country")
assert A.height == C.height and (A["i"] == C["i"]).all() and (A["j"] == C["j"]).all()
F = features_a(A.with_columns(ce_logit=C["ce_logit"]), EMP_TE)
del C
p4a = booster.predict(s4.X(F, s4.FEAT_A))
R["feature_dist"]["test"] = {c: dist(F.filter(pl.col("country") == c)) for c in ("us", "india", "france")}
R["test_active"] = {c: int((A["country"] == c).sum()) for c in ("us", "india", "france")}
p4 = T["prob"].to_numpy().astype(np.float64).copy()
p4[A["r"].to_numpy()] = p4a
pl.DataFrame({"i": A["i"], "j": A["j"], "p4": p4a.astype(np.float32)}).write_parquet(OUT / "test_stage4_prob.parquet")
T = T.with_columns(p4=pl.Series(p4))
del F, A, p4a, p4
log("test scored", R["test_active"])

# ---- records, stack r4 kept pairs
recs = pl.read_parquet(ROOT / "cache/3ee42f05d06efbcb/recs.parquet", columns=["row", "entity_id", "src", "country_key", *unseen_rules.COLUMNS])
assert (recs["row"].to_numpy() == np.arange(recs.height)).all()
ids = recs.select(pl.col("entity_id").str.strip_chars(), "row")
m = pl.read_csv(ROOT / "v8_lab/day1b/stack/r4/matching_results.tsv", separator="\t", quote_char=None, infer_schema=False)
st = (m.select(pl.col("source1_entity_id").str.strip_chars().alias("s"), pl.col("matched_entity_ids").fill_null("").str.split(",").alias("m"))
      .explode("m").with_columns(pl.col("m").str.strip_chars()).filter(pl.col("m") != "")
      .join(ids.rename({"entity_id": "s", "row": "i"}), on="s", how="left").join(ids.rename({"entity_id": "m", "row": "j"}), on="m", how="left"))
assert st["i"].null_count() == 0 and st["j"].null_count() == 0
st = st.select(pl.col("i").cast(pl.UInt32), pl.col("j").cast(pl.UInt32)).with_columns(pl.lit(True).alias("st"))
n_st = st.height
T = T.join(st, on=["i", "j"], how="left", maintain_order="left").with_columns(pl.col("st").fill_null(False))
R["r4_pairs"] = n_st; R["r4_pairs_outside_candidates"] = n_st - int(T["st"].sum())
del m, st
ccode = recs["country_key"].cast(pl.Categorical).to_physical().to_numpy()
s1_country = recs.filter(pl.col("src") == 1)["country_key"]
eid = recs["entity_id"].str.strip_chars()
log("records + r4 loaded", R["r4_pairs"], R["r4_pairs_outside_candidates"])


def stats(i, j, keep, n):
    kp = pl.DataFrame({"i": i[keep], "j": j[keep]})
    hv = relation(kp, recs)["hrel"].is_in(["edit1", "off20"]).sum()
    return {"kept_pairs": int(keep.sum()), "kept_per_s1": round(float(keep.sum() / n), 5),
            "empty_share": round(1 - len(np.unique(i[keep])) / n, 5), "house_variant_keeps": int(hv),
            "house_variant_per_s1": round(float(hv / n), 5)}


for c in ("us", "india", "france"):
    t = T.filter(pl.col("country") == c)
    i, j = t["i"].to_numpy(), t["j"].to_numpy()
    assert (ccode[i] == ccode[j]).all()  # owner rule per country == over all pairs
    n = int((s1_country == c).sum())
    pc, sc = np.full(len(i), c), np.full(n, c)
    st_k, act = t["st"].to_numpy(), t["active"].to_numpy()
    p_stack, p_4 = t["prob"].to_numpy().astype(np.float64), t["p4"].to_numpy()
    r = {"s1": n, "pairs": t.height, "active": int(act.sum())}
    # check 2: stack prob + v8 rule reproduces r4
    k0, info0 = pipeline.decide(i, j, p_stack, pc, sc, ns(RULE_STACK))
    if c == "france":
        k0, cnt0 = unseen_rules.correct(recs, i, j, p_stack, k0)
        info0[c].update(cnt0)
    r["repro_r4"] = {"diff": int((k0 != st_k).sum()), "added": int((k0 & ~st_k).sum()), "dropped": int((~k0 & st_k).sum()), "info": info0.get(c)}
    del k0
    log(c, "repro", r["repro_r4"])
    k4, info4 = pipeline.decide(i, j, p_4, pc, sc, ns(RULE4))
    if c == "france":
        r["stage4_before_rules"] = stats(i, j, k4, n)
        k4, cnt4 = unseen_rules.correct(recs, i, j, p_4, k4)
        info4[c].update(cnt4)
    else:
        assert info4 == {}
    r["stage4_info"] = info4.get(c)
    flip = k4 != st_k
    r["vs_r4"] = {"pairs_added": int((k4 & ~st_k).sum()), "pairs_dropped": int((~k4 & st_k).sum()),
                  "flips": int(flip.sum()), "flip_share_of_active": round(float(flip[act].sum() / act.sum()), 6),
                  "flips_outside_active": int(flip[~act].sum()),
                  "s1_rows_changed": int(len(np.unique(i[flip]))), "s1_rows_changed_share": round(len(np.unique(i[flip])) / n, 6)}
    r["stage4"] = stats(i, j, k4, n)
    r["r4"] = stats(i, j, st_k, n)
    r["shift_kept_per_s1"] = round(r["stage4"]["kept_per_s1"] - r["r4"]["kept_per_s1"], 5)
    r["shift_empty_share"] = round(r["stage4"]["empty_share"] - r["r4"]["empty_share"], 5)
    r["kept_min_p4"] = round(float(p_4[k4].min()), 5) if k4.any() else None
    pl.DataFrame({"s1": eid.gather(i[k4]), "m": eid.gather(j[k4])}).write_parquet(OUT / f"keep_{c}.parquet")
    R[c] = r
    log(c, json.dumps(r))
    del t, i, j, k4, flip, st_k, act, p_stack, p_4
R["peak_rss_gb"] = round(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1e9, 2)
(OUT / "b01.json").write_text(json.dumps(R, indent=1))
log("done", R["peak_rss_gb"])
