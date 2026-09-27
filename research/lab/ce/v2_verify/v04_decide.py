"""Verifier (4) TEST decision and upload files, own code (heavy.sh; ~8-10 GB).
 - ids: every S1 of test_source1 exactly once; every matched id exists in test_source2/3 (read from the TSVs), is unique
   within its row, is not the S1, has the S1's country and is one of the S1's candidate pairs; for A_all_v2, usin, A_all.
 - US / India: own exclusive owner rule over all pairs of the country (stage-4 prob on active pairs, joined on (i, j) from
   test_ACNRST_3s_v2ce_prob.parquet; stack prob elsewhere) + own expected-F0.5 with the model's rule -> vs file rows.
 - France: own none_scale bisection to the bundle's pinned empty-share target (cap from the bundle), then
   src.unseen_rules.correct (worktree) -> vs file; counts, empty share; same replay of A_all as a control; decomposition.
 - usin: US / India rows == A_all_v2, France rows == A_all.
 - label-free profile of the pairs A_all_v2 adds / drops vs A_all (encoder logits, probabilities, house variants),
   next to the holdout's adds / drops (combined vs stage4_A) with their true rates.
usage (repo venv, heavy.sh): v04_decide.py -> out/v04_decide.json"""
import json, sys, time, resource
from pathlib import Path
import numpy as np, polars as pl

HERE = Path(__file__).parent; sys.path.insert(0, str(HERE))
from common_v import owner, EF
ROOT = Path("/Users/adityarohilla/Documents/amazon-ml-2026"); CE = ROOT / "v8_lab/ce"; SET = CE / "setup/out"; VS = CE / "v2_stacker"
OUT = HERE / "out"
WT = Path("/Users/adityarohilla/Documents/amazon-ml-2026-integrate/code/business_entity_resolution")
sys.path.insert(0, str(WT))
from src import unseen_rules
assert unseen_rules.__file__.startswith(str(WT))
sys.path.insert(0, str(ROOT / "v8_lab/audit_a"))
from common import relation
T0 = time.time(); R = {}
log = lambda *a: print(f"[{time.time() - T0:6.1f}s]", *a, flush=True)


def save():
    R["peak_rss_gb"] = round(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1e9, 2)
    (OUT / "v04_decide.json").write_text(json.dumps(R, indent=1, default=float))


BUNDLE = json.load(open(WT / "artifacts/v8/bundle.json"))
TARGET, CAP = BUNDLE["config"]["unseen_empty_target"], float(BUNDLE["config"]["unseen_none_cap"])
RULE = json.load(open(VS / "model/ACNRST_3s_v2ce.json"))["rule"]; RULE_A = json.load(open(CE / "stack/model/stage4_A.json"))["rule"]
R["config"] = {"rule": RULE, "rule_stage4_A": RULE_A, "target": TARGET, "cap": CAP}

# ---------------- records and source ids
recs = pl.read_parquet(ROOT / "cache/3ee42f05d06efbcb/recs.parquet", columns=["row", "entity_id", "src", "country_key", *unseen_rules.COLUMNS])
assert (recs["row"].to_numpy() == np.arange(recs.height)).all()
CODE = {"us": 0, "india": 1, "france": 2}
eid = recs["entity_id"].str.strip_chars(); ck = recs["country_key"].replace_strict(CODE, return_dtype=pl.Int8).to_numpy(); src = recs["src"].to_numpy()
rd = lambda f, cols: pl.read_csv(ROOT / f"student_resource/dataset/test/{f}", separator="\t", quote_char=None, infer_schema=False, columns=cols)
s1f = rd("test_source1.tsv", ["entity_id", "country"])
S23 = pl.concat([rd("test_source2.tsv", ["entity_id"]), rd("test_source3.tsv", ["entity_id"])])["entity_id"]
R["sources"] = {"s1": s1f.height, "s1_unique": s1f["entity_id"].n_unique(), "s23": S23.len(), "s23_unique": S23.n_unique(),
                "recs_s1_eq_tsv": bool(eid.filter(pl.Series(src == 1)).sort().equals(s1f["entity_id"].str.strip_chars().sort())),
                "recs_s23_eq_tsv": bool(eid.filter(pl.Series(src != 1)).sort().equals(S23.str.strip_chars().sort())),
                "tsv_ids_need_strip": int((s1f["entity_id"] != s1f["entity_id"].str.strip_chars()).sum() + (S23 != S23.str.strip_chars()).sum())}
NS1 = {c: int((s1f["country"].str.strip_chars().str.to_lowercase() == c).sum()) for c in ("us", "india", "france")}
R["sources"]["s1_by_country"] = NS1
ids = pl.DataFrame({"e": eid, "row": recs["row"]})
V23 = pl.DataFrame({"m": S23}).unique()
log("sources", R["sources"])

T = pl.read_parquet(SET / "test_pairs.parquet", columns=["i", "j", "country", "prob"]).with_columns(pl.col("country").cast(pl.String).replace_strict(CODE, return_dtype=pl.Int8))
ti, tj = T["i"].to_numpy(), T["j"].to_numpy(); tc = T["country"].to_numpy()
R["test_pairs"] = {"pairs": T.height, "unique": T.select(pl.struct("i", "j").n_unique()).item(),
                   "country_eq_recs_i_and_j": bool((ck[ti] == tc).all() and (ck[tj] == tc).all()),
                   "i_all_src1": bool((src[ti] == 1).all()), "j_all_src23": bool((src[tj] != 1).all())}
save()


def parse(path, tag):
    m = pl.read_csv(path, separator="\t", quote_char=None, infer_schema=False)
    r = {"rows": m.height, "columns": m.columns, "s1_unique": m["source1_entity_id"].n_unique(),
         "s1_set_eq_test_source1": bool(m["source1_entity_id"].sort().equals(s1f["entity_id"].sort())),
         "empty_rows": int((m["matched_entity_ids"].fill_null("") == "").sum())}
    x = (m.select(pl.col("source1_entity_id").alias("s"), pl.col("matched_entity_ids").fill_null("").str.split(",").alias("m"))
         .explode("m").filter(pl.col("m") != ""))
    r["matched_ids"] = x.height
    r["ids_with_whitespace"] = int((x["m"] != x["m"].str.strip_chars()).sum())
    r["ids_not_in_source2_3"] = x.join(V23, on="m", how="anti").height
    r["dup_ids_within_row"] = x.height - x.unique().height
    r["self_matches"] = int((x["s"] == x["m"]).sum())
    x = x.join(ids.rename({"e": "s", "row": "i"}), on="s", how="left").join(ids.rename({"e": "m", "row": "j"}), on="m", how="left")
    assert x["i"].null_count() == 0 and x["j"].null_count() == 0
    xi, xj = x["i"].to_numpy(), x["j"].to_numpy()
    r["cross_country_pairs"] = int((ck[xi] != ck[xj]).sum())
    x = x.select(pl.col("i").cast(pl.UInt32), pl.col("j").cast(pl.UInt32), f=pl.lit(True)).unique()
    f = T.select("i", "j").join(x, on=["i", "j"], how="left", maintain_order="left")["f"].fill_null(False).to_numpy()
    r["pairs_outside_candidates"] = x.height - int(f.sum())
    R["files"][tag] = r
    log(tag, r)
    return f, m


R["files"] = {}
FILES = {"A_all_v2": CE / "v2_build/A_all_v2/matching_results.tsv",
         "usin": VS / "out_v2ce/upload_ACNRST_3s_v2ce_usin/matching_results.tsv", "A_all": CE / "build/A_all/matching_results.tsv"}
FL, RAW = {}, {}
for k, p in FILES.items():
    FL[k], RAW[k] = parse(p, k)
save()
# usin composition, row-wise on the raw files
cmap = s1f.select(pl.col("entity_id").alias("source1_entity_id"), pl.col("country").str.strip_chars().str.to_lowercase().alias("c"))
jn = lambda a: RAW[a].join(cmap, on="source1_entity_id", how="left").sort("source1_entity_id")
u, v2, a0 = jn("usin"), jn("A_all_v2"), jn("A_all")
same = lambda x, y: (x["matched_entity_ids"].fill_null("") == y["matched_entity_ids"].fill_null("")).to_numpy()
R["usin_composition"] = {c: {"rows": int((u["c"] == c).sum()), "eq_A_all_v2": int(same(u, v2)[(u["c"] == c).to_numpy()].sum()),
                             "eq_A_all": int(same(u, a0)[(u["c"] == c).to_numpy()].sum())} for c in ("us", "india", "france")}
R["rows_changed_vs_A_all"] = {c: int((~same(v2, a0))[(v2["c"] == c).to_numpy()].sum()) for c in ("us", "india", "france")}
del u, v2, a0, RAW
log("usin", R["usin_composition"], R["rows_changed_vs_A_all"])

# ---------------- probabilities
def p4_of(path):
    s = pl.read_parquet(path)
    assert s.select(pl.struct("i", "j").is_duplicated().any()).item() is False
    q = T.select("i", "j").join(s.rename({"p4": "q"}), on=["i", "j"], how="left", maintain_order="left")["q"]
    return q.cast(pl.Float64).fill_null(np.nan).to_numpy(), s.height


p_stack = T["prob"].to_numpy().astype(np.float64)
act = p_stack >= 0.002
q2, n2 = p4_of(VS / "out_v2ce/test_ACNRST_3s_v2ce_prob.parquet")
qa, na_ = p4_of(CE / "build/out/test_stage4_prob.parquet")
R["p4_join"] = {"v2_rows": n2, "A_rows": na_, "active": int(act.sum()), "v2_missing_on_active": int(np.isnan(q2[act]).sum()),
                "v2_present_outside_active": int((~np.isnan(q2[~act])).sum()), "A_missing_on_active": int(np.isnan(qa[act]).sum())}
P = {"v2": np.where(act, q2, p_stack), "A": np.where(act, qa, p_stack)}
del q2, qa
log("p4", R["p4_join"])


def stats(m, keep, c):
    i = ti[m][keep]
    return {"kept_pairs": int(keep.sum()), "kept_per_s1": round(keep.sum() / NS1[c], 5), "empty_share": round(1 - len(np.unique(i)) / NS1[c], 6)}


MASK = {c: tc == CODE[c] for c in ("us", "india", "france")}
KEEP = {}
for c in ("us", "india"):
    m = MASK[c]; r = {}
    for tag, rule, fl in (("v2", RULE, "A_all_v2"), ("A", RULE_A, "A_all")):
        ef = EF(ti[m], owner(tj[m], P[tag][m]), rule["missed"]); k = ef.mask()
        r[tag] = {"diff_vs_file": int((k != FL[fl][m]).sum()), "tie_cut_groups": ef.tie_cut_groups, **stats(m, k, c)}
        KEEP[(c, tag)] = k
    r["usin_flags_eq_A_all_v2"] = bool((FL["usin"][m] == FL["A_all_v2"][m]).all())
    R[c] = r; log(c, r); save()

# ---------------- France
m = MASK["france"]; fi, fj = ti[m], tj[m]; n = NS1["france"]
share = lambda keep: 1 - len(np.unique(fi[keep])) / n


def calibrate(ef):
    s1v = share(ef.mask(1.0))
    if s1v >= TARGET:
        return 1.0, s1v
    if share(ef.mask(CAP)) <= TARGET:
        return CAP, s1v
    lo, hi = 1.0, CAP
    for _ in range(25):
        mid = (lo + hi) / 2
        lo, hi = (mid, hi) if share(ef.mask(mid)) < TARGET else (lo, mid)
    return hi, s1v


fr, K0 = {}, {}
for tag, rule, fl in (("v2", RULE, "A_all_v2"), ("A", RULE_A, "A_all")):
    pf = P[tag][m]; ef = EF(fi, owner(fj, pf), rule["missed"])
    sc, s1v = calibrate(ef); k0 = ef.mask(sc)
    k1, cnt = unseen_rules.correct(recs, fi, fj, pf, k0)
    fr[tag] = {"none_scale": round(sc, 4), "empty_share_scale1": round(s1v, 6), "empty_share_calibrated": round(share(k0), 6),
               "rule_counts": cnt, "diff_vs_file": int((k1 != FL[fl][m]).sum()), "tie_cut_groups": ef.tie_cut_groups,
               "pre_rule": {"kept_pairs": int(k0.sum())}, "final": stats(m, k1, "france"),
               "file": stats(m, FL[fl][m], "france")}
    K0[tag] = k0; KEEP[("france", tag)] = k1
    if tag == "v2":  # decomposition: the v2 probabilities at A_all's scale
        k04 = ef.mask(CAP); k14, cnt4 = unseen_rules.correct(recs, fi, fj, pf, k04)
        fr["v2_at_cap_scale"] = {"none_scale": CAP, "empty_share_calibrated": round(share(k04), 6), "rule_counts": cnt4, "final": stats(m, k14, "france")}
        K0["v2_cap"], KEEP[("france", "v2_cap")] = k04, k14
    log("france", tag, json.dumps(fr[tag]))
a, b = KEEP[("france", "v2")], KEEP[("france", "A")]
c4 = KEEP[("france", "v2_cap")]
fr["decomposition_vs_A_all"] = {
    "total": {"added": int((a & ~b).sum()), "dropped": int((~a & b).sum())},
    "prob_change_at_same_scale_4": {"added": int((c4 & ~b).sum()), "dropped": int((~c4 & b).sum())},
    "scale_4_to_calibrated": {"added": int((a & ~c4).sum()), "dropped": int((~a & c4).sum())},
    "pre_rule": {"added": int((K0["v2"] & ~K0["A"]).sum()), "dropped": int((~K0["v2"] & K0["A"]).sum())},
    "s1_rows_changed": int(len(np.unique(fi[a != b])))}
fr["usin_flags_eq_A_all"] = bool((FL["usin"][m] == FL["A_all"][m]).all())
R["france"] = fr; save()
log("france decomposition", fr["decomposition_vs_A_all"])

# ---------------- label-free profile of the TEST flips (v2 vs A_all files) and the holdout flips (combined vs stage4_A)
def ce_all(path):
    s = pl.read_parquet(path)
    return T.select("i", "j").join(s, on=["i", "j"], how="left", maintain_order="left")["ce_logit"].cast(pl.Float64).fill_null(np.nan).to_numpy()


CEV = {"v2": ce_all(CE / "v2_encoder/out/TEST_ce.parquet"), "v1": ce_all(CE / "infer/out/test_ce.parquet")}


def prof(idx, extra=None):
    o = {"pairs": len(idx)}
    for k, v in CEV.items():
        x = v[idx]; x = x[~np.isnan(x)]
        o[f"ce_{k}_pos_share"] = round(float((x > 0).mean()), 4) if len(x) else None
        o[f"ce_{k}_median"] = round(float(np.median(x)), 3) if len(x) else None
    for k, v in (("p4_v2", P["v2"]), ("p4_A", P["A"]), ("p_stack", p_stack)):
        o[f"{k}_median"] = round(float(np.median(v[idx])), 4) if len(idx) else None
    return o


def hv_share(idx):
    if not len(idx):
        return None
    p = pl.DataFrame({"i": ti[idx], "j": tj[idx]})
    rows = pl.concat([p["i"], p["j"]]).unique().to_frame("row")
    rr = recs.select("row", "core", "addr", "house").join(rows, on="row", how="semi")
    h = relation(p, rr)["hrel"]
    return {"house_variant_pairs": int(h.is_in(["edit1", "off20"]).sum()), "house_variant_share": round(float(h.is_in(["edit1", "off20"]).mean()), 4)}


flips = {}
for c in ("us", "india", "france"):
    m = MASK[c]; idx = np.flatnonzero(m)
    fa, fb = FL["A_all_v2"][m], FL["A_all"][m]
    add, drop, both = idx[fa & ~fb], idx[~fa & fb], idx[fa & fb]
    flips[c] = {"added": {**prof(add), **(hv_share(add) or {})}, "dropped": {**prof(drop), **(hv_share(drop) or {})},
                "kept_both": prof(both[np.random.default_rng(0).choice(len(both), 200_000, replace=False)])}
    log("flips", c, json.dumps(flips[c]))
R["test_flips_vs_A_all"] = flips; save()
HK = pl.read_parquet(CE / "v2_build/out/holdout_keeps.parquet", columns=["i", "j", "stage4_A", "ACNRST_3s_v2ce"])
HP = pl.read_parquet(SET / "holdout_pairs.parquet", columns=["i", "j", "y", "country", "prob"])
assert (HK["i"] == HP["i"]).all() and (HK["j"] == HP["j"]).all()
h = HK.with_columns(HP["y"], HP["country"], HP["prob"])
for tag, path in (("v2", CE / "v2_encoder/out/HOLDOUT_ce.parquet"), ("v1", CE / "infer/out/holdout_ce.parquet")):
    h = h.join(pl.read_parquet(path).rename({"ce_logit": f"ce_{tag}"}), on=["i", "j"], how="left", maintain_order="left")
hf = {}
for c in ("us", "india"):
    x = h.filter(pl.col("country") == c)
    for nm, flt in (("added", pl.col("ACNRST_3s_v2ce") & ~pl.col("stage4_A")), ("dropped", ~pl.col("ACNRST_3s_v2ce") & pl.col("stage4_A"))):
        d = x.filter(flt); o = {"pairs": d.height, "true_rate": round(float(d["y"].mean()), 4)}
        for tag in ("v2", "v1"):
            z = d[f"ce_{tag}"].drop_nulls().to_numpy()
            o[f"ce_{tag}_pos_share"] = round(float((z > 0).mean()), 4); o[f"ce_{tag}_median"] = round(float(np.median(z)), 3)
            o[f"true_rate_ce_{tag}_pos"] = round(float(d.filter(pl.col(f"ce_{tag}") > 0)["y"].mean()), 4)
            o[f"true_rate_ce_{tag}_neg"] = round(float(d.filter(pl.col(f"ce_{tag}") <= 0)["y"].mean()), 4) if d.filter(pl.col(f"ce_{tag}") <= 0).height else None
        o["p_stack_median"] = round(float(d["prob"].median()), 4)
        hf[f"{c}_{nm}"] = o
R["holdout_flips_combined_vs_stage4_A"] = hf
save(); log("done", R["peak_rss_gb"])
