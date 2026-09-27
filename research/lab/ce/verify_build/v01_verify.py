"""Independent recomputation of the stage-4 test builds (verifier; own code, reuses only the model file, the v8 bundle
config and src.unseen_rules for the France replay/rule predicates).
 (3) stage-4 prob: own numpy feature code over the TEST active pairs (CE logits joined on (i, j), empty-address flag by
     a regex on text_test) + stack/model/stage4_A.txt, compared with build/out/test_stage4_prob.parquet on a random
     200k-pair sample and on all active pairs; features also compared with s01_stage4.features() on the sampled S1.
 (4) US/India decision: own exclusive owner rule over all test pairs + own expected-F0.5 (missed from stage4_A.json)
     on the saved p4 (non-active pairs keep the stack prob) -> compared with A_all rows.
 (5) France: own none_scale calibration (pinned target, cap) + unseen_rules.correct replay; per-rule adds/drops
     checked against each rule's conditions; empty share of the file.
 (6) label-free flags: CE-logit sign of the pairs A adds/drops vs r4 on test against the holdout's stage4_A vs v7 flips.
usage (repo venv, heavy.sh): v01_verify.py -> out/v01.json"""
import json, sys, time, resource
from pathlib import Path
import numpy as np, polars as pl, lightgbm as lgb

ROOT = Path("/Users/adityarohilla/Documents/amazon-ml-2026"); CE = ROOT / "v8_lab/ce"; OUT = CE / "verify_build/out"
WT = Path("/Users/adityarohilla/Documents/amazon-ml-2026-integrate/code/business_entity_resolution")
sys.path.insert(0, str(WT))
from src import unseen_rules
assert unseen_rules.__file__.startswith(str(WT))
T0 = time.time(); R = {}


def log(*a):
    print(f"[{time.time() - T0:7.1f}s]", *a, flush=True)


def save():
    R["peak_rss_gb"] = round(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1e9, 2)
    (OUT / "v01.json").write_text(json.dumps(R, indent=1, default=float))


BUNDLE = json.load(open(WT / "artifacts/v8/bundle.json"))
TARGET, CAP = BUNDLE["config"]["unseen_empty_target"], float(BUNDLE["config"]["unseen_none_cap"])
META = json.load(open(CE / "stack/model/stage4_A.json"))
FEAT, RULE4 = META["features"], META["rule"]
assert RULE4["kind"] == "expected_f05" and BUNDLE["rule"]["kind"] == "expected_f05"
R["config"] = {"features": FEAT, "rule4": RULE4, "rule_v8": BUNDLE["rule"], "target": TARGET, "cap": CAP}


class Groups:
    """Rows sorted by group, then value descending (stable)."""
    def __init__(self, g, v):
        self.o = np.lexsort((-v, g)); gs = g[self.o]; self.v = v[self.o]; n = len(v)
        start = np.r_[True, gs[1:] != gs[:-1]]
        self.gid = np.cumsum(start) - 1; self.gstart = np.flatnonzero(start)
        self.gsize = np.diff(np.r_[self.gstart, n]); self.pos = np.arange(n) - self.gstart[self.gid]

    def unsort(self, x):
        out = np.empty_like(x); out[self.o] = x; return out


def rank_gap(g, v):
    """rank('min', descending) within g and v minus the max of v over the other rows of g (NaN for singletons)."""
    G = Groups(g, v); s = G.v; n = len(s); gid = G.gid
    newrun = np.r_[True, s[1:] != s[:-1]] | (G.pos == 0)
    first = np.maximum.accumulate(np.where(newrun, np.arange(n), 0))
    rank = (first - G.gstart[gid] + 1).astype(np.float64)
    top1 = s[G.gstart]; top2 = np.full(len(G.gstart), np.nan); m = G.gsize > 1; top2[m] = s[G.gstart[m] + 1]
    ntop = np.bincount(gid, weights=(s == top1[gid]).astype(np.float64), minlength=len(G.gstart))
    other = np.where((s == top1[gid]) & (ntop[gid] == 1), top2[gid], top1[gid])
    return G.unsort(rank), G.unsort(s - other), G.unsort(G.gsize[gid].astype(np.float64))


class EF:
    """Expected-F0.5 top-k per group (own implementation of the documented rule)."""
    def __init__(self, g, p, missed):
        G = Groups(g, p); s = G.v; gid, gst, gsz = G.gid, G.gstart, G.gsize
        cum = s.copy()
        for L in range(1, int(gsz.max())):  # sequential within-group cumulative sum
            idx = gst[gsz > L] + L
            cum[idx] += cum[idx - 1]
        total = cum[gst + gsz - 1][gid]; k = G.pos + 1
        ef = 1.25 * cum / (0.25 * (total + missed) + k)
        self.none = np.exp(np.add.reduceat(np.log(np.clip(1 - s, 1e-12, None)), gst)) * np.exp(-missed)
        self.best = np.maximum.reduceat(ef, gst)
        hit = np.flatnonzero(ef == self.best[gid]); hg = gid[hit]
        fh = hit[np.r_[True, hg[1:] != hg[:-1]]]
        self.bestk = np.zeros(len(gst), np.int64); self.bestk[gid[fh]] = G.pos[fh] + 1
        self.G, self.p = G, p
        # groups whose cut falls inside a run of tied probabilities (order-dependent keeps)
        kk = self.bestk; inner = (kk > 0) & (kk < gsz) & (self.best > self.none)
        a = gst[inner] + kk[inner]
        self.tie_cut_groups = int(((s[a] == s[a - 1]) & (s[a] > 0)).sum())

    def mask(self, scale=1.0):
        kk = np.where(self.best > self.none * scale, self.bestk, 0)
        return self.G.unsort(self.G.pos + 1 <= kk[self.G.gid]) & (self.p > 0)


# ============ (3) stage-4 probabilities
T = pl.read_parquet(CE / "setup/out/test_pairs.parquet", columns=["i", "j", "country", "prob", "active"]).with_row_index("r")
MASK = {c: (T["country"] == c).to_numpy() for c in ("us", "india", "france")}
R["test_pairs"] = {"pairs": T.height, "active": int(T["active"].sum()), **{c: int(m.sum()) for c, m in MASK.items()}}
A = T.filter("active").select("r", "i", "j", "prob")
CEL = pl.read_parquet(CE / "infer/out/test_ce.parquet")
assert CEL.select(pl.struct("i", "j").is_duplicated().any()).item() is False
A = A.join(CEL, on=["i", "j"], how="left", maintain_order="left")
R["ce_join"] = {"ce_rows": CEL.height, "active_rows": A.height, "active_without_ce": int(A["ce_logit"].null_count())}
assert CEL.height == A.height and A["ce_logit"].null_count() == 0
del CEL
tx = pl.read_parquet(CE / "setup/out/text_test.parquet")
assert (tx["row"].to_numpy() == np.arange(tx.height)).all()
EMP = tx.select(pl.col("text").str.extract(r"(?s)^.*? \| (.*)$", 1).fill_null("").str.strip_chars() == "").to_series().to_numpy()
del tx
R["b_empty_records_test"] = int(EMP.sum())
ai, aj = A["i"].to_numpy(), A["j"].to_numpy()
prob_a = A["prob"].to_numpy().astype(np.float64); ce = A["ce_logit"].to_numpy().astype(np.float64)
assert np.isfinite(prob_a).all() and np.isfinite(ce).all()
q = np.clip(prob_a, 1e-6, 1 - 1e-6); lp = np.log(q / (1 - q))
ce_rank, ce_gap, n_i = rank_gap(ai, ce)
lp_rank, lp_gap, _ = rank_gap(ai, lp)
FE = {"lp": lp, "ce": ce, "b_empty": EMP[aj].astype(np.float64), "ce_rank": ce_rank, "ce_gap": ce_gap,
      "lp_rank": lp_rank, "lp_gap": lp_gap, "n_i": n_i}
X = np.column_stack([FE[f] for f in FEAT]).astype(np.float32)
booster = lgb.Booster(model_file=str(CE / "stack/model/stage4_A.txt"))
R["model_trees"] = booster.num_trees()
p4_mine = booster.predict(X)
log("features + predict", A.height)
S = pl.read_parquet(CE / "build/out/test_stage4_prob.parquet")
assert S.select(pl.struct("i", "j").is_duplicated().any()).item() is False and S.height == A.height
Sj = A.select("i", "j").join(S, on=["i", "j"], how="left", maintain_order="left")
assert Sj["p4"].null_count() == 0
p4_saved = Sj["p4"].to_numpy().astype(np.float64)
rng = np.random.default_rng(20260927)
samp = rng.choice(A.height, 200_000, replace=False)
d_all, d_s = np.abs(p4_mine - p4_saved), np.abs(p4_mine[samp] - p4_saved[samp])
R["p4_check"] = {"sample_pairs": len(samp), "sample_max_abs_diff": float(d_s.max()), "sample_n_diff_gt_1e-6": int((d_s > 1e-6).sum()),
                 "sample_exact_float32_equal": int((p4_mine[samp].astype(np.float32) == p4_saved[samp].astype(np.float32)).sum()),
                 "all_pairs": A.height, "all_max_abs_diff": float(d_all.max()), "all_n_diff_gt_1e-6": int((d_all > 1e-6).sum()),
                 "all_exact_float32_equal": int((p4_mine.astype(np.float32) == p4_saved.astype(np.float32)).sum())}
log("p4 check", R["p4_check"])
# s01_stage4.features() on the sampled S1 (all their active pairs) vs own features
sys.path.insert(0, str(CE / "stack"))
import s01_stage4 as s4
isamp = np.unique(ai[samp]); sub = np.flatnonzero(np.isin(ai, isamp))
FS = s4.features(A[sub].select("i", "j", "prob", "ce_logit"), EMP)
fcmp = {}
for f in FEAT:
    a_, b_ = FS[f].cast(pl.Float64).fill_null(np.nan).to_numpy(), FE[f][sub]
    both_nan = np.isnan(a_) & np.isnan(b_)
    fcmp[f] = int((~both_nan & (a_ != b_)).sum())
R["features_vs_s01_on_sampled_s1"] = {"s1": len(isamp), "pairs": len(sub), "mismatches": fcmp}
log("features vs s01", R["features_vs_s01_on_sampled_s1"])
del FS, X, FE, ce_rank, ce_gap, lp_rank, lp_gap, n_i, lp, q, Sj, S
save()

# ============ (4)-(6) decisions
recs = pl.read_parquet(ROOT / "cache/3ee42f05d06efbcb/recs.parquet", columns=["row", "entity_id", "src", "country_key", *unseen_rules.COLUMNS])
assert (recs["row"].to_numpy() == np.arange(recs.height)).all()
s1f = pl.read_csv(ROOT / "student_resource/dataset/test/test_source1.tsv", separator="\t", quote_char=None, infer_schema=False)
s1f = s1f.select(pl.col("entity_id").str.strip_chars().alias("e"), pl.col("country").str.strip_chars().str.to_lowercase().alias("c"))
r1 = recs.filter(pl.col("src") == 1).select(pl.col("entity_id").str.strip_chars().alias("e"), pl.col("country_key").alias("c"))
R["recs_s1_match_test_source1"] = bool(r1.sort("e").equals(s1f.sort("e")))
NS1 = {c: int((s1f["c"] == c).sum()) for c in ("us", "india", "france")}
ids = recs.select(pl.col("entity_id").str.strip_chars().alias("e"), pl.col("row"))
ccode = recs["country_key"].to_numpy()
ti, tj = T["i"].to_numpy(), T["j"].to_numpy()
for c, m in MASK.items():
    assert (ccode[ti[m]] == c).all() and (ccode[tj[m]] == c).all(), c
R["pair_country_consistent_with_recs"] = True


def file_flags(p):
    m = pl.read_csv(p, separator="\t", quote_char=None, infer_schema=False)
    x = (m.select(pl.col("source1_entity_id").str.strip_chars().alias("s"), pl.col("matched_entity_ids").fill_null("").str.split(",").alias("m"))
         .explode("m").with_columns(pl.col("m").str.strip_chars()).filter(pl.col("m") != "")
         .join(ids.rename({"e": "s", "row": "i"}), on="s", how="left").join(ids.rename({"e": "m", "row": "j"}), on="m", how="left"))
    assert x["i"].null_count() == 0 and x["j"].null_count() == 0
    x = x.select(pl.col("i").cast(pl.UInt32), pl.col("j").cast(pl.UInt32), f=pl.lit(True))
    f = T.select("i", "j").join(x, on=["i", "j"], how="left", maintain_order="left")["f"].fill_null(False).to_numpy()
    return f, x.height, x.height - int(f.sum())


FA, nA, outA = file_flags(CE / "build/A_all/matching_results.tsv")
F4, n4, out4 = file_flags(ROOT / "v8_lab/day1b/stack/r4/matching_results.tsv")
R["file_pairs"] = {"A": nA, "A_outside_test_pairs": outA, "r4": n4, "r4_outside_test_pairs": out4}
log("files loaded", R["file_pairs"])

p_stack = T["prob"].to_numpy().astype(np.float64); act = T["active"].to_numpy(); ar = A["r"].to_numpy()
P = {"saved": p_stack.copy(), "mine": p_stack.copy()}
P["saved"][ar] = p4_saved; P["mine"][ar] = p4_mine
del A


def owner(j, p):
    o = np.argsort(j, kind="stable"); js = j[o]; st = np.flatnonzero(np.r_[True, js[1:] != js[:-1]])
    mx = np.maximum.reduceat(p[o], st); top = np.empty_like(p); top[o] = np.repeat(mx, np.diff(np.r_[st, len(js)]))
    return np.where(p >= top, p, 0.0)


PE = {k: owner(tj, v) for k, v in P.items()}
PE["stack"] = owner(tj, p_stack)
log("owner rule")


def stats(m, keep, c):
    i = ti[m][keep]
    return {"kept_pairs": int(keep.sum()), "kept_per_s1": round(keep.sum() / NS1[c], 5), "empty_share": round(1 - len(np.unique(i)) / NS1[c], 6)}


for c in ("us", "india"):
    m = MASK[c]; fa = FA[m]; r = {}
    for k in ("saved", "mine"):
        ef = EF(ti[m], PE[k][m], RULE4["missed"]); kp = ef.mask()
        r[k] = {"diff_vs_A": int((kp != fa).sum()), "tie_cut_groups": ef.tie_cut_groups}
    ef = EF(ti[m], PE["stack"][m], BUNDLE["rule"]["missed"]); k0 = ef.mask()
    r["stack_v8_rule_diff_vs_r4"] = int((k0 != F4[m]).sum())
    r["A"], r["r4"] = stats(m, fa, c), stats(m, F4[m], c)
    r["shift_kept_per_s1"] = round(r["A"]["kept_per_s1"] - r["r4"]["kept_per_s1"], 5)
    r["shift_empty_share"] = round(r["A"]["empty_share"] - r["r4"]["empty_share"], 6)
    r["A_min_p_kept"] = float(P["saved"][m][fa].min())
    R[c] = r
    log(c, json.dumps(r))
    save()

# ---- France
m = MASK["france"]; fi, fj = ti[m], tj[m]; n = NS1["france"]


def share(keep):
    return 1 - len(np.unique(fi[keep])) / n


def calibrate(ef):
    s1v = share(ef.mask(1.0))
    if s1v >= TARGET:
        return 1.0, s1v, None
    s_cap = share(ef.mask(CAP))
    if s_cap <= TARGET:
        return CAP, s1v, s_cap
    lo, hi = 1.0, CAP
    for _ in range(25):
        mid = (lo + hi) / 2
        lo, hi = (mid, hi) if share(ef.mask(mid)) < TARGET else (lo, mid)
    return hi, s1v, s_cap


fr = {}
for tag, pk, rk, pr in (("stage4", "saved", RULE4, P["saved"]), ("stack_r4", "stack", BUNDLE["rule"], p_stack)):
    ef = EF(fi, PE[pk][m], rk["missed"])
    sc, s_before, s_cap = calibrate(ef); k0 = ef.mask(sc)
    k1, cnt = unseen_rules.correct(recs, fi, fj, pr[m], k0)
    ref = FA[m] if tag == "stage4" else F4[m]
    fr[tag] = {"none_scale": round(sc, 4), "empty_share_scale1": round(s_before, 6), "empty_share_at_cap": None if s_cap is None else round(s_cap, 6),
               "empty_share_after_calibration": round(share(k0), 6), "replay_counts": cnt, "replay_diff_vs_file": int((k1 != ref).sum()),
               "tie_cut_groups": ef.tie_cut_groups, "file": {**stats(m, ref, "france")}}
    if tag == "stage4":
        K0, K1 = k0, k1
    log("france", tag, json.dumps(fr[tag]))
R["france"] = fr; save()

# per-rule accounting: file (A France) vs own pre-rule decision K0
Fa = FA[m]; pr = P["saved"][m]
drop, add = np.flatnonzero(K0 & ~Fa), np.flatnonzero(~K0 & Fa)
col = lambda c_, idx: recs[c_].gather(idx)
# rule 1 condition on every pre-rule keep
k = np.flatnonzero(K0)
hrel = unseen_rules.house_relation(col("house", fi[k]), col("house", fj[k]), col("addr", fj[k]))
anch = np.isin(fi[k], fi[k][hrel == "eq"])
cond1 = np.isin(hrel, ["edit1", "off20"]) & anch & unseen_rules.legal_form_conflict(col("name", fi[k]), col("name", fj[k]))
cond1_pairs = k[cond1]
# adds: own classification from names
top = np.zeros(int(fj.max()) + 1); np.maximum.at(top, fj, pr)
TVW, tails = set(unseen_rules.TRUE_VARIANT_WORDS), sorted(set(unseen_rules.SWAP_TAILS + unseen_rules.APPEND_TAILS), key=len, reverse=True)
ci, cj = col("core", fi[add]).to_list(), col("core", fj[add]).to_list()
ni, nj = col("name", fi[add]).to_list(), col("name", fj[add]).to_list()
hadd = unseen_rules.house_relation(col("house", fi[add]), col("house", fj[add]), col("addr", fj[add]))
kind = []
for a, b, x, y in zip(ci, cj, ni, nj):
    sa, sb = set(a.split(" ")), set(b.split(" "))
    tv = sa <= sb and len(sb) == len(sa) + 1 and len(sb - sa) == 1 and next(iter(sb - sa)) in TVW
    tl = any(y.endswith(" " + t) for t in tails)
    kind.append("tv" if tv else ("tail" if tl else "other"))
kind = np.array(kind)
after_drop = K0.copy(); after_drop[drop] = False
s1_keeps_after_drop = np.isin(fi[add], fi[after_drop])
acc = {"pre_rule_keeps": int(K0.sum()), "file_keeps": int(Fa.sum()),
       "dropped": len(drop), "dropped_meeting_rule1_condition": int(np.isin(drop, cond1_pairs).sum()),
       "pre_rule_keeps_meeting_rule1_condition": len(cond1_pairs),
       "dropped_s1_rows": len(np.unique(fi[drop])),
       "added": len(add), "added_true_variant": int((kind == "tv").sum()), "added_tail": int((kind == "tail").sum()),
       "added_other": int((kind == "other").sum()),
       "added_tv_s1_rows": len(np.unique(fi[add][kind == "tv"])), "added_tail_s1_rows": len(np.unique(fi[add][kind == "tail"])),
       "added_tv_prob_ge_0.15": int((pr[add][kind == "tv"] >= unseen_rules.TRUE_VARIANT_MIN_PROB).sum()),
       "added_same_house_eq": int((hadd == "eq").sum()),
       "added_is_owner": int((pr[add] >= top[fj[add]]).sum()),
       "added_s1_already_keeps": int(s1_keeps_after_drop.sum()),
       "added_active": int(act[m][add].sum()), "added_prob_lt_0.002": int((pr[add] < 0.002).sum()),
       "s1_rows_changed_by_post_rules": len(np.unique(np.r_[fi[drop], fi[add]])),
       "empty_share_pre_rules": round(share(K0), 6), "empty_share_file": round(share(Fa), 6),
       "empty_rows_file": n - len(np.unique(fi[Fa])),
       "rows_emptied_by_drops": int(len(np.unique(fi[K0])) - len(np.unique(fi[after_drop])))}
R["france_post_rules"] = acc
log("france post rules", json.dumps(acc)); save()

# ============ (6) label-free flip profile: CE sign of the pairs A adds / drops vs r4 (test) and stage4_A vs v7 (holdout)
cel = pl.read_parquet(CE / "infer/out/test_ce.parquet")
cet = T.select("i", "j").join(cel, on=["i", "j"], how="left", maintain_order="left")["ce_logit"].to_numpy()
del cel
cet = np.where(np.isnan(cet.astype(np.float64)), np.nan, cet)


def prof(x):
    x = x[~np.isnan(x)]
    return {"n_with_ce": len(x), "ce_pos_share": round(float((x > 0).mean()), 4) if len(x) else None,
            "ce_median": round(float(np.median(x)), 3) if len(x) else None}


flip = {}
for c in ("us", "india", "france"):
    mm = MASK[c]; a, b = FA[mm], F4[mm]; x = cet[mm]
    flip[c] = {"added": {"pairs": int((a & ~b).sum()), **prof(x[a & ~b])}, "dropped": {"pairs": int((~a & b).sum()), **prof(x[~a & b])},
               "kept_both": prof(x[a & b])}
if True:
    mm = MASK["france"]; x = cet[mm]; a0 = K0
    flip["france_pre_rules_vs_r4"] = {"added": prof(x[a0 & ~F4[mm]]), "dropped": prof(x[~a0 & F4[mm]]),
                                      "added_pairs": int((a0 & ~F4[mm]).sum()), "dropped_pairs": int((~a0 & F4[mm]).sum())}
R["test_flip_ce_profile"] = flip
HK = pl.read_parquet(CE / "stack/out/holdout_keeps.parquet", columns=["i", "j", "v7_production", "stage4_A"])
HP = pl.read_parquet(CE / "setup/out/holdout_pairs.parquet", columns=["i", "j", "y", "country"])
assert (HK["i"] == HP["i"]).all() and (HK["j"] == HP["j"]).all()
hc = HK.with_columns(HP["y"], HP["country"].cast(pl.String)).join(pl.read_parquet(CE / "infer/out/holdout_ce.parquet"), on=["i", "j"], how="left", maintain_order="left")
hflip = {}
for c in ("us", "india"):
    h = hc.filter(pl.col("country") == c)
    ad, dr = h.filter(pl.col("stage4_A") & ~pl.col("v7_production")), h.filter(~pl.col("stage4_A") & pl.col("v7_production"))
    hflip[c] = {"added": {"pairs": ad.height, "true_rate": round(float(ad["y"].mean()), 4), **prof(ad["ce_logit"].cast(pl.Float64).fill_null(np.nan).to_numpy())},
                "dropped": {"pairs": dr.height, "true_rate": round(float(dr["y"].mean()), 4), **prof(dr["ce_logit"].cast(pl.Float64).fill_null(np.nan).to_numpy())}}
R["holdout_flip_ce_profile"] = hflip
log("flip profiles", json.dumps(flip), json.dumps(hflip))
save(); log("done")
