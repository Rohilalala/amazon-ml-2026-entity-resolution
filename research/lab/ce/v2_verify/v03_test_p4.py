"""Verifier (3) TEST stage-4 probabilities on a random sample: random TEST S1 (seed 20260927) taken until their active
pairs reach 200k (all active pairs of a sampled S1 are kept, so the within-S1 features are exact); features recomputed by
feat_v.py from test_pairs prob, v2 TEST logits (joined on (i, j)), text_test (own regex) and the test cache records;
mean of the 3 saved boosters; compared with v2_stacker/out_v2ce/test_ACNRST_3s_v2ce_prob.parquet (joined on (i, j)).
usage (repo venv): v03_test_p4.py -> out/v03_test_p4.json"""
import json, os, resource, sys, threading, time
from pathlib import Path
import numpy as np, polars as pl, lightgbm as lgb

HERE = Path(__file__).parent; sys.path.insert(0, str(HERE))
import feat_v
ROOT = feat_v.ROOT; CE = ROOT / "v8_lab/ce"; SET = CE / "setup/out"; VS = CE / "v2_stacker"; OUT = HERE / "out"


def guard(limit=float(os.environ.get("RSS_LIMIT_GB", 3.2)) * 1e9):
    def run():
        while True:
            if resource.getrusage(resource.RUSAGE_SELF).ru_maxrss > limit:
                print("RSS guard tripped", flush=True); os._exit(3)
            time.sleep(0.5)
    threading.Thread(target=run, daemon=True).start()


guard(); T0 = time.time(); R = {}
log = lambda *a: print(f"[{time.time() - T0:6.1f}s]", *a, flush=True)
A = (pl.scan_parquet(SET / "test_pairs.parquet").filter(pl.col("prob") >= 0.002)
     .select("i", "j", "prob", pl.col("country").cast(pl.String)).collect())
R["active_recomputed"] = A.height
cnt = A.group_by("i").len().sort("i")
rng = np.random.default_rng(20260927)
o = rng.permutation(cnt.height); c = np.cumsum(cnt["len"].to_numpy()[o])
take = cnt["i"].to_numpy()[o[:int(np.searchsorted(c, 200_000)) + 1]]
A = A.filter(pl.col("i").is_in(pl.Series(take).implode()))
R["sample"] = {"s1": len(take), "pairs": A.height, "by_country": dict(A.group_by("country").len().iter_rows())}
log("sample", R["sample"])
A = A.join(pl.read_parquet(CE / "v2_encoder/out/TEST_ce.parquet"), on=["i", "j"], how="left")
A = A.join(pl.read_parquet(VS / "out_v2ce/test_ACNRST_3s_v2ce_prob.parquet"), on=["i", "j"], how="left")
assert A["ce_logit"].null_count() == 0 and A["p4"].null_count() == 0
ai, aj = A["i"].to_numpy(), A["j"].to_numpy()
rows = pl.DataFrame({"row": np.unique(np.r_[ai, aj])})
tx = pl.scan_parquet(SET / "text_test.parquet").join(rows.lazy(), on="row", how="semi").select(
    "row", e=pl.col("text").str.extract(r"(?s)^.*? \| (.*)$", 1).fill_null("").str.strip_chars() == "").collect()
assert tx.height == rows.height
EMP = np.zeros(int(rows["row"].max()) + 1, bool); EMP[tx["row"].to_numpy()] = tx["e"].to_numpy()
recs = pl.scan_parquet(ROOT / "cache/3ee42f05d06efbcb/recs.parquet").select("row", *feat_v.RECORD_COLUMNS).join(rows.lazy(), on="row", how="semi").collect()
assert recs.height == rows.height
ck = recs.sort("row"); rr = ck["row"].to_numpy(); cc = ck["country_key"].to_numpy()
R["pair_country_eq_recs"] = bool((cc[np.searchsorted(rr, ai)] == A["country"].to_numpy()).all() and (cc[np.searchsorted(rr, aj)] == A["country"].to_numpy()).all())
F = feat_v.features(ai, aj, A["prob"].to_numpy(), A["ce_logit"].to_numpy(), EMP, recs)
log("features")
meta = json.load(open(VS / "model/ACNRST_3s_v2ce.json")); cols = meta["features"]
p = np.mean([lgb.Booster(model_file=str(VS / "model" / m)).predict(feat_v.X(F, cols)) for m in meta["models"]], axis=0)
d = np.abs(p - A["p4"].to_numpy()); ctry = A["country"].to_numpy()
R["p4_check"] = {"pairs": len(d), "max_abs_diff": float(d.max()), "n_diff_gt_1e-9": int((d > 1e-9).sum()), "n_diff_gt_1e-6": int((d > 1e-6).sum()),
                 "exact_equal": int((d == 0).sum()),
                 **{c: {"pairs": int((ctry == c).sum()), "max_abs_diff": float(d[ctry == c].max()), "n_diff_gt_1e-6": int((d[ctry == c] > 1e-6).sum())} for c in ("us", "india", "france")}}
if (d > 1e-6).any():  # which features differ from a pair whose p4 differs (for the report)
    k = np.flatnonzero(d > 1e-6)[:20]
    R["p4_diff_examples"] = [{"i": int(ai[x]), "j": int(aj[x]), "mine": float(p[x]), "saved": float(A["p4"][int(x)])} for x in k]
R["p4_mean_sample"] = {c: round(float(p[ctry == c].mean()), 5) for c in ("us", "india", "france")}
R["peak_rss_gb"] = round(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1e9, 2); R["seconds"] = round(time.time() - T0)
(OUT / "v03_test_p4.json").write_text(json.dumps(R, indent=1))
log("done", R["p4_check"], R["peak_rss_gb"])
