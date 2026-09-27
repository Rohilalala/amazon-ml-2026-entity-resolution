"""Verifier (2) holdout gate, own code. For the combined model (ACNRST_3s_v2ce), Track B (ACNRST_3s, v1 logits) and
Track A (stage4_A_v2):
 - features recomputed by feat_v.py from the raw inputs (holdout prob, encoder logits, cache records, text), compared
   with the saved gate tables (out_v2ce/f_holdout + out/s_holdout + out/t_holdout);
 - probabilities from the saved boosters on both feature sets; own owner rule + own expected-F0.5 with the model's rule;
 - keep masks compared with v2_build/out/holdout_keeps.parquet; own per-S1 F0.5; own paired bootstrap (2,000, own seed).
Baseline = current stage 4 (stack/out/holdout_keeps.parquet stage4_A, independently recomputed by the earlier verifier).
usage (repo venv): v02_holdout.py -> out/v02_holdout.json"""
import json, os, resource, sys, threading, time
from pathlib import Path
import numpy as np, polars as pl, lightgbm as lgb

HERE = Path(__file__).parent; sys.path.insert(0, str(HERE))
import feat_v
from common_v import owner, EF, f05, boot
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
H = pl.read_parquet(SET / "holdout_pairs.parquet", columns=["i", "j", "y", "country", "prob", "active"])
S = pl.read_parquet(SET / "holdout_s1.parquet")
code = pl.DataFrame({"i": S["i"], "code": np.arange(S.height)})
hc = H.select("i").join(code, on="i", how="left", maintain_order="left")["code"]
assert hc.null_count() == 0
codes, y, nt = hc.to_numpy(), H["y"].to_numpy(), S["n_true"].to_numpy()
hi, hj, hp = H["i"].to_numpy(), H["j"].to_numpy(), H["prob"].to_numpy().astype(np.float64)
act = H["active"].to_numpy(); ra = np.flatnonzero(act)
ai, aj = hi[act], hj[act]

rows = pl.DataFrame({"row": np.unique(np.r_[ai, aj])})
tx = pl.scan_parquet(SET / "text_train.parquet").join(rows.lazy(), on="row", how="semi").select(
    "row", e=pl.col("text").str.extract(r"(?s)^.*? \| (.*)$", 1).fill_null("").str.strip_chars() == "").collect()
assert tx.height == rows.height
EMP = np.zeros(int(rows["row"].max()) + 1, bool); EMP[tx["row"].to_numpy()] = tx["e"].to_numpy()
del tx
R["b_empty_records_among_holdout_active_rows"] = int(EMP.sum())
recs = pl.scan_parquet(ROOT / "cache/58f0a60c5c264fdd/recs.parquet").select("row", *feat_v.RECORD_COLUMNS).join(rows.lazy(), on="row", how="semi").collect()


def logits(path):
    c = H.filter("active").select("i", "j").join(pl.read_parquet(path), on=["i", "j"], how="left", maintain_order="left")["ce_logit"]
    assert c.null_count() == 0
    return c.to_numpy()


CEL = {"v1": logits(CE / "infer/out/holdout_ce.parquet"), "v2": logits(CE / "v2_encoder/out/HOLDOUT_ce.parquet")}
OWN = {k: feat_v.features(ai, aj, hp[act], v, EMP, recs) for k, v in CEL.items()}
log("own features")


def table(fdir, cols):
    f = pl.read_parquet(fdir / "f_holdout.parquet", columns=["r", *[c for c in cols if c not in feat_v.S_COLS and not c.startswith(("chr_", "num_"))]])
    s = pl.read_parquet(VS / "out/s_holdout.parquet"); t = pl.read_parquet(VS / "out/t_holdout.parquet")
    f = f.hstack(s.select([c for c in cols if c in s.columns])).hstack(t.select([c for c in cols if c in t.columns]))
    assert (f["r"].to_numpy() == ra).all()
    return {c: f[c].cast(pl.Float64).fill_null(np.nan).to_numpy() for c in cols}


def cmp(own, tab, cols):
    out = {}
    for c in cols:
        a, b = own[c], tab[c]; na, nb = np.isnan(a), np.isnan(b)
        a32, b32 = a.astype(np.float32), b.astype(np.float32)
        out[c] = {"nan_pattern_diff": int((na != nb).sum()), "float32_diff": int((~na & ~nb & (a32 != b32)).sum()),
                  "max_abs": float(np.nanmax(np.abs(np.where(na | nb, 0, a - b))))}
    bad = {c: v for c, v in out.items() if v["nan_pattern_diff"] or v["float32_diff"]}
    return {"columns": len(cols), "columns_with_any_diff": bad, "max_abs_all": max(v["max_abs"] for v in out.values())}


keeps, probs = {}, {}
MODELS = {"ACNRST_3s_v2ce": ("v2", VS / "out_v2ce"), "ACNRST_3s": ("v1", VS / "out")}
for name, (enc, fdir) in MODELS.items():
    meta = json.load(open(VS / f"model/{name}.json")); cols = meta["features"]
    boosters = [lgb.Booster(model_file=str(VS / "model" / m)) for m in meta["models"]]
    assert all(b.num_feature() == len(cols) for b in boosters)
    tab = table(fdir, cols)
    R[f"{name}_features_own_vs_table"] = cmp(OWN[enc], tab, cols)
    pa = np.mean([b.predict(feat_v.X(OWN[enc], cols)) for b in boosters], axis=0)
    pb = np.mean([b.predict(feat_v.X(tab, cols)) for b in boosters], axis=0)
    R[f"{name}_p4_own_vs_table"] = {"max_abs": float(np.abs(pa - pb).max()), "n_gt_1e-9": int((np.abs(pa - pb) > 1e-9).sum())}
    p = hp.copy(); p[ra] = pa
    ef = EF(codes, owner(hj, p), meta["rule"]["missed"])
    keeps[name] = ef.mask(); probs[name] = p
    R[f"{name}_rule"] = meta["rule"]; R[f"{name}_trees"] = [b.num_trees() for b in boosters]; R[f"{name}_tie_cut_groups"] = ef.tie_cut_groups
    log(name, R[f"{name}_features_own_vs_table"]["columns_with_any_diff"], R[f"{name}_p4_own_vs_table"])
# Track A: set A on v2 logits, own features
metaA = json.load(open(CE / "v2_encoder/stack_model/stage4_A_v2.json"))
bA = lgb.Booster(model_file=str(CE / "v2_encoder/stack_model/stage4_A_v2.txt"))
p = hp.copy(); p[ra] = bA.predict(feat_v.X(OWN["v2"], metaA["features"]))
keeps["stage4_A_v2"] = EF(codes, owner(hj, p), metaA["rule"]["missed"]).mask(); R["stage4_A_v2_rule"] = metaA["rule"]
K = pl.read_parquet(CE / "stack/out/holdout_keeps.parquet", columns=["i", "j", "stage4_A", "v7_production"])
assert (K["i"].to_numpy() == hi).all() and (K["j"].to_numpy() == hj).all()
keeps["stage4_A"] = K["stage4_A"].to_numpy(); keeps["v7_production"] = K["v7_production"].to_numpy()
B = pl.read_parquet(CE / "v2_build/out/holdout_keeps.parquet")
assert (B["i"].to_numpy() == hi).all() and (B["j"].to_numpy() == hj).all()
R["keep_diff_vs_builder_h01"] = {k: int((keeps[k] != B[k].to_numpy()).sum()) for k in ("ACNRST_3s_v2ce", "ACNRST_3s", "stage4_A_v2", "stage4_A")}
ctry = S["country"].to_numpy()
f = {k: f05(codes, nt, v, y) for k, v in keeps.items()}
R["holdout_f05"] = {k: {"all": round(float(v.mean()), 6), **{c: round(float(v[ctry == c].mean()), 6) for c in ("us", "india")}} for k, v in f.items()}
R["pair_errors"] = {k: {"tp": int((v & (y == 1)).sum()), "fp": int((v & (y == 0)).sum()), "fn_in_cand": int((~v & (y == 1)).sum())} for k, v in keeps.items()}
R["paired"] = {}
for a, b in (("ACNRST_3s_v2ce", "stage4_A"), ("ACNRST_3s_v2ce", "stage4_A_v2"), ("ACNRST_3s_v2ce", "ACNRST_3s"),
             ("stage4_A_v2", "stage4_A"), ("ACNRST_3s", "stage4_A"), ("stage4_A", "v7_production")):
    d = f[a] - f[b]
    R["paired"][f"{a} - {b}"] = {"all": boot(d), **{c: boot(d[ctry == c]) for c in ("us", "india")},
                                "s1_better_worse": [int((d > 0).sum()), int((d < 0).sum())], "pairs_flipped": int((keeps[a] != keeps[b]).sum())}
    log(a, "-", b, R["paired"][f"{a} - {b}"]["all"])
g = R["paired"]["ACNRST_3s_v2ce - stage4_A"]["all"]
R["gate"] = {"diff": g["diff"], "ci95": g["ci95"], "pass": bool(g["diff"] >= 0.0003 and g["ci95"][0] > 0)}
R["peak_rss_gb"] = round(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1e9, 2); R["seconds"] = round(time.time() - T0)
(OUT / "v02_holdout.json").write_text(json.dumps(R, indent=1))
log("done", R["gate"], R["peak_rss_gb"])
