"""TEST stage-4 v2 prob for one saved variant (model/NAME.{txt,json}) with the same feature code as STAGE4/HOLDOUT
(feat.pair_inputs, feat.features, feat.string_inputs), computed over S1 buckets (i % K) to keep peak RSS < 3 GB.
Every feature of a model without group J is a function of the S1's own active pairs, so the buckets are exact; this is
asserted on the HOLDOUT (bucketed recomputation == out/f_holdout + out/s_holdout, the table the gate was scored on).
usage (repo venv): p03_test.py NAME   (model/NAME.json "models": seed members whose probabilities are averaged) -> out/test_NAME_prob.parquet (i, j, p4 float64; TEST active pairs in test_pairs order),
out/p03_NAME.json (per-group TEST compute time, holdout vs TEST feature quantiles per country)"""
import json, sys, time, resource
from pathlib import Path
import numpy as np, polars as pl, lightgbm as lgb
import feat
from feat import s4

HERE = Path(__file__).parent; OUT = HERE / "out"; NAME = sys.argv[1]; K = 20
STR_OUT = feat.V2S / "out"  # S / T tables do not depend on the encoder
if feat.V2CE:  # CE_V2=1: v2 encoder logits; tables and outputs in out_v2ce/
    OUT = HERE / f"out_{feat.TAG}"
META = json.load(open(HERE / f"model/{NAME}.json")); COLS = META["features"]
assert not set(COLS) & set(feat.GROUPS["J"]), "group J depends on the pool of S1 in the table, buckets would change it"
BOOSTS = [lgb.Booster(model_file=str(HERE / "model" / fn)) for fn in META.get("models", [f"{NAME}.txt"])]  # seed members, averaged
STR = "".join(g for g in "ST" if set(COLS) & set(feat.GROUPS[g]))  # string groups the model uses
T0 = time.time(); R = {"name": NAME, "features": COLS, "rule": META["rule"], "buckets": K}
(None if __import__('os').path.exists('/Users/adityarohilla/Documents/amazon-ml-2026/v8_lab/adapt/NO_GUARD') else s4._rss_guard())  # guard off while the machine is otherwise idle


def log(*a):
    print(f"[{time.time() - T0:7.1f}s]", *a, flush=True)


def build(a: pl.DataFrame, split: str, emp: np.ndarray, tm: dict, full: bool):
    """a: r, i, j, prob, ce_logit, country -> per S1 bucket: COLS, then the prob. full: returns the whole table (r order);
    else returns (r, i, j, country, p4) plus a 5% row sample of COLS for the distribution report (bounded memory)."""
    parts, samp = [], []
    for b in range(K):
        x = a.filter(pl.col("i") % K == b)
        t = time.time(); x = feat.pair_inputs(x, split); tm["pair_inputs_PRN"] += time.time() - t
        t = time.time(); x = feat.features(x, emp); tm["features_ACN"] += time.time() - t
        if STR:
            t = time.time(); x = feat.string_inputs(x, split, STR); tm["string_inputs_ST"] += time.time() - t
        t = time.time(); xm = s4.X(x, COLS); x = x.with_columns(p4=pl.Series(np.mean([m.predict(xm) for m in BOOSTS], axis=0))); del xm; tm["predict"] += time.time() - t
        if full:
            parts.append(x.select("r", "i", "j", "country", "p4", *COLS))
        else:
            parts.append(x.select("r", "i", "j", "country", "p4"))
            samp.append(x.select("country", *COLS).sample(fraction=0.05, seed=b))
        del x
    return pl.concat(parts).sort("r"), (pl.concat(samp) if samp else None)


def empty_address(path):
    """s01_stage4.empty_address() expression on any text file (as build/b01_stage4_test.empty_address)."""
    e = (pl.scan_parquet(path)
         .select("row", b=pl.col("text").str.splitn(" | ", 2).struct.field("field_1").fill_null("").str.strip_chars() == "")
         .collect(engine="streaming"))
    assert (e["row"].to_numpy() == np.arange(e.height)).all()
    return e["b"].to_numpy()


def quant(df, c):
    x = df[c].cast(pl.Float64).to_numpy(); fin = x[np.isfinite(x)]
    return {"mean": round(float(fin.mean()), 4), **{f"p{q}": round(float(np.quantile(fin, q / 100)), 4) for q in (1, 10, 50, 90, 99)},
            "nan_share": round(1 - len(fin) / len(x), 4)}


# ---- holdout: bucketed path == the gate's feature table
EMP = np.load(STR_OUT / "emp_train.npy")
H, SH = s4.load_holdout(np.zeros(1, bool))
if feat.V2CE:
    H = H.drop("ce_logit").join(pl.read_parquet(feat.V2CE / "HOLDOUT_ce.parquet"), on=["i", "j"], how="left", maintain_order="left")
a = H.with_row_index("r").filter("active").select("r", "i", "j", "prob", "ce_logit", "country")
tm = {"pair_inputs_PRN": 0.0, "features_ACN": 0.0, "string_inputs_ST": 0.0, "predict": 0.0}
FH, _ = build(a, "train", EMP, tm, True)
ext = {g: [c for c in COLS if c in feat.GROUPS[g]] for g in "ST"}
ref = pl.read_parquet(OUT / "f_holdout.parquet", columns=["r", *[c for c in COLS if not any(c in v for v in ext.values())]])
for g, sc in ext.items():
    if sc:
        ref = ref.hstack(pl.read_parquet(STR_OUT / f"{g.lower()}_holdout.parquet", columns=sc))
assert (FH["r"] == ref["r"]).all()
R["holdout_bucketed_equal"] = FH.select(COLS).cast(pl.Float64).fill_nan(-999.0).equals(ref.select(COLS).cast(pl.Float64).fill_nan(-999.0))
assert R["holdout_bucketed_equal"]
R["holdout_seconds"] = {k: round(v, 1) for k, v in tm.items()}
ph = H["prob"].to_numpy().astype(np.float64).copy(); ph[FH["r"].to_numpy()] = FH["p4"].to_numpy()
res, _, _ = s4.score_block(H, SH, {NAME: s4.decide(H["i"].to_numpy(), H["j"].to_numpy(), ph, META["rule"])})
R["holdout_f05_repro"] = res[NAME]
assert res[NAME] == json.load(open(OUT / f"eval_{NAME}.json"))["holdout_f05"][NAME]  # the gate's number, from this path
log("holdout bucketed == gate table", R["holdout_seconds"])
hq = {c: FH.filter(pl.col("country") == c) for c in ("us", "india")}
R["dist"] = {"holdout": {c: {f: quant(d, f) for f in COLS} for c, d in hq.items()}}
del H, SH, a, FH, ref, hq

# ---- TEST
t = time.time()
if not (STR_OUT / "emp_test.npy").exists():
    np.save(STR_OUT / "emp_test.npy", empty_address(s4.SET / "text_test.parquet"))
EMP_TE = np.load(STR_OUT / "emp_test.npy"); R["b_empty_test_records"] = int(EMP_TE.sum()); assert R["b_empty_test_records"] == 264623  # b01.json
C = pl.read_parquet(feat.V2CE / "TEST_ce.parquet" if feat.V2CE else s4.INF / "test_ce.parquet")
A = (pl.scan_parquet(s4.SET / "test_pairs.parquet").with_row_index("r").filter("active")
     .select("r", "i", "j", "prob", pl.col("country").cast(pl.String)).collect())
assert A.height == C.height and (A["i"] == C["i"]).all() and (A["j"] == C["j"]).all()
A = A.with_columns(ce_logit=C["ce_logit"]); del C
R["test_load_s"] = round(time.time() - t, 1)
tm = {"pair_inputs_PRN": 0.0, "features_ACN": 0.0, "string_inputs_ST": 0.0, "predict": 0.0}
F, SMP = build(A, "test", EMP_TE, tm, False)
assert (F["r"] == A["r"]).all()
del A
p4 = F["p4"].to_numpy()
R["test_seconds"] = {k: round(v, 1) for k, v in tm.items()}
R["test_active"] = {c: int((F["country"] == c).sum()) for c in ("us", "india", "france")}
log("test", R["test_seconds"], R["test_active"])
R["dist"]["test_5pct_sample"] = {c: {f: quant(SMP.filter(pl.col("country") == c), f) for f in COLS} for c in ("us", "india", "france")}
pl.DataFrame({"i": F["i"], "j": F["j"], "p4": p4}).write_parquet(OUT / f"test_{NAME}_prob.parquet")
R["p4_mean_by_country"] = {c: round(float(p4[(F["country"] == c).to_numpy()].mean()), 5) for c in ("us", "india", "france")}
R["peak_rss_gb"] = round(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1e9, 2); R["seconds"] = round(time.time() - T0)
(OUT / f"p03_{NAME}.json").write_text(json.dumps(R, indent=1))
log("done", R["peak_rss_gb"])
