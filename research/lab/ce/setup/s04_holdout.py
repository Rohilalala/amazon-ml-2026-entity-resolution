"""HOLDOUT: the 100k v7-holdout S1 (never trained on) scored with the v7 bundle exactly as the test set is scored:
per country, pool tables of the full train pool (audit_a pool_tables_country = alignment.pool_tables restricted by
country_key + global trigram IDF), stage 1 on the holdout pairs AND their complete j-neighbourhoods (every pair whose
S2/S3 record is a candidate of a holdout S1), stage-2 context from fold-average p1 over that set, stage 3 on the
holdout pairs. Production prob: US stage 2, India stage 3 'full'. prob_cap = same with amb_s23_addr_2 capped at 1
(the stack's test-time regime; the ldf swap is the identity on the train pool).
Checks: the 15k+15k S1 already scored this way by r3_regime (e_orig_*_15000_7__base: p2, pf) must match.
Run with the repo venv. usage: s04_holdout.py -> out/holdout_pairs.parquet, out/s04.json"""
import json, sys, time
from pathlib import Path
import numpy as np, polars as pl
sys.path.insert(0, "/Users/adityarohilla/Documents/amazon-ml-2026/v8_lab/audit_a")
from common import ROOT, REPO, TRAIN_CACHE, pool_tables_country
from src import pipeline, features

OUT = Path(__file__).parent / "out"
R3 = ROOT / "v8_lab/day1b/r3_regime/out"
THR, CH = 0.002, 500_000
t0 = time.time()
bundle = pipeline.Bundle.load(REPO / "artifacts/v7")
st1, st2, st3 = bundle.stages
k_amb = st1.features.index("amb_s23_addr_2")
hold = pl.read_parquet(OUT / "holdout_s1.parquet")
truth = pl.read_parquet(OUT / "truth_train.parquet")
res, parts = {}, []


def decide(i, j, prob):
    """Exclusive owner rule + expected-F0.5 top-k (missed 0), as v7-final / the stack decide."""
    top = pl.DataFrame({"j": j, "p": prob}).select(pl.col("p").max().over("j"))["p"].to_numpy()
    pe = np.where(prob >= top, prob, 0.0)
    return pipeline.expected_f05_mask(i, pe, bundle.rule["missed"]) & (pe > 0)


for country in ("us", "india"):
    h = hold.filter(pl.col("country") == country)
    tables = pool_tables_country(TRAIN_CACHE, country)
    lz = pl.scan_parquet(TRAIN_CACHE / "pairs.parquet")
    Pi = lz.filter(pl.col("i").is_in(h["i"].implode())).collect().sort("i", "j")
    Pn = lz.filter(pl.col("j").is_in(Pi["j"].unique().implode()) & ~pl.col("i").is_in(h["i"].implode())).collect()
    P, n_i = pl.concat([Pi, Pn]), Pi.height
    del Pn
    need = pl.concat([P["i"], P["j"]]).unique().sort()
    recs = pl.scan_parquet(TRAIN_CACHE / "recs.parquet").filter(pl.col("row").is_in(need.implode())).collect().sort("row")
    remap = np.full(int(recs["row"].max()) + 1, -1, dtype=np.int64); remap[recs["row"].to_numpy()] = np.arange(recs.height)
    recs_l = recs.with_columns(pl.Series("row", np.arange(recs.height, dtype=np.uint32)))
    PL = P.with_columns(pl.Series("i", remap[P["i"].to_numpy()].astype(np.uint32)), pl.Series("j", remap[P["j"].to_numpy()].astype(np.uint32)))
    n = PL.height
    print(country, "holdout S1", h.height, "pairs", n_i, "with j-neighbourhood", n, round(time.time() - t0), "s", flush=True)
    X1 = np.empty((n_i, len(st1.features)), dtype=np.float32)
    p1, p1c = np.empty(n, dtype=np.float32), np.empty(n, dtype=np.float32)
    for s in range(0, n, CH):
        x = pipeline.matrix(features.pair_features(recs_l, PL.slice(s, CH), tables), st1.features)
        p1[s:s + len(x)] = st1.predict(x)
        if s < n_i:
            X1[s:min(s + len(x), n_i)] = x[:n_i - s]
        x[:, k_amb] = np.minimum(x[:, k_amb], 1.0)
        p1c[s:s + len(x)] = st1.predict(x)
        if s // CH % 5 == 0:
            print(country, "stage1 chunk", s, round(time.time() - t0), "s peak", round(pipeline.peak_gb(), 2), flush=True)
    ij = PL.select("i", "j")
    out = {}
    for tag, pp in (("", p1), ("_cap", p1c)):
        extra = features.probability_context(ij.with_columns(pl.Series("p1", pp))).select(st2.features[len(st1.features):]) \
            .head(n_i).to_numpy().astype(np.float32)
        x1 = X1.copy()
        if tag:
            x1[:, k_amb] = np.minimum(x1[:, k_amb], 1.0)
        p2 = st2.predict(np.hstack([x1, extra]))
        pf = pipeline.stage3_scores(recs_l, ij.head(n_i), p2, st3)
        out[f"p2{tag}"], out[f"pf{tag}"] = p2.astype(np.float32), pf.astype(np.float32)
        out[f"prob{tag}"] = out[f"p2{tag}"] if country == "us" else out[f"pf{tag}"]
        del x1, extra
    df = (Pi.select("i", "j").with_columns(*[pl.Series(k, v) for k, v in out.items()], pl.lit(country).alias("country"))
          .join(truth.with_columns(pl.lit(1, dtype=pl.Int8).alias("y")), on=["i", "j"], how="left").with_columns(pl.col("y").fill_null(0)))
    # check against the r3_regime scoring of 15k holdout S1 of this country
    ref = pl.read_parquet(R3 / f"e_orig_{country}_15000_7__base.parquet", columns=["i", "j", "p2", "pf"]).rename({"p2": "r2", "pf": "rf"})
    chk = ref.join(df.select("i", "j", "p2", "pf"), on=["i", "j"], how="left")
    r = {"s1": h.height, "pairs": n_i, "neighbourhood_pairs": n, "positives": int(df["y"].sum()), "true_total": int(h["n_true"].sum()),
         "check_r3_pairs": chk.height, "check_r3_missing": int(chk["p2"].null_count()),
         "check_r3_max_abs_diff_p2": float((chk["p2"] - chk["r2"]).abs().max()), "check_r3_max_abs_diff_pf": float((chk["pf"] - chk["rf"]).abs().max())}
    codes = np.searchsorted(h["i"].to_numpy(), df["i"].to_numpy())
    for tag in ("", "_cap"):
        keep = decide(df["i"].to_numpy(), df["j"].to_numpy(), df[f"prob{tag}"].to_numpy())
        r[f"f05{tag or '_nocap'}"] = round(pipeline.macro_f05(codes, h["n_true"].to_numpy(), keep, df["y"].to_numpy()), 5)
        df = df.with_columns(pl.Series(f"kept{tag}", keep))
    res[country] = r
    print(country, json.dumps(r), round(time.time() - t0), "s peak", round(pipeline.peak_gb(), 2), flush=True)
    parts.append(df)
    del X1, P, PL, recs, recs_l, p1, p1c, Pi
hp = pl.concat(parts).with_columns((pl.col("prob") >= THR).alias("active"), (pl.col("prob_cap") >= THR).alias("active_cap"))
hp.write_parquet(OUT / "holdout_pairs.parquet")
n_s1 = hold.height
codes = np.searchsorted(hold["i"].to_numpy(), hp["i"].to_numpy())
res["all"] = {"f05_nocap": round(pipeline.macro_f05(codes, hold["n_true"].to_numpy(), hp["kept"].to_numpy(), hp["y"].to_numpy()), 5),
              "f05_cap": round(pipeline.macro_f05(codes, hold["n_true"].to_numpy(), hp["kept_cap"].to_numpy(), hp["y"].to_numpy()), 5),
              "pairs": hp.height, "active": int(hp["active"].sum()), "active_cap": int(hp["active_cap"].sum()),
              "true_in_cands": int(hp["y"].sum()), "true_total": int(hold["n_true"].sum()),
              "true_active": int((hp["active"] & (hp["y"] == 1)).sum()), "kept_outside_active": int((hp["kept"] & ~hp["active"]).sum()),
              "seconds": round(time.time() - t0)}
(OUT / "s04.json").write_text(json.dumps(res, indent=1))
print("done", json.dumps(res["all"]), "peak", round(pipeline.peak_gb(), 2))
