"""Re-score every test pair of one country with v7 stages 1-2 (and stage 3), capping the stage-1 feature
amb_s23_addr_2 at 1 (all pairs, or only house-variant pairs), then decide that country exactly as v7-final does
(owner rule + expected-F0.5; stage-2 for US/France, stage 3 for India) and save kept (s1, m) id pairs for mix.py.
Pairs of other countries never interact (blocking is per country), so per-country re-scoring is exact.
usage: g01_rescore_country.py COUNTRY {all|variant|none} [decide_col: stage2|full]"""
import sys, time, tempfile
import numpy as np, polars as pl
from common import *
from src import alignment, features, pipeline

country, mode = sys.argv[1], sys.argv[2]
decide_col = sys.argv[3] if len(sys.argv) > 3 else ("full" if country == "india" else "stage2")
t0 = time.time()
bundle = pipeline.Bundle.load(REPO / "artifacts/v7")
st1, st2, st3 = bundle.stages
tables = pool_tables_country(TEST_CACHE, country)
print("pool tables", round(time.time() - t0), "peak", round(pipeline.peak_gb(), 2), flush=True)
recs_all = pl.scan_parquet(TEST_CACHE / "recs.parquet")
rows = recs_all.filter(pl.col("country_key") == country).select("row").collect()["row"]
recs = recs_all.filter(pl.col("country_key") == country).collect().sort("row")
remap = np.full(int(rows.max()) + 1, -1, dtype=np.int64); remap[recs["row"].to_numpy()] = np.arange(recs.height)
recs_l = recs.with_columns(pl.col("row").alias("orig_row"), pl.Series("row", np.arange(recs.height, dtype=np.uint32)))
P = pl.scan_parquet(TEST_CACHE / "pairs.parquet").filter(pl.col("i").is_in(rows.implode())).collect()
Pl = P.with_columns(pl.Series("i", remap[P["i"].to_numpy()].astype(np.uint32)), pl.Series("j", remap[P["j"].to_numpy()].astype(np.uint32)))
n = Pl.height
print("pairs", n, "records", recs.height, flush=True)
k_amb = st1.features.index("amb_s23_addr_2")
CH = 500_000
with tempfile.TemporaryDirectory(prefix="audit-g01-", dir=str(OUT)) as tmp:
    X1 = np.lib.format.open_memmap(f"{tmp}/x1.npy", mode="w+", dtype=np.float32, shape=(n, len(st1.features)))
    p1 = np.empty(n, dtype=np.float32)
    for s in range(0, n, CH):
        ch = Pl.slice(s, CH)
        f = features.pair_features(recs_l, ch, tables)
        x = pipeline.matrix(f, st1.features)
        if mode == "all":
            x[:, k_amb] = np.minimum(x[:, k_amb], 1.0)
        elif mode == "variant":
            rel = relation(ch.select("i", "j"), recs_l.select("row", "core", "addr", "house"))
            sel = rel["hrel"].is_in(["edit1", "off20", "far"]).to_numpy()
            x[sel, k_amb] = np.minimum(x[sel, k_amb], 1.0)
        X1[s:s + len(x)] = x
        p1[s:s + len(x)] = st1.predict(x)
        if s // CH % 5 == 0:
            print("stage1 chunk", s, round(time.time() - t0), "s peak", round(pipeline.peak_gb(), 2), flush=True)
    ctx = features.probability_context(Pl.select("i", "j").with_columns(pl.Series("p1", p1)))
    extra = ctx.select(st2.features[len(st1.features):]).to_numpy().astype(np.float32)
    del ctx
    p2 = np.empty(n, dtype=np.float32)
    for s in range(0, n, CH):
        p2[s:s + CH] = st2.predict(np.hstack([X1[s:s + CH], extra[s:s + CH]]))
    del X1, extra
print("stage2 done", round(time.time() - t0), "s peak", round(pipeline.peak_gb(), 2), flush=True)
if mode == "none":
    saved = pl.read_parquet(REPO / "artifacts/v7/test_scores/pairs.parquet", columns=["i", "j", "prob_stage2"]).filter(pl.col("i").is_in(rows.implode()))
    chk = P.select("i", "j").with_columns(pl.Series("p2", p2)).join(saved, on=["i", "j"], how="left")
    print("VERIFY full-country p2 vs saved: max abs diff", float((chk["p2"] - chk["prob_stage2"]).abs().max()), flush=True)
prob = p2
if decide_col == "full":
    prob = pipeline.stage3_scores(recs_l, Pl.select("i", "j"), p2, st3)
i, j = Pl["i"].to_numpy(), Pl["j"].to_numpy()
top = pl.DataFrame({"j": j, "p": prob}).select(pl.col("p").max().over("j"))["p"].to_numpy()
pe = np.where(prob >= top, prob, 0.0)
keep = pipeline.expected_f05_mask(i, pe, bundle.rule["missed"]) & (pe > 0)
if country == "france":  # unseen-country calibration to the seen countries' empty share of v7-final (US 0.05597)
    n1 = int((recs["src"] == 1).sum()); target = 1 - 625990 / 663106  # exact US empty share of v7-final
    def es(sc):
        return 1 - len(np.unique(i[pipeline.expected_f05_mask(i, pe, 0.0, sc) & (pe > 0)])) / n1
    lo, hi = 1.0, 4.0
    if es(1.0) < target:
        if es(hi) > target:
            for _ in range(25):
                mid = (lo + hi) / 2
                lo, hi = (mid, hi) if es(mid) < target else (lo, mid)
        keep = pipeline.expected_f05_mask(i, pe, 0.0, hi) & (pe > 0)
        print("france none_scale", hi, flush=True)
ids = recs["entity_id"].to_numpy()
kept = pl.DataFrame({"s1": ids[i[keep]], "m": ids[j[keep]]})
# consistency check of this decision code: the same code on the SAVED probabilities must give v7-final's rows
if country != "france":
    scol = "prob_full" if decide_col == "full" else "prob_stage2"
    sv = P.select("i", "j").join(pl.read_parquet(REPO / "artifacts/v7/test_scores/pairs.parquet", columns=["i", "j", scol]), on=["i", "j"], how="left")[scol].to_numpy()
    tops = pl.DataFrame({"j": j, "p": sv}).select(pl.col("p").max().over("j"))["p"].to_numpy()
    pes = np.where(sv >= tops, sv, 0.0)
    ks = pipeline.expected_f05_mask(i, pes, bundle.rule["missed"]) & (pes > 0)
    ref = kept_pairs("output_v7_final", recs.select("row", "entity_id")).filter(pl.col("i").is_in(recs["row"].implode()))
    mine = pl.DataFrame({"i": recs["row"].to_numpy()[i[ks]], "j": recs["row"].to_numpy()[j[ks]]})
    same = mine.join(ref, on=["i", "j"], how="anti").height == 0 and ref.join(mine, on=["i", "j"], how="anti").height == 0
    print("CHECK decision code on saved probabilities reproduces output_v7_final rows:", same, mine.height, ref.height, flush=True)
tag = f"{country}_{mode}_{decide_col}"
kept.write_parquet(OUT / f"g01_kept_{tag}.parquet")
P.select("i", "j").with_columns(pl.Series("p1", p1), pl.Series("p2", p2), pl.Series("prob", prob.astype(np.float32))).write_parquet(OUT / f"g01_scores_{tag}.parquet")
n1 = int((recs["src"] == 1).sum())
print(f"kept per S1 {keep.sum() / n1:.4f}, empty share {1 - len(np.unique(i[keep])) / n1:.5f}; total {round(time.time() - t0)} s peak {pipeline.peak_gb():.2f}", flush=True)
