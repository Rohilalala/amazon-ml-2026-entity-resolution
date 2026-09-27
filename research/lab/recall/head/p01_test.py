"""TEST head prob for the new pairs (no decision): h01_head.features on text_test + model/head.txt.
Also a drift check against the HOLDOUT: per country, new pairs per S1, reverse-only share, CE>0 share and pairs per S1
over the decision thresholds (before the owner rule).
usage (repo venv, under heavy.sh): p01_test.py -> out/test_head_prob.parquet (s1_row, j_row, country, source, prob), out/p01.json"""
import json, resource, time
import numpy as np, polars as pl, lightgbm as lgb
import h01_head as h

T0 = time.time()
meta = json.load(open(h.MOD / "head.json")); assert meta["features"] == h.FEATS
model = lgb.Booster(model_file=str(h.MOD / "head.txt"))
P = pl.read_parquet(h.PAIRS / "test_m1_r1_cap10.parquet")
C = pl.read_parquet(h.OUT / "TEST_ce.parquet")
assert (P["s1_row"] == C["i"]).all() and (P["j_row"] == C["j"]).all() and C["ce_logit"].is_finite().all()
F = h.features(P.with_columns(ce_logit=C["ce_logit"]), h.SET / "text_test.parquet")
F = F.with_columns(prob=pl.Series(model.predict(h.X(F))))
F.select("s1_row", "j_row", "country", "source", "prob").write_parquet(h.OUT / "test_head_prob.parquet")

V = pl.read_parquet(h.OUT / "holdout_add.parquet", columns=["s1_row", "j_row", "prob"]).join(
    h.features(pl.read_parquet(h.PAIRS / "holdout_m1_r1_cap10.parquet").with_columns(
        ce_logit=pl.read_parquet(h.OUT / "HOLDOUT_ce.parquet")["ce_logit"]), h.SET / "text_train.parquet"),
    on=["s1_row", "j_row"], maintain_order="left")
n_s1 = {"test": pl.read_parquet(h.SET / "test_s1.parquet").group_by("country").len(),
        "holdout": pl.read_parquet(h.SET / "holdout_s1.parquet").group_by("country").len()}
R = {}
for name, D in (("holdout", V), ("test", F)):
    D = D.with_columns(pl.col("country").cast(pl.String)).join(n_s1[name].with_columns(pl.col("country").cast(pl.String)), on="country")
    R[name] = D.group_by("country").agg(
        s1=pl.col("len").first(), pairs=pl.len(), new_per_s1=pl.len() / pl.col("len").first(),
        rev_only_share=pl.col("rev_only").mean(), ce_gt0_share=(pl.col("ce") > 0).mean(),
        **{f"per_s1_p_ge_{t}": (pl.col("prob") >= t).sum() / pl.col("len").first() for t in (0.5, meta["t_none"], meta["t_keep"], 0.9)},
        n_new_mean=pl.col("n_new").mean()).sort("country").to_dicts()
R["seconds"] = round(time.time() - T0); R["peak_rss_gb"] = round(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1e9, 2)
(h.OUT / "p01.json").write_text(json.dumps(R, indent=1, default=float))
print(json.dumps(R, indent=1, default=float))
