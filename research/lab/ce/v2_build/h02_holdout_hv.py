"""Holdout label check of the TEST-side shifts, as build/b04_holdout_hv.py for stage4_A: per country, ACNRST_3s_v2ce vs
stage4_A keeps (out/holdout_keeps.parquet): kept/S1, empty share, pairs added / dropped with their true-pair rate,
house-variant keeps/S1 (common.relation hrel edit1 / off20) and the true rate of added / dropped house variants.
usage (repo venv): h02_holdout_hv.py -> out/h02.json"""
import json, sys
from pathlib import Path
import numpy as np, polars as pl
ROOT = Path("/Users/adityarohilla/Documents/amazon-ml-2026"); CE = ROOT / "v8_lab/ce"; D = Path(__file__).parent
sys.path.insert(0, str(CE / "stack"))
import s01_stage4 as s4
sys.path.insert(0, str(ROOT / "v8_lab/audit_a"))
from common import relation
s4._rss_guard()
A, B = "stage4_A", "ACNRST_3s_v2ce"
K = pl.read_parquet(D / "out/holdout_keeps.parquet", columns=["i", "j", A, B])
H = pl.read_parquet(CE / "setup/out/holdout_pairs.parquet", columns=["i", "j", "y", "country"])
assert (K["i"] == H["i"]).all() and (K["j"] == H["j"]).all()
K = K.with_columns(H["y"], H["country"].cast(pl.String)).filter(pl.col(A) | pl.col(B))
S = pl.read_parquet(CE / "setup/out/holdout_s1.parquet")
rows = pl.concat([K["i"], K["j"]]).unique().to_frame("row")
recs = pl.scan_parquet(ROOT / "cache/58f0a60c5c264fdd/recs.parquet").select("row", "core", "addr", "house").join(rows.lazy(), on="row", how="semi").collect()
K = K.join(relation(K.select("i", "j"), recs).select("i", "j", hv=pl.col("hrel").is_in(["edit1", "off20"])), on=["i", "j"], how="left")
R = {}
for c in ("us", "india"):
    k, n = K.filter(pl.col("country") == c), int((S["country"] == c).sum())
    add, drop = k.filter(pl.col(B) & ~pl.col(A)), k.filter(pl.col(A) & ~pl.col(B))
    r = {"s1": n}
    for t in (A, B):
        kt = k.filter(pl.col(t))
        r[t] = {"kept_per_s1": round(kt.height / n, 5), "empty_share": round(1 - kt["i"].n_unique() / n, 5),
                "hv_per_s1": round(int(kt["hv"].sum()) / n, 5), "hv_true_rate": round(float(kt.filter("hv")["y"].mean()), 4)}
    r.update(pairs_added=add.height, added_true_rate=round(float(add["y"].mean()), 4), pairs_dropped=drop.height,
             dropped_true_rate=round(float(drop["y"].mean()), 4), hv_added=int(add["hv"].sum()),
             hv_added_true_rate=round(float(add.filter("hv")["y"].mean()), 4), hv_dropped=int(drop["hv"].sum()),
             hv_dropped_true_rate=round(float(drop.filter("hv")["y"].mean()), 4),
             hv_shift_per_s1=round(r[B]["hv_per_s1"] - r[A]["hv_per_s1"], 5))
    R[c] = r
    print(c, json.dumps(r), flush=True)
(D / "out/h02.json").write_text(json.dumps(R, indent=1))
