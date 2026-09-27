"""Label check on the HOLDOUT for the test-side shifts: per country, stage4_A vs v7_production keeps
(stack/out/holdout_keeps.parquet): kept/S1, empty share, flip share of active pairs, S1 rows changed, house-variant
keeps/S1 (audit_a common.relation hrel edit1/off20) and the true-pair rate of added/dropped pairs. -> out/b04.json"""
import json, sys
from pathlib import Path
import numpy as np, polars as pl
ROOT = Path("/Users/adityarohilla/Documents/amazon-ml-2026"); CE = ROOT / "v8_lab/ce"
sys.path.insert(0, str(ROOT / "v8_lab/audit_a"))
from common import relation
K = pl.read_parquet(CE / "stack/out/holdout_keeps.parquet", columns=["i", "j", "v7_production", "stage4_A"])
H = pl.read_parquet(CE / "setup/out/holdout_pairs.parquet", columns=["i", "j", "y", "country", "active"])
assert (K["i"] == H["i"]).all() and (K["j"] == H["j"]).all()
K = K.with_columns(H["y"], H["country"].cast(pl.String), H["active"])
S = pl.read_parquet(CE / "setup/out/holdout_s1.parquet")
recs = pl.read_parquet(ROOT / "cache/58f0a60c5c264fdd/recs.parquet", columns=["row", "core", "addr", "house"])
assert (recs["row"].to_numpy() == np.arange(recs.height)).all()
U = K.filter(pl.col("v7_production") | pl.col("stage4_A"))
U = U.with_columns(hv=relation(U.select("i", "j"), recs)["hrel"].is_in(["edit1", "off20"]))
R = {}
for c in ("us", "india"):
    n = int((S["country"].cast(pl.String) == c).sum()); k = K.filter(pl.col("country") == c); u = U.filter(pl.col("country") == c)
    v, s, act = k["v7_production"].to_numpy(), k["stage4_A"].to_numpy(), k["active"].to_numpy()
    flip = v != s; i = k["i"].to_numpy()
    add, drop = u.filter(pl.col("stage4_A") & ~pl.col("v7_production")), u.filter(~pl.col("stage4_A") & pl.col("v7_production"))
    R[c] = {"s1": n, "flip_share_of_active": round(float(flip[act].sum() / act.sum()), 6), "flips_outside_active": int(flip[~act].sum()),
            "s1_rows_changed_share": round(len(np.unique(i[flip])) / n, 6),
            "pairs_added": add.height, "pairs_dropped": drop.height,
            "added_true_rate": round(float(add["y"].mean()), 4), "dropped_true_rate": round(float(drop["y"].mean()), 4),
            "hv_per_s1": {"v7_production": round(float(u.filter("v7_production")["hv"].sum() / n), 5),
                          "stage4_A": round(float(u.filter("stage4_A")["hv"].sum() / n), 5)},
            "hv_added": int(add["hv"].sum()), "hv_added_true_rate": round(float(add.filter("hv")["y"].mean()), 4) if add["hv"].any() else None,
            "hv_dropped": int(drop["hv"].sum()), "hv_dropped_true_rate": round(float(drop.filter("hv")["y"].mean()), 4) if drop["hv"].any() else None}
    R[c]["hv_shift_per_s1"] = round(R[c]["hv_per_s1"]["stage4_A"] - R[c]["hv_per_s1"]["v7_production"], 5)
    print(c, json.dumps(R[c]), flush=True)
(CE / "build/out/b04.json").write_text(json.dumps(R, indent=1))
