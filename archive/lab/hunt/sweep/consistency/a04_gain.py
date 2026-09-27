"""Holdout gain of 'add the unowned pool copies of a decided j' (paired bootstrap by S1)."""
import sys, json, numpy as np, polars as pl
from pathlib import Path
D = Path(__file__).parent; R = Path("/Users/adityarohilla/Documents/amazon-ml-2026")
sys.path.insert(0, str(R / "code/business_entity_resolution")); sys.path.insert(0, str(R / "v8_lab/ce/stack"))
from src.metric import f05_counts
from s01_stage4 import boot
S = pl.read_parquet(R / "v8_lab/ce/setup/out/holdout_s1.parquet").sort("i"); rows = S["i"].to_numpy(); nt = S["n_true"].to_numpy().astype(float)
ctry = S["country"].to_numpy()
C = pl.read_parquet(D / "hold_cand.parquet").filter("dec")
code = np.searchsorted(rows, C["i"].to_numpy()); kept = np.bincount(code, minlength=len(rows)).astype(float)
tp = np.bincount(code, weights=C["y"].to_numpy() == 1, minlength=len(rows))
f0 = f05_counts(kept, nt, tp); print("base", round(f0.mean(), 6))
out = {}
for key in ["k_text", "k_norm", "k_core"]:
    cop = pl.read_parquet(D / f"hold_copies_{key}.parquet").filter(pl.col("own").is_null())
    for name, f in {"all": pl.lit(True), "notcand": ~pl.col("incand"), "notcand_addr": ~pl.col("incand") & ~pl.col("e"), "addr": ~pl.col("e")}.items():
        a = cop.filter(f).unique(["i", "j2"]).filter(pl.col("j2").is_first_distinct())  # one S1 per j2 (exclusive)
        c = np.searchsorted(rows, a["i"].to_numpy())
        f1 = f05_counts(kept + np.bincount(c, minlength=len(rows)), nt, tp + np.bincount(c, weights=a["y2"].to_numpy(), minlength=len(rows)))
        out[f"{key}/{name}"] = {"adds": a.height, "true": int(a["y2"].sum()), "gain": boot(f1 - f0), "us": round(float((f1 - f0)[ctry == "us"].mean()), 6), "india": round(float((f1 - f0)[ctry == "india"].mean()), 6)}
        print(key, name, out[f"{key}/{name}"])
(D / "a04_gain.json").write_text(json.dumps(out, indent=1))
