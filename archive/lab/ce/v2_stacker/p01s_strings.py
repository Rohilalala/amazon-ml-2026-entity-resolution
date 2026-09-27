"""GROUPS["S"] (feat.string_inputs) for the STAGE4 and HOLDOUT active pairs, row-aligned with out/f_{stage4,holdout}.parquet.
usage (repo venv): p01s_strings.py stage4|holdout [S|T] -> out/{s,t}_{mode}.parquet, out/p01s_{G}_{mode}.json"""
import json, sys, time, resource
from pathlib import Path
import polars as pl
import feat
from feat import s4

OUT = Path(__file__).parent / "out"; MODE = sys.argv[1]; G = sys.argv[2] if len(sys.argv) > 2 else "S"
COLS = [c for g in G for c in feat.GROUPS[g]]
s4._rss_guard()
F = pl.read_parquet(OUT / f"f_{MODE}.parquet", columns=["i", "j"])
t = time.time(); S = feat.string_inputs(F, "train", G)
R = {"pairs": F.height, "string_inputs_s": round(time.time() - t, 1),
     "null_share": {c: round(S[c].null_count() / S.height, 4) for c in COLS if S[c].null_count()},
     "nan_share": {c: round(float(S[c].is_nan().mean()), 4) for c in COLS if S[c].is_nan().any()}}
S.select(COLS).write_parquet(OUT / f"{G.lower()}_{MODE}.parquet")
R["peak_rss_gb"] = round(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1e9, 2)
(OUT / f"p01s_{G}_{MODE}.json").write_text(json.dumps(R, indent=1)); print(R)
