"""v2 logit files vs v1's: identical (i, j) rows in the same order, all finite; logit agreement summary.
usage (ce venv): c01_rows.py [HOLDOUT STAGE4 TEST] -> out/c01_rows.json (merged)"""
import json, sys
from pathlib import Path
import numpy as np, polars as pl
HERE = Path(__file__).parent; CE = HERE.parent; OUT = HERE / "out"
R = json.loads((OUT / "c01_rows.json").read_text()) if (OUT / "c01_rows.json").exists() else {}
for ds in sys.argv[1:] or ["HOLDOUT", "STAGE4", "TEST"]:
    a, b = pl.read_parquet(CE / f"infer/out/{ds.lower()}_ce.parquet"), pl.read_parquet(OUT / f"{ds}_ce.parquet")
    assert a.height == b.height and a["i"].equals(b["i"]) and a["j"].equals(b["j"]) and b["ce_logit"].is_finite().all()
    z1, z2 = a["ce_logit"].to_numpy().astype(np.float64), b["ce_logit"].to_numpy().astype(np.float64)
    R[ds] = {"rows": b.height, "rows_equal_v1": True, "all_finite": True, "pearson_v1_v2": round(float(np.corrcoef(z1, z2)[0, 1]), 5),
             "frac_gt0_v1": round(float((z1 > 0).mean()), 4), "frac_gt0_v2": round(float((z2 > 0).mean()), 4),
             "sign_agree": round(float(((z1 > 0) == (z2 > 0)).mean()), 5)}
    print(ds, R[ds], flush=True)
(OUT / "c01_rows.json").write_text(json.dumps(R, indent=1))
