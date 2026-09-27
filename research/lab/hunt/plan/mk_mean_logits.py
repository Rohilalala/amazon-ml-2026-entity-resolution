"""Mean of two encoders' logit files (rows asserted identical): OUT/{HOLDOUT,STAGE4,TEST}_ce.parquet, ce_logit = (a+b)/2.
usage (repo venv): mk_mean_logits.py DIR_A DIR_B OUT [HOLDOUT STAGE4 TEST]   (file names resolve case-insensitively on this disk)"""
import sys
from pathlib import Path
import numpy as np, polars as pl
A, B, O = (Path(x) for x in sys.argv[1:4]); O.mkdir(parents=True, exist_ok=True)
for ds in sys.argv[4:] or ("HOLDOUT", "STAGE4", "TEST"):
    a, b = pl.read_parquet(A / f"{ds}_ce.parquet"), pl.read_parquet(B / f"{ds}_ce.parquet")
    assert a.height == b.height and a["i"].equals(b["i"]) and a["j"].equals(b["j"]), ds
    z = ((a["ce_logit"].cast(pl.Float64) + b["ce_logit"].cast(pl.Float64)) / 2).cast(pl.Float32)
    assert z.is_finite().all()
    a.with_columns(ce_logit=z).write_parquet(O / f"{ds}_ce.parquet")
    print(ds, a.height, "pearson", round(float(np.corrcoef(a["ce_logit"].to_numpy(), b["ce_logit"].to_numpy())[0, 1]), 5), flush=True)
