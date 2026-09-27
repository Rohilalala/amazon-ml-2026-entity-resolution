"""Holdout: sibling similarity. For each candidate of an S1, its best name/address similarity to a KEPT candidate of the same S1
(other than itself). Tabulate truth by decision and similarity band."""
import sys, numpy as np, polars as pl
from pathlib import Path
from rapidfuzz import fuzz
D = Path(__file__).parent; R = Path("/Users/adityarohilla/Documents/amazon-ml-2026")
C = pl.read_parquet(D / "hold_cand.parquet").select("i", "j", "y", "p", "dec", "src")
rows = C["j"].unique()
rec = (pl.scan_parquet(R / "cache/58f0a60c5c264fdd/recs.parquet").select("row", "name", "addr", "house")
       .filter(pl.col("row").is_in(rows.implode())).collect())
K = C.filter("dec").select("i", k="j")
P = C.join(K, on="i").filter(pl.col("j") != pl.col("k"))
P = P.join(rec.rename({"row": "j"}), on="j").join(rec.rename({"row": "k", "name": "name_k", "addr": "addr_k", "house": "house_k"}), on="k")
print("pairs", P.height)
n1, n2, a1, a2 = (P[c].to_list() for c in ("name", "name_k", "addr", "addr_k"))
P = P.with_columns(sn=pl.Series([fuzz.token_set_ratio(x, y) for x, y in zip(n1, n2)]),
                   sa=pl.Series([fuzz.token_set_ratio(x, y) for x, y in zip(a1, a2)]),
                   sh=(pl.col("house") == pl.col("house_k")) & (pl.col("house") != ""))
P.select("i", "j", "k", "y", "p", "dec", "src", "sn", "sa", "sh").write_parquet(D / "hold_sib.parquet")
