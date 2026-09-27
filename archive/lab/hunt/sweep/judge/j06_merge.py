"""Merge the accepted TEST add lists into final_adds.parquet (s1, m, rule-free) and write the empty final_drops.parquet.
Accepted: France cie->compagnie (628), France groupe/france non-last tail swaps (1,306), India unscored pool copies (140).
Drops: none accepted."""
import polars as pl
S = "/Users/adityarohilla/Documents/amazon-ml-2026/v8_lab/hunt/sweep/"
L = {"fr_cie": S + "france/adds_cie.parquet", "fr_tailmid": S + "france/adds_tailmid.parquet", "in_copy": S + "consistency/add_copy_india.parquet"}
A = pl.concat([pl.read_parquet(p).select("s1", "m").with_columns(rule=pl.lit(k)) for k, p in L.items()])
print(A.group_by("rule").len().sort("rule").rows(), "total", A.height)
assert A.unique(["s1", "m"]).height == A.height and A["m"].is_unique().all(), "a pair or an m appears in two lists"
A.select("s1", "m").write_parquet(S + "judge/final_adds.parquet")
A.write_parquet(S + "judge/final_adds_rule.parquet")
pl.DataFrame(schema={"s1": pl.String, "m": pl.String}).write_parquet(S + "judge/final_drops.parquet")
