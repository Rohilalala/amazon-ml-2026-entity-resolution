"""Row-level comparison of the assembled files with day1b/stack/r4 per country (TSV level, independent of b01):
S1 rows whose matched list differs and matched-id counts. -> out/b03.json"""
import json
from pathlib import Path
import polars as pl
ROOT = Path("/Users/adityarohilla/Documents/amazon-ml-2026"); D = ROOT / "v8_lab/ce/build"
s1 = pl.read_csv(ROOT / "student_resource/dataset/test/test_source1.tsv", separator="\t", quote_char=None, infer_schema=False)
s1 = s1.select(pl.col("entity_id").str.strip_chars().alias("s1"), pl.col("country").str.strip_chars().str.to_lowercase().alias("c"))


def load(p):
    m = pl.read_csv(p, separator="\t", quote_char=None, infer_schema=False)
    return m.select(pl.col("source1_entity_id").str.strip_chars().alias("s1"),
                    pl.col("matched_entity_ids").fill_null("").str.split(",").list.eval(pl.element().filter(pl.element() != "")).list.sort().alias("m"))


base = s1.join(load(ROOT / "v8_lab/day1b/stack/r4/matching_results.tsv"), on="s1", how="left")
R = {}
for k in ("A_all", "B_usin", "C_fr"):
    x = base.join(load(D / k / "matching_results.tsv").rename({"m": "m2"}), on="s1", how="left")
    R[k] = {c: {"s1": g.height, "rows_changed_vs_r4": int((g["m"] != g["m2"]).sum()),
                "matched_ids": int(g["m2"].list.len().sum()), "matched_ids_r4": int(g["m"].list.len().sum()),
                "empty_rows": int((g["m2"].list.len() == 0).sum())}
            for (c,), g in x.group_by(["c"], maintain_order=True)}
(D / "out/b03.json").write_text(json.dumps(R, indent=1))
print(json.dumps(R, indent=1))
