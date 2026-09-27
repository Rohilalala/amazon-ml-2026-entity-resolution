"""Build a leaderboard file whose rows come from different versions per country (mixture probe).
usage: mix.py OUT_DIR us=output_v7_final india=output_v7_final france=EMPTY [--cand]
A source is an output folder under the project root, EMPTY (no matches), or a .parquet of kept (s1, m) id pairs.
Writes OUT_DIR/matching_results.tsv (row order = test_source1.tsv) and checks the submission rules."""
import sys
from pathlib import Path
import polars as pl
ROOT = Path("/Users/adityarohilla/Documents/amazon-ml-2026")
out_dir = Path(sys.argv[1]); want_cand = "--cand" in sys.argv
spec = dict(a.split("=", 1) for a in sys.argv[2:] if a != "--cand")
s1 = pl.read_csv(ROOT / "student_resource/dataset/test/test_source1.tsv", separator="\t", quote_char=None, infer_schema=False)
s1 = s1.select(pl.col("entity_id").str.strip_chars().alias("s1"), pl.col("country").str.strip_chars().str.to_lowercase().alias("c"))
assert set(spec) == set(s1["c"].unique()), f"need a source for every country: {sorted(s1['c'].unique())}"
parts = []
for c, src in spec.items():
    ids = s1.filter(pl.col("c") == c).select("s1")
    if src == "EMPTY":
        rows = ids.with_columns(pl.lit("").alias("m"))
    elif src.endswith(".parquet"):
        k = pl.read_parquet(src).group_by("s1").agg(pl.col("m").unique().sort().str.join(","))
        rows = ids.join(k, on="s1", how="left").with_columns(pl.col("m").fill_null(""))
    else:
        m = pl.read_csv(ROOT / src / "matching_results.tsv", separator="\t", quote_char=None, infer_schema=False)
        m = m.select(pl.col("source1_entity_id").alias("s1"), pl.col("matched_entity_ids").fill_null("").alias("m"))
        rows = ids.join(m, on="s1", how="left").with_columns(pl.col("m").fill_null(""))
    parts.append(rows)
res = s1.select("s1").join(pl.concat(parts), on="s1", how="left", maintain_order="left")
assert res.height == s1.height and res["s1"].equals(s1["s1"]) and res["s1"].n_unique() == res.height
ex = res.select("s1", pl.col("m").str.split(",")).explode("m").filter(pl.col("m") != "")
assert not ex["m"].str.starts_with("S1-").any() and not ex.select(pl.struct("s1", "m").is_duplicated().any()).item()
valid = pl.concat([pl.read_csv(ROOT / f"student_resource/dataset/test/test_source{k}.tsv", separator="\t", quote_char=None,
                               infer_schema=False, columns=["entity_id"]) for k in (2, 3)])["entity_id"].str.strip_chars()
assert ex.filter(~pl.col("m").is_in(valid.implode())).height == 0, "unknown ids"
out_dir.mkdir(parents=True, exist_ok=True)
res.select(pl.col("s1").alias("source1_entity_id"), pl.col("m").alias("matched_entity_ids")).write_csv(
    out_dir / "matching_results.tsv", separator="\t", quote_style="never")
print("wrote", out_dir / "matching_results.tsv", "rows", res.height, "matched ids", ex.height, "spec", spec)

if want_cand:  # final-package candidate file: each country's candidate rows from the version its matches came from
    cparts = []
    for c, src in spec.items():
        folder = src if (src != "EMPTY" and not src.endswith(".parquet")) else "output_v7_final"
        cm = pl.read_csv(ROOT / folder / "candidate_pairs.tsv", separator="\t", quote_char=None, infer_schema=False)
        ids = s1.filter(pl.col("c") == c).select("s1")
        cparts.append(ids.join(cm.select(pl.col("source1_entity_id").alias("s1"), pl.col("candidate_entity_ids").fill_null("").alias("m")),
                               on="s1", how="left").with_columns(pl.col("m").fill_null("")))
        del cm
    cand = s1.select("s1").join(pl.concat(cparts), on="s1", how="left", maintain_order="left")
    cx = cand.select("s1", pl.col("m").str.split(",")).explode("m").filter(pl.col("m") != "")
    outside = ex.join(cx, on=["s1", "m"], how="anti").height
    assert outside == 0, f"{outside} matched ids missing from the candidate set"
    cand.select(pl.col("s1").alias("source1_entity_id"), pl.col("m").alias("candidate_entity_ids")).write_csv(
        out_dir / "candidate_pairs.tsv", separator="\t", quote_style="never")
    print("wrote", out_dir / "candidate_pairs.tsv", "candidate ids", cx.height)
