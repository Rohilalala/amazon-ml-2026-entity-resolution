"""Relation categories (house relation x name relation) for US/India pairs.
usage: cats.py holdout|test ; writes out/<side>_cats.parquet"""
import sys, difflib
import polars as pl
from rapidfuzz.fuzz import ratio
R = "/Users/adityarohilla/Documents/amazon-ml-2026/"; O = R + "v8_lab/hunt/deep/usin/out/"
side = sys.argv[1]
def hrel(a, b): return "miss" if not a or not b else ("same" if a.lstrip("0") == b.lstrip("0") else "diff")
def nrel(a, b):
    x, y = (a or "").split(), (b or "").split()
    if x == y: return "exact"
    if sorted(x) == sorted(y): return "reorder"
    sx, sy = set(x), set(y)
    if sx < sy: return "add%d" % min(len(sy - sx), 2)          # candidate has extra words
    if sy < sx: return "drop%d" % min(len(sx - sy), 2)         # candidate lost words
    if len(x) == len(y):
        d = [(p, q) for p, q in zip(x, y) if p != q]
        if len(d) == 1: return "typo1" if difflib.SequenceMatcher(None, *d[0]).ratio() >= 0.6 else "swap1"
    if len(x) == len(y) + 0 and x[:-1] == y[:-1]: return "lastswap"
    if len(x) >= 2 and y[:len(x) - 1] == x[:-1] and len(y) > len(x) - 1: return "droplast_add"
    r = ratio(a or "", b or "")
    return "fz90" if r >= 90 else "fz75" if r >= 75 else "fz50" if r >= 50 else "fzlo"
if side == "holdout":
    h = pl.read_parquet(R + "v8_lab/ce/setup/out/holdout_pairs.parquet", columns=["i", "j", "y", "active", "country"])
    h = h.with_columns(pl.read_parquet(R + "v8_lab/ce/v2_build/out/holdout_keeps.parquet", columns=["ACNRST_3s_v2ce"])["ACNRST_3s_v2ce"].alias("keep"))
    h = h.filter(pl.col("active") | (pl.col("y") == 1) | pl.col("keep"))
    recs = pl.read_parquet(R + "cache/58f0a60c5c264fdd/recs.parquet", columns=["row", "core", "house"])
else:
    s1 = pl.read_parquet(R + "v8_lab/ce/setup/out/test_s1.parquet").filter(pl.col("country").is_in(["us", "india"]))
    recs = pl.read_parquet(R + "cache/3ee42f05d06efbcb/recs.parquet", columns=["row", "entity_id", "core", "house"])
    best = pl.read_csv(R + "v8_lab/hunt/final_adapt_all_sister/matching_results.tsv", separator="\t", quote_char=None, infer_schema=False)
    e2r = recs.select("entity_id", "row")
    best = (best.with_columns(pl.col("matched_entity_ids").str.split(",")).explode("matched_entity_ids")
            .filter(pl.col("matched_entity_ids").is_not_null() & (pl.col("matched_entity_ids") != ""))
            .join(e2r.rename({"entity_id": "source1_entity_id", "row": "i"}), on="source1_entity_id")
            .join(e2r.rename({"entity_id": "matched_entity_ids", "row": "j"}), on="matched_entity_ids")
            .select("i", "j").with_columns(pl.lit(True).alias("keep")))
    p4 = pl.read_parquet(R + "v8_lab/hunt/plan/s4ce/out_adapt/test_ACNRST_3s_adapt_prob.parquet")
    h = p4.join(best, on=["i", "j"], how="full", coalesce=True).with_columns(pl.col("keep").fill_null(False))
    h = h.join(s1, on="i")  # US/India only
    recs = recs.drop("entity_id")
h = (h.join(recs.rename({"row": "i", "core": "c1", "house": "h1"}), on="i", how="left")
      .join(recs.rename({"row": "j", "core": "c2", "house": "h2"}), on="j", how="left"))
del recs
h = h.with_columns(pl.struct("h1", "h2").map_elements(lambda s: hrel(s["h1"], s["h2"]), return_dtype=pl.String).alias("hr"),
                   pl.struct("c1", "c2").map_elements(lambda s: nrel(s["c1"], s["c2"]), return_dtype=pl.String).alias("nr"))
import os; os.makedirs(O, exist_ok=True)
h.write_parquet(O + side + "_cats.parquet"); print(side, h.shape, h.group_by("hr", "nr").len().sort("len", descending=True).head(30))
