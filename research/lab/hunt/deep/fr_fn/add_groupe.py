"""France 'groupe' tail-copy adds (+ optional sister-scramble restores) on top of the best file.
A France candidate j is added to S1 i when: j is in the test candidate set and kept by no S1; the candidate's core equals the
S1 core minus its last word followed by 'groupe'; in the candidate name 'groupe' comes after the legal form ('x sarl groupe');
both house numbers are equal and non-empty; the legal-form tokens of both names are equal; base prob > 0.5.
--nolegal also adds the same pattern for S1 names without a legal form ('x club' -> 'x groupe'; lower precision, see notes).
Scramble restore (--scramble): re-add pairs the sister filter dropped when the candidate's swapped word is rare in France
(< 100 core-token occurrences) and starts with the same letter as the S1 word (a typo, e.g. club -> cbu/ctb).
usage: add_groupe.py OUT_DIR [--scramble] [--addr90] [--nolegal]"""
import sys, os, json
import polars as pl
from rapidfuzz.fuzz import token_set_ratio
R = "/Users/adityarohilla/Documents/amazon-ml-2026/"
BEST = R + "v8_lab/hunt/final_adapt_all_sister/matching_results.tsv"
OUT = sys.argv[1]; SCR = "--scramble" in sys.argv; A90 = "--addr90" in sys.argv; NL = "--nolegal" in sys.argv
from tailfmt import STOP
def leg(n, c):
    cs = set(c.split()); return " ".join(sorted(t for t in n.split() if t not in cs and t not in STOP))
H = os.path.dirname(os.path.abspath(__file__))
fu = pl.read_parquet(H + "/fr_unclaimed_tailfmt.parquet")
g = fu.filter((pl.col("tw") == "groupe") & pl.col("fmt").is_in(["after_legal", "nolegal"] if NL else ["after_legal"]) & pl.col("hs") & (pl.col("prob") > 0.5))
g = g.filter(pl.struct("n1", "c1", "n2", "c2").map_elements(lambda r: leg(r["n1"], r["c1"]) == leg(r["n2"], r["c2"]), return_dtype=pl.Boolean))
if A90: g = g.filter(pl.struct("a1", "a2").map_elements(lambda r: token_set_ratio(r["a1"], r["a2"]) >= 90, return_dtype=pl.Boolean))
g = g.unique(["i", "j"]).unique("j", keep="first")  # a candidate goes to one S1 only
recs = pl.read_parquet(R + "cache/3ee42f05d06efbcb/recs.parquet", columns=["row", "entity_id"])
add = g.select("i", "j").join(recs.rename({"row": "i", "entity_id": "s1"}), on="i").join(recs.rename({"row": "j", "entity_id": "m"}), on="j").select("s1", "m")
n_g = add.height
if SCR:
    d = pl.read_parquet(H + "/sister_drops.parquet").filter((pl.col("fb") < 100) & (pl.col("wa").str.slice(0, 1) == pl.col("wb").str.slice(0, 1)))
    add = pl.concat([add, d.select(pl.col("source1_entity_id").alias("s1"), pl.col("matched_entity_ids").alias("m"))])
base = pl.read_csv(BEST, separator="\t", quote_char=None, infer_schema=False)
ex = base.with_columns(pl.col("matched_entity_ids").fill_null("").str.split(",")).explode("matched_entity_ids").filter(pl.col("matched_entity_ids") != "")
ex = pl.concat([ex, add.rename({"s1": "source1_entity_id", "m": "matched_entity_ids"})]).unique(maintain_order=True)
out = base.select("source1_entity_id").join(ex.group_by("source1_entity_id", maintain_order=True).agg(pl.col("matched_entity_ids").str.join(",")), on="source1_entity_id", how="left").with_columns(pl.col("matched_entity_ids").fill_null(""))
assert out.height == base.height and out["source1_entity_id"].n_unique() == base.height
os.makedirs(OUT, exist_ok=True)
out.write_csv(OUT + "/matching_results.tsv", separator="\t", quote_style="never")
n_before = ex.height - (add.height)
info = {"groupe_adds": n_g, "scramble_adds": add.height - n_g, "total_pairs_before": base.with_columns(pl.col("matched_entity_ids").fill_null("").str.split(",").list.eval(pl.element().filter(pl.element() != "")).list.len().alias("n"))["n"].sum(), "total_pairs_after": ex.height, "s1_touched": add["s1"].n_unique()}
json.dump(info, open(OUT + "/info.json", "w"), indent=1); print(info)
