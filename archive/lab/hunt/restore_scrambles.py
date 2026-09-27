"""Re-add pairs the France sister filter dropped that are generator typo scrambles (candidate word rare in France, same first
letter) or abbreviations (one word a prefix of the other), when their j is kept by no S1 in the post-filter file.
usage: restore_scrambles.py PRE_TSV POST_TSV OUT_DIR"""
import sys, os, re, unicodedata, collections
import polars as pl
R = "/Users/adityarohilla/Documents/amazon-ml-2026/"; T = R + "student_resource/dataset/test/"
PRE, POST, OUT = sys.argv[1], sys.argv[2], sys.argv[3]
rd = lambda f: pl.read_csv(T + f, separator="\t", quote_char=None, infer_schema=False).with_columns(pl.col("entity_id").str.strip_chars())
s1 = rd("test_source1.tsv"); pool = pl.concat([rd("test_source2.tsv"), rd("test_source3.tsv")])
LF = {"sarl", "sas", "sasu", "sa", "eurl", "snc", "sci", "scop", "selarl"}
def core(n): n = unicodedata.normalize("NFKD", n or "").encode("ascii", "ignore").decode().lower(); return [t for t in re.sub(r"[^a-z0-9 ]", " ", n).split() if t not in LF]
fr = pool.filter(pl.col("country").str.strip_chars().str.to_lowercase() == "france")
F = collections.Counter(t for n in fr["business_name"].to_list() for t in core(n))
N1 = dict(zip(s1["entity_id"].to_list(), s1["business_name"].to_list())); N2 = dict(zip(pool["entity_id"].to_list(), pool["business_name"].to_list()))
def pairs(p): return pl.read_csv(p, separator="\t", quote_char=None, infer_schema=False).select(pl.col("source1_entity_id").alias("s"), pl.col("matched_entity_ids").fill_null("").str.split(",").alias("m")).explode("m").filter(pl.col("m") != "")
pre, post = pairs(PRE), pairs(POST)
dr = pre.join(post, on=["s", "m"], how="anti")
def scr(s, m):
    d = [(a, b) for a, b in zip(core(N1.get(s, "")), core(N2.get(m, ""))) if a != b]
    if len(d) != 1: return False
    a, b = d[0]; return (F[b] < 100 and a[:1] == b[:1]) or bool(a and b and (a.startswith(b) or b.startswith(a)))
claimed = set(post["m"].to_list())
add = [(s, m) for s, m in dr.iter_rows() if scr(s, m) and m not in claimed]
base = pl.read_csv(POST, separator="\t", quote_char=None, infer_schema=False)
ex = pl.concat([post, pl.DataFrame({"s": [a for a, _ in add], "m": [b for _, b in add]}, schema={"s": pl.String, "m": pl.String})])
assert ex["m"].n_unique() == ex.height
out = base.select("source1_entity_id").join(ex.group_by("s", maintain_order=True).agg(pl.col("m").str.join(",")).rename({"s": "source1_entity_id", "m": "matched_entity_ids"}), on="source1_entity_id", how="left").with_columns(pl.col("matched_entity_ids").fill_null(""))
assert out.height == base.height and (out["source1_entity_id"] == base["source1_entity_id"]).all()
os.makedirs(OUT, exist_ok=True); out.write_csv(OUT + "/matching_results.tsv", separator="\t", quote_style="never")
print("sister drops", dr.height, "restored scrambles", len(add))
