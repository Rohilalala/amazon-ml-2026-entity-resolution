"""France generator-tail ADD on top of the best file.
France true copies = S1 name with its LAST core word dropped and a tail appended (swap), or S1 name + tail (append);
tails: et/and fils, cie, services, developpement, groupe, france, et/and associes (near-uniform counts at equal house+street).
The stage-4 cross-encoder (trained on US/India, where 'group'/'associates' are decoy words) rejects 'groupe' swaps:
kept 0.16 at equal house+street vs 0.85-0.97 for the other six tails (and 0.94 in the day1b r4 stack).
Adds (i, j) when: France S1, j is a candidate of i, j's core = S1 core minus last word + TAIL (swap, S1 core >= 2 words) or
S1 core + TAIL (append), raw candidate name ends with TAIL, integer house equal, digit-stripped street token_set_ratio >= 85,
legal forms equal or both absent (a legal-form conflict/drop/add is the decoy signature: kept 0.00-0.04 for every tail),
j kept by no S1 in the base file and qualifying for exactly one S1, pair not already kept.
modes: R1 = swap 'groupe' only; R2 = R1 + swaps fils/associes/france + appends fils/associes/groupe/france (the tails whose France
keep rate sits far below the US/India analog; cie/services/developpement are already kept 0.94-1.0, their residual is decoy-rich).
usage: fr_tail_add.py BASE_TSV OUT_DIR R1|R2
Inputs built by rel.py, test_types.py, add_street.py in this folder (test_typed_st.parquet)."""
import sys, os, polars as pl
R = "/Users/adityarohilla/Documents/amazon-ml-2026/"; G = R + "v8_lab/hunt/deep/generator/"
BASE, OUT, MODE = sys.argv[1], sys.argv[2], sys.argv[3]
TAILS = ["fils", "cie", "services", "developpement", "groupe", "france", "associes"]
t = pl.read_parquet(G + "test_typed_st.parquet", columns=["i", "j", "country", "c1", "nt", "nd", "at", "st_sim"]).filter(pl.col("country") == "france")
recs = pl.scan_parquet(R + "cache/3ee42f05d06efbcb/recs.parquet").filter(pl.col("country_key") == "france").select("row", "entity_id", "name").collect()
base = pl.read_csv(BASE, separator="\t", quote_char=None, infer_schema=False)
kp = base.with_columns(pl.col("matched_entity_ids").str.split(",")).explode("matched_entity_ids") \
    .join(recs.select(pl.col("entity_id").alias("source1_entity_id"), pl.col("row").alias("i")), on="source1_entity_id") \
    .join(recs.select(pl.col("entity_id").alias("matched_entity_ids"), pl.col("row").alias("j")), on="matched_entity_ids").select("i", pl.col("j").cast(pl.UInt32))
t = t.with_columns(w=pl.col("nd").str.split(">").list.last()).join(recs.select(pl.col("row").alias("j"), pl.col("name").alias("n2")), on="j").join(recs.select(pl.col("row").alias("i"), pl.col("name").alias("n1")), on="i")
LF = ["sarl", "sas", "sasu", "sa", "eurl", "snc", "sci", "scop", "selarl", "ei"]
lfs = lambda c: pl.col(c).str.split(" ").list.eval(pl.element().filter(pl.element().is_in(LF))).list.unique().list.sort()
t = t.with_columns(lf_ok=lfs("n1") == lfs("n2"))
sig = (pl.col("at") == "same") & (pl.col("st_sim") >= 85) & pl.col("n2").str.ends_with(pl.lit(" ") + pl.col("w")) & (pl.col("c1").str.split(" ").list.len() >= 2) & pl.col("lf_ok")
swap = (pl.col("nt") == "swaplast") & pl.col("w").is_in(["groupe"] if MODE == "R1" else ["groupe", "fils", "associes", "france"])
app = pl.lit(False) if MODE == "R1" else (pl.col("nt") == "append") & pl.col("w").is_in(["fils", "associes", "groupe", "france"])
c = t.filter(sig & (swap | app)).join(kp, on=["i", "j"], how="anti").join(kp.select("j").unique(), on="j", how="anti")
c = c.filter(pl.col("j").count().over("j") == 1)
print(MODE, "adds", c.height, "S1", c["i"].n_unique(), c.group_by("nt", "w").len().sort("len", descending=True).rows())
adds = c.join(recs.select(pl.col("row").alias("i"), pl.col("entity_id").alias("s1")), on="i").join(recs.select(pl.col("row").alias("j"), pl.col("entity_id").alias("m")), on="j").group_by("s1").agg(pl.col("m"))
d = dict(zip(adds["s1"].to_list(), adds["m"].to_list()))
os.makedirs(OUT, exist_ok=True)
rows = []
for s, m in zip(base["source1_entity_id"].to_list(), base["matched_entity_ids"].to_list()):
    ids = [x for x in (m or "").split(",") if x] + d.get(s, [])
    rows.append((s, ",".join(ids)))
out = pl.DataFrame(rows, schema=["source1_entity_id", "matched_entity_ids"], orient="row")
assert out.height == base.height and out["source1_entity_id"].n_unique() == out.height
out.write_csv(OUT + "/matching_results.tsv", separator="\t", quote_style="never")
c.select("i", "j", "nt", "w", "c1", "n2").write_parquet(OUT + "/adds.parquet")
print("written", OUT, "pairs before", kp.height, "added", c.height)
