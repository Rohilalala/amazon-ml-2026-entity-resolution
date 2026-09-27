"""PROBE (adds, not drops): France candidates that match the France true-copy generator pattern exactly
(S1 core minus its last word + one tail word: fils/cie/services/developpement/groupe/france/associes), at the S1's house
number with the same legal form, that stage 4 rejected. Labelled US/India analog of this class is 99.5% true.
Uses the classified active-pair table fr_act_r.parquet built in this directory. Candidate must be unassigned, S1 kept <= 7.
usage: fr_tailswap_add.py BASE_TSV OUT_DIR [TAILS comma list, default all; e.g. groupe]"""
import sys, os, json
import polars as pl
from cls import core
from street2 import strict_diff
from street3 import diff_street
D = os.path.dirname(os.path.abspath(__file__)); BASE, OUT = sys.argv[1], sys.argv[2]; TW = set(sys.argv[3].split(",")) if len(sys.argv) > 3 else None
base = pl.read_csv(BASE, separator="\t", quote_char=None, infer_schema=False).fill_null("")
used = {b for m in base["matched_entity_ids"] for b in m.split(",") if b}
kept = {a: [x for x in m.split(",") if x] for a, m in base.iter_rows()}
c = pl.read_parquet(D + "/fr_act_r.parquet").filter((pl.col("k") == "swap_tail") & (pl.col("h") == "same") & (pl.col("lr") == "lf_same") & ~pl.col("keep"))
tw = lambda n1, n2: [y for x, y in zip(core(n1), core(n2)) if x != y][0]
if TW: c = c.filter(pl.struct("n1", "n2").map_elements(lambda r: tw(r["n1"], r["n2"]) in TW, return_dtype=pl.Boolean))
c = c.filter(~pl.struct("ad1", "ad2").map_elements(lambda r: bool(strict_diff(r["ad1"], r["ad2"])) and bool(diff_street(r["ad1"], r["ad2"])), return_dtype=pl.Boolean))  # skip different-street decoys
c = c.filter(~pl.col("b").is_in(list(used))).sort("p4", descending=True).unique("b", keep="first", maintain_order=True)
add = {}
for a, b in c.select("a", "b").iter_rows():
    if a in kept and len(kept[a]) + len(add.get(a, [])) < 8: add.setdefault(a, []).append(b)
out = base.with_columns(pl.struct("source1_entity_id", "matched_entity_ids").map_elements(
    lambda s: ",".join([x for x in s["matched_entity_ids"].split(",") if x] + add.get(s["source1_entity_id"], [])), return_dtype=pl.String).alias("matched_entity_ids"))
os.makedirs(OUT, exist_ok=True); out.write_csv(OUT + "/matching_results.tsv", separator="\t", quote_style="never")
rep = {"base": BASE, "tails": sorted(TW) if TW else "all", "add_pairs": sum(map(len, add.values())), "add_s1": len(add)}
open(OUT + "/fr_tailswap_add.json", "w").write(json.dumps(rep)); print(rep)
