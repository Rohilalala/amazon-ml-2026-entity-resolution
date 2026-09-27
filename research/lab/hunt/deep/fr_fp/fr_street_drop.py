"""France different-street decoy filter (+ optional position-free sister swaps).
A France kept pair is dropped when the candidate has the S1's house number but a different street: none of the S1 street
words (len >= 4, street-type words removed) fuzzy-matches (ratio >= 0.6) any word of the candidate's numbered component,
nor any word anywhere in the candidate address. Generic 'City + OrgWord' names collide across entities in the same city.
usage: fr_street_drop.py BASE_TSV OUT_DIR [sister2]"""
import sys, os, re, json, collections
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import polars as pl
from load import rd
from addr import nums
from street2 import strict_diff
from street3 import diff_street
from cls import norm
STT = {"rue","r","avenue","av","ave","boulevard","bd","blvd","place","pl","impasse","imp","allee","all","chemin","chem","route","rte","quai","q",
       "cours","crs","square","sq","passage","pas","psg","mail","cite","sentier","sent","esplanade","promenade","parvis","rond","voie"}
def has_type(a):  # numbered address component names a street type (skips 'appt 18', 'hangar 18' style components)
    for p in (a or "").split(","):
        if re.search(r"\d", p): return bool(set(norm(p).split()) & STT)
    return False
BASE, OUT = sys.argv[1], sys.argv[2]; SIS = len(sys.argv) > 3 and sys.argv[3] == "sister2"
fr = lambda d: d.filter(pl.col("country").str.strip_chars().str.to_lowercase() == "france")
s1 = fr(rd("test", 1)); pool = fr(pl.concat([rd("test", 2), rd("test", 3)]))
N = dict(zip(pool["entity_id"], pool["business_name"])); A = dict(zip(pool["entity_id"], pool["business_address"].fill_null("")))
if SIS: from sis2 import swap
base = pl.read_parquet(BASE) if BASE.endswith(".parquet") else pl.read_csv(BASE, separator="\t", quote_char=None, infer_schema=False)
S = {r[0]: (r[1], r[2] or "") for r in s1.select("entity_id", "business_name", "business_address").iter_rows()}
drops, why = {}, collections.Counter()
for a, m in base.iter_rows():
    if a not in S or not m: continue
    n1, a1 = S[a]; h1 = nums(a1)
    for b in m.split(","):
        a2 = A.get(b, ""); h2 = nums(a2)
        if h1 and h2 and h1[0] == h2[0] and has_type(a1) and has_type(a2) and strict_diff(a1, a2) and diff_street(a1, a2):
            drops.setdefault(a, set()).add(b); why["street"] += 1
        elif SIS and swap(n1, N.get(b, "")):
            drops.setdefault(a, set()).add(b); why["sister2"] += 1
out = base.with_columns(pl.struct("source1_entity_id", "matched_entity_ids").map_elements(
    lambda s: ",".join(x for x in (s["matched_entity_ids"] or "").split(",") if x and x not in drops.get(s["source1_entity_id"], ())), return_dtype=pl.String).alias("matched_entity_ids"))
os.makedirs(OUT, exist_ok=True); out.write_csv(OUT + "/matching_results.tsv", separator="\t", quote_style="never")
kept = dict(base.iter_rows())
emptied = sum(1 for a, d in drops.items() if len(d) == len([x for x in kept[a].split(",") if x]))
rep = {"base": BASE, "sister2": SIS, "drop_pairs": sum(why.values()), "by_rule": dict(why), "drop_s1": len(drops), "emptied_s1": emptied}
open(OUT + "/fr_street.json", "w").write(json.dumps(rep, indent=1)); print(json.dumps(rep))
