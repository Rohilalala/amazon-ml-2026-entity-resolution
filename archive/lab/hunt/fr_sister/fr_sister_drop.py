"""France (unseen-country) sister-organisation decoy filter, LB-confirmed (strict mode on the v3+recall file: 0.986 -> 0.987).
A kept pair is dropped when, at the S1's own house number, the candidate's name equals the S1's name except that ONE word is
replaced by a different word that is not a typo of it (SequenceMatcher ratio < 0.6), not a generator tail word (the true-copy
endings fils/cie/compagnie/services/developpement/groupe/france/associes) and not cie<->compagnie. Legal forms are ignored.
strict : only if the S1 also keeps an exact-core copy, drop count < kept count.
relaxed: only if the S1 keeps at least one other pair that is not dropped (no exact copy needed); also matches a one-word swap
         after removing trailing tail words from the candidate (e.g. 'x club' vs 'x comite france').
v2     : both swapped words must be common France name words (>= MINF occurrences as name tokens in the France pool, so scrambled
         typos like cbbu/club stay), house numbers compared as integers (1 == 001), any address relation, and an S1 whose kept
         pairs are ALL sister swaps is emptied (those S1 look like true singletons that only matched decoys).
v3     : v2, but only the CANDIDATE's word must be a common France name word (typos/scrambles only occur on the S2/S3 side;
         the S1 name is the clean original, so a rare S1 word is fine).
usage: fr_sister_drop.py BASE_TSV OUT_DIR strict|relaxed|v2|v3 [MINF]"""
import sys, re, unicodedata, difflib, collections, os, json
import polars as pl
R = "/Users/adityarohilla/Documents/amazon-ml-2026/"; T = R + "student_resource/dataset/test/"
BASE, OUT, MODE = sys.argv[1], sys.argv[2], sys.argv[3]; MINF = int(sys.argv[4]) if len(sys.argv) > 4 else 20
rd = lambda f: pl.read_csv(T + f, separator="\t", quote_char=None, infer_schema=False).with_columns(pl.col("entity_id").str.strip_chars())
s1 = rd("test_source1.tsv").filter(pl.col("country").str.strip_chars().str.to_lowercase() == "france")
pool = pl.concat([rd("test_source2.tsv"), rd("test_source3.tsv")]).filter(pl.col("country").str.strip_chars().str.to_lowercase() == "france")
LF = {"sarl", "sas", "sasu", "sa", "eurl", "snc", "sci", "scop", "selarl"}
TAIL = {"fils", "cie", "compagnie", "services", "developpement", "groupe", "france", "associes", "et", "and"}
def norm(s): s = unicodedata.normalize("NFKD", s or "").encode("ascii", "ignore").decode().lower(); return re.sub(r"[^a-z0-9 ]", " ", s)
def core(n): return [t for t in norm(n).split() if t not in LF]
def house(a):
    m = re.search(r"\b(\d+)\b", norm(a))
    return (str(int(m.group(1))) if MODE in ("v2", "v3") else m.group(1)) if m else None
N = dict(zip(pool["entity_id"].to_list(), pool["business_name"].to_list())); A = dict(zip(pool["entity_id"].to_list(), pool["business_address"].fill_null("").to_list()))
FREQ = collections.Counter(t for n in pool["business_name"].to_list() for t in core(n))
def swap(c1, c2):
    if len(c1) != len(c2) or len(c1) < 2: return None
    d = [(a, b) for a, b in zip(c1, c2) if a != b]
    if len(d) != 1: return None
    a, b = d[0]
    if not (a.isalpha() and b.isalpha() and len(a) >= 3 and len(b) >= 3): return None
    if difflib.SequenceMatcher(None, a, b).ratio() >= 0.6 or b in TAIL or {a, b} <= {"cie", "compagnie"}: return None
    if MODE == "v2" and (FREQ[a] < MINF or FREQ[b] < MINF): return None
    if MODE == "v3" and FREQ[b] < MINF: return None
    return (a, b)
base = pl.read_csv(BASE, separator="\t", quote_char=None, infer_schema=False)
fr = base.rename({"source1_entity_id": "entity_id"}).join(s1.select("entity_id", "business_name", "business_address"), on="entity_id")
drops, cnt = {}, collections.Counter()
for r in fr.iter_rows(named=True):
    ids = [x for x in (r["matched_entity_ids"] or "").split(",") if x]
    if len(ids) < (1 if MODE in ("v2", "v3") else 2): continue
    c1, h1 = core(r["business_name"]), house(r["business_address"])
    if not h1 and MODE not in ("v2", "v3"): continue
    cores = {i: core(N.get(i, "")) for i in ids}
    if MODE == "strict" and not any(cores[i] == c1 for i in ids): continue
    bad = []
    for i in ids:
        if MODE not in ("v2", "v3") and house(A.get(i, "")) != h1: continue
        c2 = cores[i]; s = swap(c1, c2)
        if s is None and MODE in ("relaxed", "v2", "v3"):
            c2t = list(c2)
            while len(c2t) > len(c1) and c2t and c2t[-1] in TAIL: c2t.pop()
            s = swap(c1, c2t)
        if s: bad.append(i); cnt[s] += 1
    if bad and (len(bad) < len(ids) or MODE in ("v2", "v3")): drops[r["entity_id"]] = set(bad)
nd = sum(len(v) for v in drops.values())
out = base.with_columns(pl.struct("source1_entity_id", "matched_entity_ids").map_elements(
    lambda s: ",".join(x for x in (s["matched_entity_ids"] or "").split(",") if x and x not in drops.get(s["source1_entity_id"], ())), return_dtype=pl.String).alias("matched_entity_ids"))
os.makedirs(OUT, exist_ok=True); out.write_csv(OUT + "/matching_results.tsv", separator="\t", quote_style="never")
emptied = sum(1 for r in fr.iter_rows(named=True) if r["entity_id"] in drops and len(drops[r["entity_id"]]) == len([x for x in (r["matched_entity_ids"] or "").split(",") if x]))
rep = {"mode": MODE, "emptied_s1": emptied, "minf": MINF, "base": BASE, "drop_pairs": nd, "drop_s1": len(drops), "per_france_s1": round(nd / s1.height, 4), "top": cnt.most_common(15)}
open(OUT + "/fr_sister.json", "w").write(json.dumps(rep, indent=1)); print(json.dumps(rep)[:900])
