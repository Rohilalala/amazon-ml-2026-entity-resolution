"""France (unseen country) exact-copy recall adds on top of a decision file: add a pool record j to S1 i when j has the same
core name (lowercase, accents and punctuation stripped, legal forms ignored) and the same house number as i, no S1 in the file
keeps j, no other France S1 has that same (core, house), and the pool holds at most MAXG such copies (generic chain names excluded).
usage: fr_exact_add.py BASE_TSV OUT_DIR [MAXG] [empty_only]"""
import sys, re, unicodedata, collections, os, json
import polars as pl
R = "/Users/adityarohilla/Documents/amazon-ml-2026/"; T = R + "student_resource/dataset/test/"
BASE, OUT = sys.argv[1], sys.argv[2]; MAXG = int(sys.argv[3]) if len(sys.argv) > 3 else 6; EMPTY_ONLY = len(sys.argv) > 4 and sys.argv[4] == "empty_only"
rd = lambda f: pl.read_csv(T + f, separator="\t", quote_char=None, infer_schema=False).with_columns(pl.col("entity_id").str.strip_chars())
s1 = rd("test_source1.tsv").filter(pl.col("country").str.strip_chars().str.to_lowercase() == "france")
pool = pl.concat([rd("test_source2.tsv"), rd("test_source3.tsv")]).filter(pl.col("country").str.strip_chars().str.to_lowercase() == "france")
LF = {"sarl", "sas", "sasu", "sa", "eurl", "snc", "sci", "scop", "selarl", "ei"}
def norm(s): s = unicodedata.normalize("NFKD", s or "").encode("ascii", "ignore").decode().lower(); return re.sub(r"[^a-z0-9 ]", " ", s)
def core(n): return " ".join(t for t in norm(n).split() if t not in LF)
def house(a):
    m = re.search(r"\b(\d+)\b", norm(a)); return int(m.group(1)) if m else None
base = pl.read_csv(BASE, separator="\t", quote_char=None, infer_schema=False)
claimed = set(x for m in base["matched_entity_ids"].fill_null("").to_list() for x in m.split(",") if x)
idx = collections.defaultdict(list)
for e, n, a in zip(pool["entity_id"].to_list(), pool["business_name"].to_list(), pool["business_address"].fill_null("").to_list()):
    h = house(a)
    if h is not None: idx[(core(n), h)].append(e)
s1key = collections.Counter((core(n), house(a)) for n, a in zip(s1["business_name"].to_list(), s1["business_address"].fill_null("").to_list()))
M = dict(zip(base["source1_entity_id"].to_list(), base["matched_entity_ids"].fill_null("").to_list()))
adds = {}; stats = collections.Counter()
for e, n, a in zip(s1["entity_id"].to_list(), s1["business_name"].to_list(), s1["business_address"].fill_null("").to_list()):
    h = house(a)
    if h is None: continue
    k = (core(n), h); js = idx.get(k, [])
    if not js or len(js) > MAXG or s1key[k] != 1 or len(k[0]) < 3: continue
    cur = [x for x in M.get(e, "").split(",") if x]
    if EMPTY_ONLY and cur: continue
    free = [j for j in js if j not in claimed and j not in cur]
    if free:
        adds[e] = free; stats["s1_empty" if not cur else "s1_nonempty"] += 1; stats["pairs"] += len(free)
out = base.with_columns(pl.struct("source1_entity_id", "matched_entity_ids").map_elements(
    lambda s: ",".join([x for x in (s["matched_entity_ids"] or "").split(",") if x] + adds.get(s["source1_entity_id"], [])), return_dtype=pl.String).alias("matched_entity_ids"))
os.makedirs(OUT, exist_ok=True); out.write_csv(OUT + "/matching_results.tsv", separator="\t", quote_style="never")
rep = {"base": BASE, "maxg": MAXG, "empty_only": EMPTY_ONLY, **stats}; open(OUT + "/fr_exact.json", "w").write(json.dumps(rep, indent=1)); print(json.dumps(rep))
