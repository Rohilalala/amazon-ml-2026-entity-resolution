"""Independent file checks of recall/build/{ALL,USIN} against the best file (ce/v2_build/A_all_v2) and output_v7_final.
Streams all six TSVs in lockstep (same S1 order required) with plain Python; no build code imported.
Checks per row: same S1; new match list = best list + appended ids (prefix, no duplicates); every matched id inside the
file's own candidate row; candidate row = v7 row + appended new ids (disjoint from v7 row); every added id is one of the
appended (new) candidates; USIN France rows byte-identical to best/v7; USIN adds == ALL adds on US/India rows.
Global: no j matched to two S1 in any file. -> out/v01.json, out/adds_ALL.tsv (s1, jid, country)"""
import json, resource, sys, time
from collections import Counter
import polars as pl

R = "/Users/adityarohilla/Documents/amazon-ml-2026"; O = f"{R}/v8_lab/recall/verify/out"
F = {"best": f"{R}/v8_lab/ce/v2_build/A_all_v2/matching_results.tsv", "v7c": f"{R}/output_v7_final/candidate_pairs.tsv",
     "allm": f"{R}/v8_lab/recall/build/ALL/matching_results.tsv", "allc": f"{R}/v8_lab/recall/build/ALL/candidate_pairs.tsv",
     "usm": f"{R}/v8_lab/recall/build/USIN/matching_results.tsv", "usc": f"{R}/v8_lab/recall/build/USIN/candidate_pairs.tsv"}
T0 = time.time()
rec = pl.read_parquet(f"{R}/cache/3ee42f05d06efbcb/recs.parquet", columns=["entity_id", "src", "country_key"]).filter(pl.col("src") == 1)
ctry = dict(zip(rec["entity_id"].str.strip_chars().to_list(), rec["country_key"].to_list())); del rec
fh = {k: open(v, encoding="utf-8") for k, v in F.items()}
hdr = {k: f.readline() for k, f in fh.items()}
assert hdr["best"] == hdr["allm"] == hdr["usm"] == "source1_entity_id\tmatched_entity_ids\n", hdr
assert hdr["v7c"] == hdr["allc"] == hdr["usc"] == "source1_entity_id\tcandidate_entity_ids\n", hdr
sp = lambda s: s.split(",") if s else []
bad = Counter(); stats = {n: Counter() for n in ("ALL", "USIN")}; seen = {n: set() for n in ("best", "ALL", "USIN")}
dup = Counter(); adds_out = open(f"{O}/adds_ALL.tsv", "w"); adds_out.write("s1\tjid\tcountry\n")
rows = 0
while True:
    L = {k: f.readline() for k, f in fh.items()}
    if not L["best"]:
        assert all(v == "" for v in L.values()), "files end at different rows"; break
    rows += 1
    P = {k: v.rstrip("\n").split("\t") for k, v in L.items()}
    s1 = P["best"][0]
    if any(p[0] != s1 or len(p) != 2 for p in P.values()): bad["s1_or_cols_mismatch"] += 1; continue
    c = ctry.get(s1)
    if c is None: bad["s1_unknown"] += 1
    b, v7 = sp(P["best"][1]), sp(P["v7c"][1])
    for j in b:
        if j in seen["best"]: dup["best"] += 1
        seen["best"].add(j)
    v7s = set(v7)
    added = {}
    for n, mk, ck in (("ALL", "allm", "allc"), ("USIN", "usm", "usc")):
        m, cd = sp(P[mk][1]), sp(P[ck][1])
        st = stats[n]
        if m[:len(b)] != b: bad[f"{n}_not_prefix_of_best"] += 1; continue
        x = m[len(b):]
        if len(set(m)) != len(m): bad[f"{n}_dup_in_row"] += 1
        if cd[:len(v7)] != v7: bad[f"{n}_cand_not_prefix_of_v7"] += 1
        cx = cd[len(v7):]
        if len(set(cd)) != len(cd): bad[f"{n}_cand_dup_in_row"] += 1
        cs, cxs = set(cd), set(cx)
        if any(j not in cs for j in m): bad[f"{n}_match_outside_candidates"] += 1
        if any(j not in cxs for j in x): bad[f"{n}_add_not_new_candidate"] += 1
        if any(j in v7s for j in cx): bad[f"{n}_new_cand_already_v7"] += 1
        if any(not (j.startswith("S2-") or j.startswith("S3-")) for j in x): bad[f"{n}_add_bad_prefix"] += 1
        for j in m:
            if j in seen[n]: dup[n] += 1
            seen[n].add(j)
        st[f"new_cands_{c}"] += len(cx); st[f"added_{c}"] += len(x); st[f"rows_changed_{c}"] += bool(x)
        st["matched_ids"] += len(m); st["empty_rows"] += not m; st["candidate_ids"] += len(cd)
        if (P[mk][1] == P["best"][1]) != (not x): bad[f"{n}_row_text_differs_without_add"] += 1
        added[n] = x
    if c == "france":
        if L["usm"] != L["best"]: bad["USIN_france_match_row_changed"] += 1
        if L["usc"] != L["v7c"]: bad["USIN_france_cand_row_changed"] += 1
    elif added.get("USIN") != added.get("ALL"): bad["USIN_adds_differ_from_ALL_on_us_india"] += 1
    for j in added.get("ALL", []): adds_out.write(f"{s1}\t{j}\t{c}\n")
    if rows % 400000 == 0: print(f"[{time.time()-T0:.0f}s] {rows}", flush=True)
adds_out.close()
out = {"rows": rows, "violations": dict(bad), "j_matched_to_two_s1": dict(dup), "best_matched_ids": len(seen["best"]),
       "stats": {n: dict(s) for n, s in stats.items()}, "seconds": round(time.time() - T0),
       "peak_rss_gb": round(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1e9, 2)}
out["PASS"] = rows == 1732544 and not bad and not any(dup.values())
json.dump(out, open(f"{O}/v01.json", "w"), indent=1); print(json.dumps(out, indent=1))
