"""Independent row-level isolation check of build/{A_all,B_usin,C_fr} against day1b/stack/r4 (raw TSV lines and
id sets). -> out/v02.json"""
import json, csv
from pathlib import Path
ROOT = Path("/Users/adityarohilla/Documents/amazon-ml-2026"); B = ROOT / "v8_lab/ce/build"; OUT = ROOT / "v8_lab/ce/verify_build/out"
HDR = "source1_entity_id\tmatched_entity_ids"

ctry = {}
with open(ROOT / "student_resource/dataset/test/test_source1.tsv", encoding="utf-8", newline="") as f:
    r = csv.reader(f, delimiter="\t", quoting=csv.QUOTE_NONE)
    h = next(r); ki, kc = h.index("entity_id"), h.index("country")
    for row in r:
        ctry[row[ki].strip()] = row[kc].strip().lower()


def rows(p):
    L = Path(p).read_text(encoding="utf-8").split("\n")
    if L[-1] == "":
        L.pop()
    assert L[0].rstrip("\r") == HDR, (p, L[0])
    d, order = {}, []
    for line in L[1:]:
        a, _, b = line.rstrip("\r").partition("\t")
        a = a.strip(); assert a not in d; d[a] = b; order.append(a)
    return d, order


F = {k: rows(p) for k, p in (("r4", ROOT / "v8_lab/day1b/stack/r4/matching_results.tsv"), ("A", B / "A_all/matching_results.tsv"),
                             ("B", B / "B_usin/matching_results.tsv"), ("C", B / "C_fr/matching_results.tsv"))}
R = {"s1": len(ctry), "rows": {k: len(v[0]) for k, v in F.items()}}
for k, (d, order) in F.items():
    assert set(d) == set(ctry), k
R["same_row_order_ABC"] = F["A"][1] == F["B"][1] == F["C"][1]
R["same_row_order_A_r4"] = F["A"][1] == F["r4"][1]


def s(x):
    return frozenset(t.strip() for t in x.split(",") if t.strip())


def cmp(x, y):
    dx, dy = F[x][0], F[y][0]
    out = {}
    for c in ("us", "india", "france"):
        ids = [a for a in dx if ctry[a] == c]
        out[c] = {"s1": len(ids), "raw_diff": sum(dx[a] != dy[a] for a in ids), "set_diff": sum(s(dx[a]) != s(dy[a]) for a in ids)}
    return out


R["B_vs_r4"], R["C_vs_r4"], R["A_vs_r4"] = cmp("B", "r4"), cmp("C", "r4"), cmp("A", "r4")
R["A_vs_B"], R["A_vs_C"] = cmp("A", "B"), cmp("A", "C")
# A == B's US/India rows + C's France rows (raw strings)
R["A_equals_B_usin_plus_C_fr_raw"] = all(F["A"][0][a] == (F["C"][0][a] if ctry[a] == "france" else F["B"][0][a]) for a in ctry)
# per-country content: matched ids, empty rows, duplicate ids in a row
for k in ("r4", "A", "B", "C"):
    d = F[k][0]; st = {}
    for c in ("us", "india", "france"):
        ids = [a for a in d if ctry[a] == c]
        lists = [[t.strip() for t in d[a].split(",") if t.strip()] for a in ids]
        st[c] = {"matched_ids": sum(map(len, lists)), "empty_rows": sum(len(l) == 0 for l in lists),
                 "empty_share": round(sum(len(l) == 0 for l in lists) / len(ids), 6),
                 "kept_per_s1": round(sum(map(len, lists)) / len(ids), 5), "dup_in_row": sum(len(l) != len(set(l)) for l in lists)}
    R[f"content_{k}"] = st
# matched id used by two S1 rows (exclusivity) per file
for k in ("r4", "A", "B", "C"):
    seen, dup = set(), 0
    for a, b in F[k][0].items():
        for t in s(b):
            dup += t in seen; seen.add(t)
    R[f"ids_matched_to_2plus_s1_{k}"] = dup
(OUT / "v02.json").write_text(json.dumps(R, indent=1))
print(json.dumps(R, indent=1))
