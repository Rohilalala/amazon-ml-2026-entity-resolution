"""Per-country comparison of the A_all_v2 upload file with build/A_all (the 0.985 file), from the two TSVs:
S1 rows whose match list changed, pairs added / dropped, kept pairs per S1, empty share (S1 with no match), house-variant
keeps per S1 (common.relation hrel edit1 / off20, as b01_stage4_test.stats). Cross-checks: A_all equals build/out/b01.json
stage4 stats, A_all_v2 equals v2_stacker's p04_ACNRST_3s_v2ce.json kept pairs and empty share.
usage (repo venv): b02_stats.py -> out/b02.json"""
import json, sys, time, resource
from pathlib import Path
import numpy as np, polars as pl
ROOT = Path("/Users/adityarohilla/Documents/amazon-ml-2026"); CE = ROOT / "v8_lab/ce"; D = Path(__file__).parent
sys.path.insert(0, str(CE / "stack"))
import s01_stage4 as s4
sys.path.insert(0, str(ROOT / "v8_lab/audit_a"))
from common import relation
import os
if not os.environ.get("B02_HEAVY"):  # run_b02.sh runs this under day1b/heavy.sh (peak ~4 GB)
    s4._rss_guard()
T0 = time.time()
FILES = {"A_all": CE / "build/A_all/matching_results.tsv", "A_all_v2": D / "A_all_v2/matching_results.tsv"}
B01 = json.load(open(CE / "build/out/b01.json")); P04 = json.load(open(CE / "v2_stacker/out_v2ce/p04_ACNRST_3s_v2ce.json"))
REC = ROOT / "cache/3ee42f05d06efbcb/recs.parquet"
rows = {k: pl.read_csv(p, separator="\t", quote_char=None, infer_schema=False).select(
            pl.col("source1_entity_id").alias("s1"), pl.col("matched_entity_ids").fill_null("").alias("m")) for k, p in FILES.items()}
assert rows["A_all"]["s1"].equals(rows["A_all_v2"]["s1"])
R = {}
for c in ("us", "india", "france"):
    rc = (pl.scan_parquet(REC).filter(pl.col("country_key") == c).select("row", pl.col("entity_id").str.strip_chars(), "src", "core", "addr", "house")
          .collect(engine="streaming"))
    s1 = rc.filter(pl.col("src") == 1).select(pl.col("entity_id").alias("s1"), pl.col("row").alias("i"))
    n = s1.height
    ids = rc.select(pl.col("entity_id").alias("m"), pl.col("row").alias("j"))
    pairs, r = {}, {"s1": n}
    for k, t in rows.items():
        t = t.join(s1, on="s1", how="semi")
        assert t.height == n
        pairs[k] = (t.select("s1", pl.col("m").str.split(",")).explode("m").filter(pl.col("m") != "")
                    .join(s1, on="s1", how="left").join(ids, on="m", how="left").select("i", "j"))
        assert pairs[k].null_count().sum_horizontal().item() == 0
        r[k] = {"kept_pairs": pairs[k].height, "kept_per_s1": round(pairs[k].height / n, 5),
                "empty_share": round(float((t["m"] == "").sum()) / n, 5)}
    a, b = rows["A_all"].join(s1, on="s1", how="semi"), rows["A_all_v2"].join(s1, on="s1", how="semi")
    ch = int((a["m"] != b["m"]).sum())
    r["rows_changed"] = ch; r["rows_changed_share"] = round(ch / n, 5)
    r["pairs_added"] = pairs["A_all_v2"].join(pairs["A_all"], on=["i", "j"], how="anti").height
    r["pairs_dropped"] = pairs["A_all"].join(pairs["A_all_v2"], on=["i", "j"], how="anti").height
    u = pl.concat([pairs["A_all"], pairs["A_all_v2"]]).unique()
    rel = relation(u, rc.select("row", "core", "addr", "house")).select("i", "j", hv=pl.col("hrel").is_in(["edit1", "off20"]))
    for k in FILES:
        hv = int(pairs[k].join(rel, on=["i", "j"], how="left")["hv"].sum())
        r[k].update(house_variant_keeps=hv, house_variant_per_s1=round(hv / n, 5))
    ref = B01[c]["stage4"]
    r["check_A_all_eq_b01"] = all(r["A_all"][f] == ref[f] for f in ("kept_pairs", "kept_per_s1", "empty_share", "house_variant_keeps"))
    p = P04[c]["ACNRST_3s_v2ce"]
    r["check_A_all_v2_eq_p04"] = r["A_all_v2"]["kept_pairs"] == p["kept_pairs"] and r["A_all_v2"]["empty_share"] == p["empty_share"]
    r["decision_info"] = p["info"]
    R[c] = r
    print(f"[{time.time() - T0:6.1f}s]", c, json.dumps(r), flush=True)
    del rc, s1, ids, pairs, u, rel, a, b
R["peak_rss_gb"] = round(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1e9, 2)
(D / "out/b02.json").write_text(json.dumps(R, indent=1))
print("done", R["peak_rss_gb"])
