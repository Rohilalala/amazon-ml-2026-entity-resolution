"""Audit 4 (test inputs): per country, test_pairs.prob == the stack source score file (US a22 p2, India a24 prob,
France g01 p2); ACTIVE (prob >= 0.002) counts; test_ce rows == ACTIVE rows; kept == the final stack submission rows
(build/r3_fr_tv_add). Spot-check: src.pipeline.decide (owner rule + expected-F0.5, missed 0, no CE) on the US and India
test pairs reproduces the stack rows, over all pairs and over ACTIVE pairs only.
Repo venv under heavy.sh. usage: v03_test.py -> out/v03_test.json"""
import json, sys, resource
from pathlib import Path
from types import SimpleNamespace
import numpy as np, polars as pl

ROOT = Path("/Users/adityarohilla/Documents/amazon-ml-2026")
REPO = ROOT / "code/business_entity_resolution"
sys.path.insert(0, str(REPO))
from src import pipeline
V8 = ROOT / "v8_lab"; SET = V8 / "ce/setup/out"; OUT = Path(__file__).parent / "out"
B = json.load(open(REPO / "artifacts/v7/bundle.json"))
R = {}

recs = pl.read_parquet(ROOT / "cache/3ee42f05d06efbcb/recs.parquet", columns=["row", "entity_id", "src", "country_key"])
assert (recs["row"].to_numpy() == np.arange(recs.height)).all()
T = pl.read_parquet(SET / "test_pairs.parquet")
R["pairs"] = T.height
R["pairs_eq_cache"] = T.height == pl.scan_parquet(ROOT / "cache/3ee42f05d06efbcb/pairs.parquet").select(pl.len()).collect().item()
R["active_flag_eq_recomputed"] = bool((T["active"] == (T["prob"] >= 0.002)).all())
cty = recs["country_key"].to_numpy()
R["country_col_eq_s1_country"] = bool((T["country"].cast(pl.String).to_numpy() == cty[T["i"].to_numpy()]).all())

# final stack submission -> (i, j)
m = pl.read_csv(V8 / "day1b/build/r3_fr_tv_add/matching_results.tsv", separator="\t", quote_char=None, infer_schema=False)
ids = recs.select(pl.col("entity_id").str.strip_chars(), "row")
stack = (m.select(pl.col("source1_entity_id").str.strip_chars().alias("s"), pl.col("matched_entity_ids").fill_null("").str.split(",").alias("m"))
         .explode("m").with_columns(pl.col("m").str.strip_chars()).filter(pl.col("m") != "")
         .join(ids.rename({"entity_id": "s", "row": "i"}), on="s", how="left").join(ids.rename({"entity_id": "m", "row": "j"}), on="m", how="left"))
assert stack["i"].null_count() == 0 and stack["j"].null_count() == 0
stack = stack.select(pl.col("i").cast(T["i"].dtype), pl.col("j").cast(T["j"].dtype)).with_columns(pl.lit(True).alias("st"))
R["stack_rows"] = {"s1_rows": m.height, "pairs": stack.height}
del m, ids
T = T.join(stack, on=["i", "j"], how="left", maintain_order="left").with_columns(pl.col("st").fill_null(False))
R["stack_pairs_not_in_candidates"] = stack.height - int(T["st"].sum())
del stack
ce = pl.read_parquet(V8 / "ce/infer/out/test_ce.parquet", columns=["i", "j"])
A = T.filter("active")
R["test_ce_rows_eq_active_rows"] = bool(ce.height == A.height and (ce["i"] == A["i"]).all() and (ce["j"] == A["j"]).all())
del ce, A

src = {"us": (V8 / "day1b/a_us/out/a22_scores_us_ldftrain.parquet", "p2"),
       "india": (V8 / "day1b/a_in/out/a24_scores_india_ldftrain.parquet", "prob"),
       "france": (V8 / "audit_a/out/g01_scores_france_all_stage2.parquet", "p2")}
s1c = recs.filter(pl.col("src") == 1)["country_key"].to_numpy()
for c, (f, col) in src.items():
    t = T.filter(pl.col("country") == c)
    sc = pl.read_parquet(f, columns=["i", "j", col])
    x = t.select("i", "j", "prob").join(sc, on=["i", "j"], how="full", coalesce=True)
    r = {"pairs": t.height, "source_pairs": sc.height, "unmatched": int(x["prob"].null_count() + x[col].null_count()),
         "prob_eq_source": bool((x["prob"] == x[col]).all()), "active": int(t["active"].sum()),
         "kept_pairs": int(t["kept"].sum()), "kept_eq_final_stack": int((t["kept"] != t["st"]).sum()),
         "stack_kept_outside_active": int((t["st"] & ~t["active"]).sum())}
    del sc, x
    if c in ("us", "india"):
        ns = SimpleNamespace(config=SimpleNamespace(exclusive=True, unseen_none_cap=B["config"]["unseen_none_cap"]), rule=B["rule"],
                             report={"oof_f05_by_country": B["report"]["oof_f05_by_country"]})
        i, j, p, st = t["i"].to_numpy(), t["j"].to_numpy(), t["prob"].to_numpy(), t["st"].to_numpy()
        keep, info = pipeline.decide(i, j, p, cty[i], s1c[s1c == c], ns)
        assert info == {}
        r["decide_all_pairs_diff_vs_stack"] = int((keep != st).sum())
        a = t["active"].to_numpy()
        keep_a, _ = pipeline.decide(i[a], j[a], p[a], cty[i[a]], s1c[s1c == c], ns)
        full_a = np.zeros(len(a), bool); full_a[np.flatnonzero(a)] = keep_a
        r["decide_active_only_diff_vs_stack"] = int((full_a != st).sum())
        r["decide_active_only_diff_detail"] = {"added": int((full_a & ~st).sum()), "dropped": int((~full_a & st).sum())}
        del i, j, p, st, keep, keep_a, full_a, a
    R[c] = r
    print(c, json.dumps(r), flush=True)
    del t
R["peak_rss_gb"] = round(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1e9, 2)
(OUT / "v03_test.json").write_text(json.dumps(R, indent=1))
print(json.dumps(R, indent=1))
