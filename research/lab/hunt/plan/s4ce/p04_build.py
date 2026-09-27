"""TEST decision for a stage-4 v2 variant, per country, as build/b01_stage4_test.py decides stage 4 (set A):
src.pipeline.decide of the integration worktree (exclusive owner rule over the country's pairs, non-active pairs keep
the stack prob, then the variant's rule); France is unseen: pinned empty-share target from the worktree's v8 bundle,
then src.unseen_rules.correct. Per country with that country's records only, so peak RSS stays < 3 GB (b01 needs ~9 GB).
Check: the same path on build/out/test_stage4_prob.parquet (float32) with the set-A rule vs build/out/keep_{c}.parquet.
usage (repo venv): p04_build.py NAME -> out/build_NAME/keep_{us,india,france}.parquet, out/p04_NAME.json"""
import json, sys, time, resource
from pathlib import Path
from types import SimpleNamespace
import numpy as np, polars as pl

ROOT = Path("/Users/adityarohilla/Documents/amazon-ml-2026")
WT = Path("/Users/adityarohilla/Documents/amazon-ml-2026-integrate/code/business_entity_resolution")
HERE = Path(__file__).parent; OUT = HERE / "out"; CE = ROOT / "v8_lab/ce"; NAME = sys.argv[1]
import os
OUT = HERE / f"out_{os.environ['CE_TAG']}"  # hunt copy
sys.path.insert(0, str(WT))
from src import pipeline, unseen_rules  # worktree modules, as b01
sys.path.insert(0, str(CE / "stack"))
import s01_stage4 as s4
assert pipeline.__file__.startswith(str(WT)) and unseen_rules.__file__.startswith(str(WT))
import os
if not os.environ.get("V2_HEAVY"):  # set by run_build.sh, which runs this under day1b/heavy.sh
    s4._rss_guard()
T0 = time.time(); BO = OUT / f"build_{NAME}"; BO.mkdir(exist_ok=True)
BUNDLE = json.load(open(WT / "artifacts/v8/bundle.json"))
RULE = json.load(open(HERE / f"model/{NAME}.json"))["rule"]; RULE_A = json.load(open(CE / "stack/model/stage4_A.json"))["rule"]
R = {"name": NAME, "rule": RULE, "rule_stage4_A": RULE_A, "unseen_empty_target": BUNDLE["config"]["unseen_empty_target"]}


def ns(rule):
    return SimpleNamespace(config=SimpleNamespace(exclusive=True, unseen_none_cap=BUNDLE["config"]["unseen_none_cap"],
                                                  unseen_empty_target=BUNDLE["config"]["unseen_empty_target"]),
                           rule=rule, report={"oof_f05_by_country": BUNDLE["report"]["oof_f05_by_country"]})


TA = pl.read_parquet(s4.SET / "test_pairs.parquet", columns=["country", "active"])
CODE = {"us": 0, "india": 1, "france": 2}
ctry_act = TA["country"].filter(TA["active"]).cast(pl.String).replace_strict(CODE, return_dtype=pl.Int8).to_numpy(); del TA
p4 = {}  # active pairs only, in test_pairs order
for tag, path in ((NAME, OUT / f"test_{NAME}_prob.parquet"), ("stage4_A", CE / "build/out/test_stage4_prob.parquet")):
    p4[tag] = pl.read_parquet(path)
    assert p4[tag].height == len(ctry_act)
for c in ("us", "india", "france"):
    T = (pl.scan_parquet(s4.SET / "test_pairs.parquet").select("i", "j", "prob", "active", "country")
         .filter(pl.col("country") == c).collect(engine="streaming"))
    act = T["active"].to_numpy()
    probs = {}
    for tag, q in p4.items():  # stage-4 prob of the country's active pairs; the stack prob elsewhere
        q = q.filter(pl.Series(ctry_act == CODE[c]))
        assert (q["i"] == T["i"].filter(act)).all() and (q["j"] == T["j"].filter(act)).all()
        probs[tag] = q["p4"].to_numpy()
    base = T["prob"].to_numpy()
    # the country's records, rows remapped (decide and unseen_rules.correct only index records by i, j)
    rc = (pl.scan_parquet(ROOT / "cache/3ee42f05d06efbcb/recs.parquet").filter(pl.col("country_key") == c)
          .select("row", "src", *(unseen_rules.COLUMNS if c == "france" else ())).collect(engine="streaming").sort("row"))
    rows = rc["row"].to_numpy()
    i, j = np.searchsorted(rows, T["i"].to_numpy()).astype(np.uint32), np.searchsorted(rows, T["j"].to_numpy()).astype(np.uint32)
    assert (rows[i] == T["i"].to_numpy()).all() and (rows[j] == T["j"].to_numpy()).all()
    # one-country label arrays as zero-copy broadcasts (np.full of strings would be ~0.5 GB for India)
    n = int((rc["src"] == 1).sum()); pc, sc = np.broadcast_to(np.array(c), len(i)), np.broadcast_to(np.array(c), n)
    del T
    r, keeps = {"s1": n, "pairs": len(i), "active": int(act.sum())}, {}
    for tag, rule in ((NAME, RULE), ("stage4_A", RULE_A)):
        p = base.astype(np.float64); p[act] = probs.pop(tag)
        k, info = pipeline.decide(i, j, p, pc, sc, ns(rule))
        if c == "france":
            k, cnt = unseen_rules.correct(rc, i, j, p, k); info[c].update(cnt)
        else:
            assert info == {}
        del p
        keeps[tag] = k
        r[tag] = {"kept_pairs": int(k.sum()), "kept_per_s1": round(float(k.sum() / n), 5),
                  "empty_share": round(1 - len(np.unique(i[k])) / n, 5), "info": info.get(c)}
    del rc  # ids only after the decision (lower peak)
    eid = (pl.scan_parquet(ROOT / "cache/3ee42f05d06efbcb/recs.parquet").filter(pl.col("country_key") == c)
           .select("row", pl.col("entity_id").str.strip_chars()).collect(engine="streaming").sort("row"))
    assert (eid["row"].to_numpy() == rows).all()
    eid = eid["entity_id"]
    ref = pl.read_parquet(CE / f"build/out/keep_{c}.parquet")
    mine_a = pl.DataFrame({"s1": eid.gather(i[keeps["stage4_A"]]), "m": eid.gather(j[keeps["stage4_A"]])})
    r["repro_A_all_keep"] = {"rows": ref.height, "mine": mine_a.height,
                             "diff": mine_a.join(ref, on=["s1", "m"], how="anti").height + ref.join(mine_a, on=["s1", "m"], how="anti").height}
    # b01 decided on the float64 prediction, test_stage4_prob.parquet stores it as float32: a few pairs may flip
    k4, ka = keeps[NAME], keeps["stage4_A"]; flip = k4 != ka
    r["vs_A_all"] = {"added": int((k4 & ~ka).sum()), "dropped": int((~k4 & ka).sum()),
                     "s1_rows_changed": int(len(np.unique(i[flip]))), "s1_rows_changed_share": round(len(np.unique(i[flip])) / n, 5)}
    pl.DataFrame({"s1": eid.gather(i[k4]), "m": eid.gather(j[k4])}).write_parquet(BO / f"keep_{c}.parquet")
    R[c] = r
    print(f"[{time.time() - T0:6.1f}s]", c, json.dumps(r), flush=True)
    del eid, probs, base, i, j, keeps, k4, ka, flip, mine_a, ref, act, rows, pc, sc
R["peak_rss_gb"] = round(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1e9, 2); R["seconds"] = round(time.time() - T0)
(OUT / f"p04_{NAME}.json").write_text(json.dumps(R, indent=1))
print("done", R["peak_rss_gb"])
