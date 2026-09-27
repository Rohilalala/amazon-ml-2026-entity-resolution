"""France-only decision variants for LB probes (empty-share targets): p06 = p05 with the pinned unseen empty-share TARGET varied
(delta 0, none_scale cap lifted). Target 0.05597 must equal p05 delta 0 (if its cap did not bind). Original p05 doc:
the variant's stage-4 France probabilities shifted by a logit delta before the
unseen-country decision (owner rule + rule + empty-share calibration to the pinned target), then src.unseen_rules.correct.
delta 0 with the bundle's none_scale cap must reproduce out_<tag>/build_NAME/keep_france.parquet exactly.
For delta != 0 the none_scale cap is lifted (CAP) so the calibration can hold the pinned empty share.
usage (repo venv, CE_TAG set): p05_fr_shift.py NAME CAP delta [delta ...] -> out_<tag>/fr_shift/keep_france_d<delta>.parquet, fr_shift.json"""
import json, os, sys, time
from pathlib import Path
from types import SimpleNamespace
import numpy as np, polars as pl

ROOT = Path("/Users/adityarohilla/Documents/amazon-ml-2026")
WT = Path("/Users/adityarohilla/Documents/amazon-ml-2026-integrate/code/business_entity_resolution")
HERE = Path(__file__).parent; CE = ROOT / "v8_lab/ce"; NAME, CAP = sys.argv[1], float(sys.argv[2]); TARGETS = [float(t) for t in sys.argv[3:]]
OUT = HERE / f"out_{os.environ['CE_TAG']}"; FO = OUT / "fr_empty"; FO.mkdir(exist_ok=True)
sys.path.insert(0, str(WT))
from src import pipeline, unseen_rules
sys.path.insert(0, str(CE / "stack"))
import s01_stage4 as s4
T0 = time.time()
BUNDLE = json.load(open(WT / "artifacts/v8/bundle.json")); RULE = json.load(open(HERE / f"model/{NAME}.json"))["rule"]
def ns(cap, target):
    return SimpleNamespace(config=SimpleNamespace(exclusive=True, unseen_none_cap=cap, unseen_empty_target=target),
                           rule=RULE, report={"oof_f05_by_country": BUNDLE["report"]["oof_f05_by_country"]})
TA = pl.read_parquet(s4.SET / "test_pairs.parquet", columns=["country", "active"])
fr_act = (TA["country"].filter(TA["active"]).cast(pl.String) == "france").to_numpy(); del TA
q = pl.read_parquet(OUT / f"test_{NAME}_prob.parquet").filter(pl.Series(fr_act))
T = (pl.scan_parquet(s4.SET / "test_pairs.parquet").select("i", "j", "prob", "active", "country")
     .filter(pl.col("country") == "france").collect(engine="streaming"))
act = T["active"].to_numpy()
assert (q["i"] == T["i"].filter(act)).all() and (q["j"] == T["j"].filter(act)).all()
p4 = q["p4"].to_numpy().astype(np.float64); base = T["prob"].to_numpy().astype(np.float64)
rc = (pl.scan_parquet(ROOT / "cache/3ee42f05d06efbcb/recs.parquet").filter(pl.col("country_key") == "france")
      .select("row", "src", "entity_id", *unseen_rules.COLUMNS).collect(engine="streaming").sort("row"))
rows = rc["row"].to_numpy()
i = np.searchsorted(rows, T["i"].to_numpy()).astype(np.uint32); j = np.searchsorted(rows, T["j"].to_numpy()).astype(np.uint32)
assert (rows[i] == T["i"].to_numpy()).all() and (rows[j] == T["j"].to_numpy()).all()
n = int((rc["src"] == 1).sum()); pc, sc = np.broadcast_to(np.array("france"), len(i)), np.broadcast_to(np.array("france"), n)
eid = rc["entity_id"].str.strip_chars(); del T
ref = pl.read_parquet(OUT / f"build_{NAME}/keep_france.parquet")
R = {"name": NAME, "cap_for_shifted": CAP, "n_s1": n}
lg = lambda x: np.log(np.clip(x, 1e-9, 1 - 1e-9) / (1 - np.clip(x, 1e-9, 1 - 1e-9)))
for d in TARGETS:
    p = base.copy(); p[act] = p4
    k, info = pipeline.decide(i, j, p, pc, sc, ns(CAP, d))
    k, cnt = unseen_rules.correct(rc, i, j, p, k); info["france"].update(cnt)
    mine = pl.DataFrame({"s1": eid.gather(i[k]), "m": eid.gather(j[k])})
    r = {"kept_pairs": int(k.sum()), "kept_per_s1": round(float(k.sum() / n), 5), "empty_share": round(1 - len(np.unique(i[k])) / n, 5),
         "info": info["france"], "vs_delta0_ref": {"added": mine.join(ref, on=["s1", "m"], how="anti").height,
                                                   "dropped": ref.join(mine, on=["s1", "m"], how="anti").height}}
    mine.write_parquet(FO / f"keep_france_t{d:g}.parquet"); R[f"t{d:g}"] = r
    print(f"[{time.time() - T0:6.1f}s] target {d:g}", json.dumps(r, default=str), flush=True)
(FO / "fr_empty.json").write_text(json.dumps(R, indent=1, default=str)); print("done")
