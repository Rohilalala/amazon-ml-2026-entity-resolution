"""Independent replay of the TEST add decision from the saved head probabilities (own code, not h01_head.add_new):
thr = 0.80 if the S1 keeps >= 1 id in the best file else 0.75; drop pairs whose j is kept by any S1 in the best file;
per j keep the highest prob (ties: first in pair-list order). Compare with the adds found in the upload files (v01).
Also: head-prob rows aligned with the pair list; probabilities re-predicted with model/head.txt on a random S1 sample.
-> out/v02.json"""
import json, resource, sys, time
import numpy as np, polars as pl, lightgbm as lgb

R = "/Users/adityarohilla/Documents/amazon-ml-2026"; V8 = f"{R}/v8_lab"; O = f"{V8}/recall/verify/out"
T0 = time.time(); TSV = dict(separator="\t", quote_char=None, infer_schema=False)
meta = json.load(open(f"{V8}/recall/head/model/head.json")); TK, TN = meta["t_keep"], meta["t_none"]
HP = pl.read_parquet(f"{V8}/recall/head/out/test_head_prob.parquet")
PL = pl.read_parquet(f"{V8}/recall/retrieve/out/pairs/test_m1_r1_cap10.parquet")
aligned = bool((HP["s1_row"] == PL["s1_row"]).all() and (HP["j_row"] == PL["j_row"]).all())
recs = pl.read_parquet(f"{R}/cache/3ee42f05d06efbcb/recs.parquet", columns=["row", "entity_id"]).with_columns(pl.col("entity_id").str.strip_chars())
best = pl.read_csv(f"{V8}/ce/v2_build/A_all_v2/matching_results.tsv", **TSV).select(
    pl.col("source1_entity_id").alias("s1"), pl.col("matched_entity_ids").fill_null("").alias("m"))
kept = best.select(pl.col("m").str.split(",")).explode("m").filter(pl.col("m") != "")["m"]
has_keep = best.select("s1", has=pl.col("m") != "")
D = (HP.with_row_index("idx")
       .join(recs.rename({"row": "s1_row", "entity_id": "s1"}), on="s1_row", how="left")
       .join(recs.rename({"row": "j_row", "entity_id": "jid"}), on="j_row", how="left")
       .join(has_keep, on="s1", how="left"))
assert D["s1"].null_count() == 0 and D["jid"].null_count() == 0 and D["has"].null_count() == 0
D = D.with_columns(thr=pl.when("has").then(TK).otherwise(TN), taken=pl.col("jid").is_in(kept.implode()))
mine = (D.filter((pl.col("prob") >= pl.col("thr")) & ~pl.col("taken"))
          .sort(["jid", "prob", "idx"], descending=[False, True, False]).unique("jid", keep="first", maintain_order=True)
          .select("s1", "jid", "country"))
files = pl.read_csv(f"{O}/adds_ALL.tsv", **TSV)
key = lambda d: d.select(pl.concat_str("s1", "jid", separator="|"))[:, 0]
a, b = set(key(mine).to_list()), set(key(files).to_list())
out = {"pairs": HP.height, "rows_aligned_with_pair_list": aligned, "t_keep": TK, "t_none": TN,
       "replay_adds": len(a), "file_adds": len(b), "only_replay": len(a - b), "only_file": len(b - a),
       "replay_by_country": dict(mine.group_by("country").len().iter_rows()),
       "file_by_country": dict(files.group_by("country").len().iter_rows())}
out["decision_match"] = aligned and a == b and files.height == len(b)
print(json.dumps(out), flush=True)
# ---- probabilities vs the shipped model on a random S1 sample (features() is the head's own feature code)
sys.path.insert(0, f"{V8}/recall/head"); import h01_head as h
rng = np.random.default_rng(90127)
s1s = rng.choice(PL["s1_row"].unique().to_numpy(), 5000, replace=False)
C = pl.read_parquet(f"{V8}/recall/head/out/TEST_ce.parquet")
assert (C["i"] == PL["s1_row"]).all() and (C["j"] == PL["j_row"]).all()
S = PL.with_columns(ce_logit=C["ce_logit"], prob=HP["prob"]).filter(pl.col("s1_row").is_in(s1s))
Fx = h.features(S, f"{V8}/ce/setup/out/text_test.parquet")
pr = lgb.Booster(model_file=f"{V8}/recall/head/model/head.txt").predict(h.X(Fx))
out["prob_check"] = {"s1": 5000, "pairs": Fx.height, "max_abs_diff": float(np.abs(pr - Fx["prob"].to_numpy()).max())}
out["seconds"] = round(time.time() - T0); out["peak_rss_gb"] = round(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1e9, 2)
out["PASS"] = out["decision_match"] and out["prob_check"]["max_abs_diff"] < 1e-9
json.dump(out, open(f"{O}/v02.json", "w"), indent=1); print(json.dumps(out, indent=1))
