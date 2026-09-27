"""Step 1: where the HOLDOUT's blocking-missed true pairs sit in the sparse key space, and what widening buys.
Inputs: out/{fwd,rev,pc}_train_{us,india}_s{2,3}.parquet from eng.py. Candidate set = train cache pairs (k10/kr3).
-> out/a01.json, out/a01_missed.parquet, printed examples."""
import json, sys
from pathlib import Path
import numpy as np, polars as pl
sys.path.insert(0, str(Path(__file__).parent))
import wd  # noqa: E402
wd.start()

ROOT = Path("/Users/adityarohilla/Documents/amazon-ml-2026")
CACHE = ROOT / "cache/58f0a60c5c264fdd"
O = Path(__file__).parent / "out"
RET = ROOT / "v8_lab/recall/retrieve/out/pairs"
hold = pl.read_parquet(ROOT / "v8_lab/ce/setup/out/holdout_s1.parquet")
H = hold["i"].implode()
truth = pl.read_parquet(ROOT / "v8_lab/ce/setup/out/truth_train.parquet").filter(pl.col("i").is_in(H))
cand = pl.scan_parquet(CACHE / "pairs.parquet").select("i", "j", "score", "jsrc").filter(pl.col("i").is_in(H)).collect()
recs = pl.read_parquet(CACHE / "recs.parquet", columns=["row", "src", "name", "addr", "core"])
missed = truth.join(cand, on=["i", "j"], how="anti").join(recs.select(pl.col("row").alias("j"), pl.col("src").alias("jsrc")), on="j")
missed = missed.join(hold.select("i", "country"), on="i")
R = {"n_true": truth.height, "n_missed": missed.height}
fw, rv, pc, have = [], [], [], []
for c in ("us", "india"):
    for s in (2, 3):
        t = f"train_{c}_s{s}"
        if not (O / f"eng_{t}.json").exists():
            continue
        have.append((c, s))
        fw.append(pl.read_parquet(O / f"fwd_{t}.parquet").with_columns(pl.lit(s, pl.Int8).alias("jsrc")))
        rv.append(pl.read_parquet(O / f"rev_{t}.parquet").with_columns(pl.lit(s, pl.Int8).alias("jsrc")))
        pc.append(pl.read_parquet(O / f"pc_{t}.parquet"))
        R[f"eng_{t}"] = json.load(open(O / f"eng_{t}.json"))
fw, rv, pc = pl.concat(fw), pl.concat(rv), pl.concat(pc, how="diagonal")
R["frames"] = have
if len(have) < 4:  # partial run: restrict to the frames searched
    fr = pl.DataFrame({"country": [c for c, _ in have], "jsrc": [s for _, s in have]}, schema={"country": pl.String, "jsrc": pl.Int8})
    missed = missed.join(fr, on=["country", "jsrc"], how="semi")
    cand = cand.join(hold.select("i", "country"), on="i").join(fr, on=["country", "jsrc"], how="semi").drop("country")
    R["n_missed_frames"] = missed.height
rvh = rv.filter(pl.col("i").is_in(H))

# reproduction: my fwd top-10 U rev top-3 (holdout S1) vs the cache
mine = pl.concat([fw.filter(pl.col("rank") <= 10).select("i", "j"), rvh.filter(pl.col("rev_rank") <= 3).select("i", "j")]).unique()
R["repro"] = {"cache_pairs": cand.height, "mine_k10r3": mine.height,
              "cache_not_mine": cand.join(mine, on=["i", "j"], how="anti").height,
              "mine_not_cache": mine.join(cand, on=["i", "j"], how="anti").height,
              "pc_cached_max_abs_diff": float((pc.drop_nulls("score")["cos"] - pc.drop_nulls("score")["score"]).abs().max())}

# per missed true: key cosine, forward rank (<=30), reverse rank (<=10), 10th-best forward of i, 3rd-best reverse of j
kth = fw.filter(pl.col("rank").is_in([10, 20, 30])).pivot(on="rank", index=["i", "jsrc"], values="cos")
kth = kth.rename({"10": "f10", "20": "f20", "30": "f30"})
rk = rv.filter(pl.col("rev_rank").is_in([3, 6, 10])).pivot(on="rev_rank", index="j", values="cos").rename({"3": "r3", "6": "r6", "10": "r10"})
m = (missed.join(pc.select("i", "j", "cos"), on=["i", "j"], how="left")
     .join(fw.select("i", "j", "rank"), on=["i", "j"], how="left")
     .join(rvh.select("i", "j", "rev_rank"), on=["i", "j"], how="left")
     .join(kth, on=["i", "jsrc"], how="left").join(rk, on="j", how="left"))
assert m["cos"].null_count() == 0, m["cos"].null_count()
m = m.with_columns(
    pl.when(pl.col("cos") <= 0).then(pl.lit("zero_overlap"))
    .when(pl.col("rank") <= 20).then(pl.lit("fwd_11_20"))
    .when(pl.col("rank") <= 30).then(pl.lit("fwd_21_30"))
    .when(pl.col("rev_rank") <= 6).then(pl.lit("rev_4_6"))
    .when(pl.col("rev_rank") <= 10).then(pl.lit("rev_7_10"))
    .otherwise(pl.lit("beyond")).alias("kind"),
    (pl.col("cos") / pl.col("f10")).alias("ratio_f10"), (pl.col("cos") / pl.col("r3")).alias("ratio_r3"))
m.write_parquet(O / "a01_missed.parquet")
R["kinds"] = {c: dict(zip(*m.filter(pl.col("country") == c).group_by("kind").len().sort("kind").to_dict(as_series=False).values()))
              for c in ("us", "india")}
R["enter"] = {f"k{k}": int((m["rank"] <= k).sum()) for k in (15, 20, 30)} | \
             {f"krev{k}": int((m["rev_rank"] <= k).sum()) for k in (4, 6, 10)} | \
             {"k20_or_krev6": int(((m["rank"] <= 20) | (m["rev_rank"] <= 6)).sum()),
              "k30_or_krev10": int(((m["rank"] <= 30) | (m["rev_rank"] <= 10)).sum())}
R["ratio_quantiles"] = m.filter(pl.col("cos") > 0).group_by("kind").agg(
    pl.len(), *[pl.col("ratio_f10").quantile(q).round(3).alias(f"cos_over_f10_q{int(q*100)}") for q in (0.1, 0.5, 0.9)],
    *[pl.col("ratio_r3").quantile(q).round(3).alias(f"cos_over_r3_q{int(q*100)}") for q in (0.1, 0.5, 0.9)]).sort("kind").to_dicts()
ea = recs.select(pl.col("row").alias("j"), (pl.col("addr").fill_null("").str.strip_chars() == "").alias("j_empty"))
m = m.join(ea, on="j", how="left")
R["kind_x_empty_addr"] = m.group_by("kind").agg(pl.len(), pl.col("j_empty").sum()).sort("kind").to_dicts()
print(json.dumps(R, indent=1), flush=True)

# widening rules on the holdout: new pairs (not cached) per S1, recovered missed trues, overlap with dense new pairs
keeps = pl.read_parquet(ROOT / "v8_lab/ce/v2_build/out/holdout_keeps.parquet", columns=["i", "j", "ACNRST_3s_v2ce"])
kt = keeps.join(truth.with_columns(pl.lit(True).alias("t")), on=["i", "j"], how="left").with_columns(pl.col("t").fill_null(False))
agg = hold.select("i", "n_true").join(kt.group_by("i").agg(pl.col("ACNRST_3s_v2ce").sum().alias("np"),
                                                           (pl.col("ACNRST_3s_v2ce") & pl.col("t")).sum().alias("tp")), on="i", how="left").fill_null(0)


def f05(n_pred, n_true, tp):
    n_pred, n_true, tp = (np.asarray(x, np.float64) for x in (n_pred, n_true, tp))
    with np.errstate(divide="ignore", invalid="ignore"):
        p, r = tp / n_pred, tp / n_true
        f = np.where(tp > 0, 1.25 * p * r / (0.25 * p + r), 0.0)
    return np.where(n_true == 0, (n_pred == 0).astype(np.float64), f)


base = f05(agg["np"], agg["n_true"], agg["tp"]).mean()
dense = {k: pl.read_parquet(RET / f"holdout_{k}.parquet", columns=["s1_row", "j_row"]).rename({"s1_row": "i", "j_row": "j"})
         for k in ("m1_r1_cap10", "m2_r1_cap10", "wide_k10_r3")}
tset = truth.with_columns(pl.lit(1, pl.Int8).alias("y"))
F = fw.join(kth.select("i", "jsrc", "f10"), on=["i", "jsrc"], how="left")
B = rvh.join(rk.select("j", "r3"), on="j", how="left")
rules = {}
for k in (10, 12, 15, 20, 30):
    for a in (0.0, 0.8, 0.85, 0.9, 0.95, 1.0):
        rules[f"fwd_k{k}_a{a}"] = F.filter((pl.col("rank") <= k) & (pl.col("cos") >= a * pl.col("f10") - 1e-6))
for r in (3, 4, 6, 10):
    for a in (0.0, 0.8, 0.9, 0.95, 1.0):
        rules[f"rev_r{r}_a{a}"] = B.filter((pl.col("rev_rank") <= r) & (pl.col("cos") >= a * pl.col("r3") - 1e-6))
W = {}
for name, x in rules.items():
    x = x.select("i", "j").unique().join(cand, on=["i", "j"], how="anti").join(tset, on=["i", "j"], how="left").with_columns(pl.col("y").fill_null(0))
    rec = x.filter(pl.col("y") == 1)
    a2 = agg.join(rec.group_by("i").len("r"), on="i", how="left").fill_null(0)
    W[name] = {"new_per_s1": round(x.height / 1e5, 3), "recovered": rec.height, "share_of_missed": round(rec.height / missed.height, 4),
               "ceiling_gain": round(float(f05(a2["np"] + a2["r"], a2["n_true"], a2["tp"] + a2["r"]).mean() - base), 6)} | \
              {f"not_in_dense_{k}": rec.join(d, on=["i", "j"], how="anti").height for k, d in dense.items()}
R["rules"] = W
for n, w in W.items():
    print(n, w, flush=True)
dm = m.join(dense["m2_r1_cap10"].with_columns(pl.lit(True).alias("dense_m2")), on=["i", "j"], how="left") \
      .join(dense["wide_k10_r3"].with_columns(pl.lit(True).alias("dense_wide")), on=["i", "j"], how="left")
R["kind_x_dense"] = dm.group_by("kind").agg(pl.len(), pl.col("dense_m2").fill_null(False).sum(), pl.col("dense_wide").fill_null(False).sum()) \
    .sort("kind").to_dicts()
json.dump(R, open(O / "a01.json", "w"), indent=1)
print(json.dumps(R["kind_x_dense"]))

# examples
txt = recs.select("row", "name", "addr")
pl.Config.set_tbl_rows(40); pl.Config.set_fmt_str_lengths(60); pl.Config.set_tbl_width_chars(250)
for kind in ("fwd_11_20", "fwd_21_30", "rev_4_6", "rev_7_10", "beyond", "zero_overlap"):
    e = dm.filter(pl.col("kind") == kind).sample(min(30, dm.filter(pl.col("kind") == kind).height), seed=1)
    e = e.join(txt.select(pl.col("row").alias("i"), pl.col("name").alias("n1"), pl.col("addr").alias("a1")), on="i") \
         .join(txt.select(pl.col("row").alias("j"), pl.col("name").alias("n2"), pl.col("addr").alias("a2")), on="j")
    print(f"\n==== {kind}")
    print(e.select("country", "cos", "rank", "rev_rank", "f10", "r3", "dense_m2", "n1", "n2", "a1", "a2"))
