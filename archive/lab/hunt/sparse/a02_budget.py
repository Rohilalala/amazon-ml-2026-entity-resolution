"""Budget curves on the HOLDOUT: every sparse new pair (forward rank 11..30 or reverse rank 4..10, not cached) ranked by
a priority; recovered missed trues at 1/2/3/5 extra pairs per holdout S1, alone and beyond the dense new pairs.
-> out/a02.json"""
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
truth = pl.read_parquet(ROOT / "v8_lab/ce/setup/out/truth_train.parquet").filter(pl.col("i").is_in(H)).with_columns(pl.lit(1, pl.Int8).alias("y"))
cand = pl.scan_parquet(CACHE / "pairs.parquet").select("i", "j").filter(pl.col("i").is_in(H)).collect()
n_missed = truth.join(cand, on=["i", "j"], how="anti").height
parts, have = [], []
for c in ("us", "india"):
    for s in (2, 3):
        t = f"train_{c}_s{s}"
        if not (O / f"eng_{t}.json").exists():
            continue
        have.append(t)
        fw = pl.read_parquet(O / f"fwd_{t}.parquet")
        rv = pl.read_parquet(O / f"rev_{t}.parquet")
        f10 = fw.filter(pl.col("rank") == 10).select("i", pl.col("cos").alias("f10"))
        r3 = rv.filter(pl.col("rev_rank") == 3).select("j", pl.col("cos").alias("r3"))
        a = fw.select("i", "j", "cos", "rank")
        b = rv.filter(pl.col("i").is_in(H)).select("i", "j", "cos", "rev_rank")
        x = a.join(b, on=["i", "j"], how="full", coalesce=True).with_columns(pl.coalesce("cos", "cos_right").alias("cos")).drop("cos_right")
        x = x.join(f10, on="i", how="left").join(r3, on="j", how="left").with_columns(pl.lit(s, pl.Int8).alias("jsrc"))
        parts.append(x)
x = pl.concat(parts).join(cand, on=["i", "j"], how="anti").join(truth, on=["i", "j"], how="left").with_columns(pl.col("y").fill_null(0))
dense = pl.read_parquet(RET / "holdout_m1_r1_cap10.parquet", columns=["s1_row", "j_row"]).rename({"s1_row": "i", "j_row": "j"})
x = x.join(dense.with_columns(pl.lit(True).alias("in_dense")), on=["i", "j"], how="left").with_columns(pl.col("in_dense").fill_null(False))
x = x.with_columns((pl.col("cos") / pl.col("f10")).fill_null(0).alias("rf"), (pl.col("cos") / pl.col("r3")).fill_null(0).alias("rr"))
x = x.with_columns(pl.max_horizontal("rf", "rr").alias("ratio"),
                   pl.when(pl.col("rank").is_not_null()).then(pl.col("rank")).otherwise(30).alias("rk"),
                   pl.when(pl.col("rev_rank").is_not_null()).then(pl.col("rev_rank") * 3).otherwise(90).alias("rrk"))
x = x.with_columns(pl.min_horizontal("rk", "rrk").alias("mixrank"))
ea = pl.scan_parquet(CACHE / "recs.parquet").select(pl.col("row").alias("j"), (pl.col("addr").fill_null("").str.strip_chars() == "").alias("j_empty")) \
    .filter(pl.col("j").is_in(x["j"].unique().implode())).collect()
x = x.join(ea, on="j", how="left")
R = {"frames": have, "n_missed_all": n_missed, "missed_in_frames": int(x["y"].sum()) + 0, "pool_pairs_per_s1": round(x.height / 1e5, 3),
     "pool_trues": int(x["y"].sum()), "pool_trues_not_dense": int(x.filter(~pl.col("in_dense"))["y"].sum()),
     "pool_trues_empty_addr": int(x.filter(pl.col("j_empty"))["y"].sum())}
keeps = pl.read_parquet(ROOT / "v8_lab/ce/v2_build/out/holdout_keeps.parquet", columns=["i", "j", "ACNRST_3s_v2ce"])
kt = keeps.join(truth.select("i", "j", pl.lit(True).alias("t")), on=["i", "j"], how="left").with_columns(pl.col("t").fill_null(False))
agg = hold.select("i", "n_true").join(kt.group_by("i").agg(pl.col("ACNRST_3s_v2ce").sum().alias("np"),
                                                           (pl.col("ACNRST_3s_v2ce") & pl.col("t")).sum().alias("tp")), on="i", how="left").fill_null(0)


def f05(n_pred, n_true, tp):
    n_pred, n_true, tp = (np.asarray(x, np.float64) for x in (n_pred, n_true, tp))
    with np.errstate(divide="ignore", invalid="ignore"):
        p, r = tp / n_pred, tp / n_true
        f = np.where(tp > 0, 1.25 * p * r / (0.25 * p + r), 0.0)
    return np.where(n_true == 0, (n_pred == 0).astype(np.float64), f)


base = f05(agg["np"], agg["n_true"], agg["tp"]).mean()


def ceiling(sel):
    a2 = agg.join(sel.filter(pl.col("y") == 1).group_by("i").len("r"), on="i", how="left").fill_null(0)
    return round(float(f05(a2["np"] + a2["r"], a2["n_true"], a2["tp"] + a2["r"]).mean() - base), 6)


prio = {"ratio": pl.col("ratio"), "cos": pl.col("cos"), "neg_mixrank": -pl.col("mixrank").cast(pl.Float64) + pl.col("ratio") * 1e-3,
        "ratio_nonempty_first": pl.col("ratio") - pl.col("j_empty").cast(pl.Float64) * 10}
for name, e in prio.items():
    xs = x.sort(e, descending=True)
    y = xs["y"].to_numpy()
    ye = (xs["y"].to_numpy() == 1) & xs["j_empty"].to_numpy()
    yn = x.filter(~pl.col("in_dense")).sort(e, descending=True)["y"].to_numpy()
    cs, csn = np.cumsum(y), np.cumsum(yn)
    R[name] = {f"b{b}": {"recovered": int(cs[min(len(cs), int(b * 1e5)) - 1]), "share": round(cs[min(len(cs), int(b * 1e5)) - 1] / n_missed, 4),
                         "beyond_dense_recovered": int(csn[min(len(csn), int(b * 1e5)) - 1]),
                         "recovered_empty_addr": int(ye[:int(b * 1e5)].sum()),
                         "ceiling_gain": ceiling(xs.head(int(b * 1e5))),
                         "ceiling_gain_beyond_dense": ceiling(xs.head(int(b * 1e5)).filter(~pl.col("in_dense")))} for b in (0.5, 1, 2, 3, 5, 10)}
    print(name, R[name], flush=True)
json.dump(R, open(O / "a02.json", "w"), indent=1)
print(json.dumps({k: v for k, v in R.items() if not isinstance(v, dict)}))
