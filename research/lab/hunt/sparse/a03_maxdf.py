"""HOLDOUT: blocking at k10/kr3 with a higher max_df (keys with df up to MAXDF kept, IDF-weighted) -> new pairs vs the
cache, recovered missed trues, beyond dense, empty-address share, perfect-judge ceiling on the best file.
usage: a03_maxdf.py MAXDF [K R]  -> out/a03_md{MAXDF}_k{K}r{R}.json"""
import json, sys
from pathlib import Path
import numpy as np, polars as pl
sys.path.insert(0, str(Path(__file__).parent))
import wd  # noqa: E402
wd.start()

ROOT = Path("/Users/adityarohilla/Documents/amazon-ml-2026")
CACHE = ROOT / "cache/58f0a60c5c264fdd"
O = Path(__file__).parent / "out"
MD = int(sys.argv[1])
K, RR = (int(sys.argv[2]), int(sys.argv[3])) if len(sys.argv) > 3 else (10, 3)
hold = pl.read_parquet(ROOT / "v8_lab/ce/setup/out/holdout_s1.parquet")
H = hold["i"].implode()
truth = pl.read_parquet(ROOT / "v8_lab/ce/setup/out/truth_train.parquet").filter(pl.col("i").is_in(H)).with_columns(pl.lit(1, pl.Int8).alias("y"))
cand = pl.scan_parquet(CACHE / "pairs.parquet").select("i", "j").filter(pl.col("i").is_in(H)).collect()
missed = truth.join(cand, on=["i", "j"], how="anti")
dense = pl.read_parquet(ROOT / "v8_lab/recall/retrieve/out/pairs/holdout_m1_r1_cap10.parquet", columns=["s1_row", "j_row"]).rename({"s1_row": "i", "j_row": "j"})
R, parts, own = {"maxdf": MD, "k": K, "k_rev": RR, "n_missed": missed.height}, [], []
for c in ("us", "india"):
    for s in (2, 3):
        t = f"train_{c}_s{s}"
        if not (O / f"eng_{t}_md{MD}.json").exists():
            continue
        R.setdefault("frames", []).append(t)
        for tg, acc in ((f"{t}_md{MD}", parts), (t, own)):
            fw = pl.read_parquet(O / f"fwd_{tg}.parquet").filter(pl.col("rank") <= K).select("i", "j", "cos")
            rv = pl.read_parquet(O / f"rev_{tg}.parquet").filter(pl.col("i").is_in(H) & (pl.col("rev_rank") <= RR)).select("i", "j", "cos")
            acc.append(pl.concat([fw, rv]).group_by("i", "j").agg(pl.col("cos").max()))
x = pl.concat(parts).join(cand, on=["i", "j"], how="anti").join(truth, on=["i", "j"], how="left").with_columns(pl.col("y").fill_null(0))
x_pure = x.join(pl.concat(own).select("i", "j"), on=["i", "j"], how="anti")
ea = pl.scan_parquet(CACHE / "recs.parquet").select(pl.col("row").alias("j"), (pl.col("addr").fill_null("").str.strip_chars() == "").alias("j_empty")) \
    .filter(pl.col("j").is_in(x["j"].unique().implode())).collect()
keeps = pl.read_parquet(ROOT / "v8_lab/ce/v2_build/out/holdout_keeps.parquet", columns=["i", "j", "ACNRST_3s_v2ce"])
kt = keeps.join(truth.select("i", "j", pl.lit(True).alias("t")), on=["i", "j"], how="left").with_columns(pl.col("t").fill_null(False))
agg = hold.select("i", "n_true").join(kt.group_by("i").agg(pl.col("ACNRST_3s_v2ce").sum().alias("np"),
                                                           (pl.col("ACNRST_3s_v2ce") & pl.col("t")).sum().alias("tp")), on="i", how="left").fill_null(0)


def f05(n_pred, n_true, tp):
    n_pred, n_true, tp = (np.asarray(v, np.float64) for v in (n_pred, n_true, tp))
    with np.errstate(divide="ignore", invalid="ignore"):
        p, r = tp / n_pred, tp / n_true
        f = np.where(tp > 0, 1.25 * p * r / (0.25 * p + r), 0.0)
    return np.where(n_true == 0, (n_pred == 0).astype(np.float64), f)


base = f05(agg["np"], agg["n_true"], agg["tp"]).mean()


def ceiling(sel):
    a2 = agg.join(sel.filter(pl.col("y") == 1).group_by("i").len("r"), on="i", how="left").fill_null(0)
    return round(float(f05(a2["np"] + a2["r"], a2["n_true"], a2["tp"] + a2["r"]).mean() - base), 6)


for name, d in (("vs_cache", x), ("pure_maxdf_effect", x_pure)):
    d = d.join(ea, on="j", how="left").join(dense.with_columns(pl.lit(True).alias("in_dense")), on=["i", "j"], how="left") \
         .with_columns(pl.col("in_dense").fill_null(False))
    rec = d.filter(pl.col("y") == 1)
    R[name] = {"new_per_s1": round(d.height / 1e5, 3), "recovered": rec.height, "share_of_missed": round(rec.height / missed.height, 4),
               "recovered_empty_addr": int(rec["j_empty"].sum()), "recovered_not_in_dense_m1r1": int((~rec["in_dense"]).sum()),
               "ceiling_gain": ceiling(d), "ceiling_gain_beyond_dense": ceiling(d.filter(~pl.col("in_dense")))}
print(json.dumps(R, indent=1))
json.dump(R, open(O / f"a03_md{MD}_k{K}r{RR}.json", "w"), indent=1)
