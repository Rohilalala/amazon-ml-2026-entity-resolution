"""Write widened sparse NEW pairs (not in the split's cached candidate set) for HOLDOUT and TEST.
usage: w01_write.py K A R B
  forward: rank <= K and cos >= A * f10 (f10 = the S1's 10th-best forward cosine in that source)
  reverse: rev_rank <= R and cos >= B * r3 (r3 = the S2/S3 record's 3rd-best reverse cosine)   (R=3 -> no reverse)
-> out/pairs/{holdout,test}_sk{K}a{A}_r{R}b{B}.parquet
   s1_row, j_row, sparse_cos, sparse_rank (forward rank, null = reverse only), new_rank (forward rank among non-cached
   targets), rev_rank, f10, r3, source, country[, y]"""
import json, sys
from pathlib import Path
import polars as pl
sys.path.insert(0, str(Path(__file__).parent))
import wd  # noqa: E402
wd.start()

ROOT = Path("/Users/adityarohilla/Documents/amazon-ml-2026")
CACHE = {"train": ROOT / "cache/58f0a60c5c264fdd", "test": ROOT / "cache/3ee42f05d06efbcb"}
O = Path(__file__).parent / "out"
(O / "pairs").mkdir(exist_ok=True)
K, A, R, B = int(sys.argv[1]), float(sys.argv[2]), int(sys.argv[3]), float(sys.argv[4])
key = f"sk{K}a{A}_r{R}b{B}"
truth = pl.read_parquet(ROOT / "v8_lab/ce/setup/out/truth_train.parquet").with_columns(pl.lit(1, pl.Int8).alias("y"))
hold = pl.read_parquet(ROOT / "v8_lab/ce/setup/out/holdout_s1.parquet")
res = {}
for name, split, countries in (("holdout", "train", ("us", "india")), ("test", "test", ("us", "india", "france"))):
    parts = []
    for c in countries:
        for s in (2, 3):
            t = f"{split}_{c}_s{s}"
            fw = pl.read_parquet(O / f"fwd_{t}.parquet")
            rv = pl.read_parquet(O / f"rev_{t}.parquet")
            qi = fw["i"].unique().implode()
            cand = pl.scan_parquet(CACHE[split] / "pairs.parquet").filter((pl.col("jsrc") == s) & pl.col("i").is_in(qi)) \
                .select("i", "j").collect().with_columns(pl.lit(True).alias("in_cand"))
            f10 = fw.filter(pl.col("rank") == 10).select("i", pl.col("cos").alias("f10"))
            r3 = rv.filter(pl.col("rev_rank") == 3).select("j", pl.col("cos").alias("r3"))
            fw = fw.join(cand, on=["i", "j"], how="left").with_columns(pl.col("in_cand").fill_null(False))
            fw = fw.with_columns(pl.col("rank").rank("ordinal").over("i", "in_cand").cast(pl.UInt8).alias("new_rank"))
            fw = fw.filter(~pl.col("in_cand")).join(f10, on="i", how="left")
            sf = fw.filter((pl.col("rank") <= K) & (pl.col("cos") >= A * pl.col("f10")))
            rv = rv.filter(pl.col("i").is_in(qi) & (pl.col("rev_rank") <= R)).join(r3, on="j", how="left")
            sr = rv.filter(pl.col("cos") >= B * pl.col("r3")).join(cand, on=["i", "j"], how="anti")
            x = sf.select("i", "j", "cos", "rank", "new_rank").join(sr.select("i", "j", "cos", "rev_rank"), on=["i", "j"], how="full", coalesce=True)
            x = x.with_columns(pl.coalesce("cos", "cos_right").alias("cos")).drop("cos_right")
            x = x.join(f10, on="i", how="left").join(r3, on="j", how="left")
            parts.append(x.with_columns(pl.lit(s, pl.UInt8).alias("source"), pl.lit(c).alias("country")))
    x = pl.concat(parts).select(pl.col("i").alias("s1_row"), pl.col("j").alias("j_row"), pl.col("cos").alias("sparse_cos"),
                                pl.col("rank").alias("sparse_rank"), "new_rank", "rev_rank", "f10", "r3", "source", "country")
    if name == "holdout":
        x = x.join(truth.rename({"i": "s1_row", "j": "j_row"}), on=["s1_row", "j_row"], how="left").with_columns(pl.col("y").fill_null(0))
    assert x.select(pl.struct("s1_row", "j_row").n_unique()).item() == x.height
    n_s1 = hold.height if name == "holdout" else pl.scan_parquet(CACHE["test"] / "recs.parquet").filter(pl.col("src") == 1).select(pl.len()).collect().item()
    p = O / "pairs" / f"{name}_{key}.parquet"
    x.sort("s1_row", "source", "sparse_cos", descending=[False, False, True]).write_parquet(p)
    res[name] = {"path": str(p), "pairs": x.height, "per_s1": round(x.height / n_s1, 3), "s1_with_pairs": x["s1_row"].n_unique(),
                 "by_country": {c: round(x.filter(pl.col("country") == c).height / (hold.filter(pl.col("country") == c).height if name == "holdout" else
                                     pl.scan_parquet(CACHE["test"] / "recs.parquet").filter((pl.col("src") == 1) & (pl.col("country_key") == c)).select(pl.len()).collect().item()), 3)
                                for c in x["country"].unique().sort()}} | ({"trues": int(x["y"].sum())} if name == "holdout" else {})
    print(res[name], flush=True)
json.dump(res, open(O / "pairs" / f"w01_{key}.json", "w"), indent=1)
