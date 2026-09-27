"""S1 sets for the recall step: HOLDOUT (100k, v8 holdout) and HEAD (200k train S1 outside the v7 500k sample and the
holdout, seed 2026). -> out/sets/{holdout,head,q_train}_rows.npy"""
import os, numpy as np, polars as pl
ROOT = "/Users/adityarohilla/Documents/amazon-ml-2026"
O = os.path.join(os.path.dirname(os.path.abspath(__file__)), "out", "sets"); os.makedirs(O, exist_ok=True)
sp = pl.read_parquet(f"{ROOT}/v8_lab/day1b/r3_regime/out/regime/split.parquet", columns=["row", "holdout", "sample"])
ho = pl.read_parquet(f"{ROOT}/v8_lab/ce/setup/out/holdout_s1.parquet")["i"]
assert set(ho.to_list()) == set(sp.filter("holdout")["row"].to_list())
oof_s1 = pl.scan_parquet(f"{ROOT}/code/business_entity_resolution/artifacts/v7/oof.parquet").select("i").unique().collect()["i"]
assert set(oof_s1.to_list()) == set(sp.filter("sample")["row"].to_list())
pool = sp.filter(~pl.col("holdout") & ~pl.col("sample"))
head = pool.sample(200_000, seed=2026).sort("row")["row"]
assert not set(head.to_list()) & (set(ho.to_list()) | set(oof_s1.to_list()))
np.save(f"{O}/holdout_rows.npy", np.sort(ho.to_numpy()).astype(np.uint32))
np.save(f"{O}/head_rows.npy", head.to_numpy().astype(np.uint32))
np.save(f"{O}/q_train_rows.npy", np.sort(np.concatenate([ho.to_numpy(), head.to_numpy()])).astype(np.uint32))
print({"pool": pool.height, "head": len(head), "holdout": len(ho)})
