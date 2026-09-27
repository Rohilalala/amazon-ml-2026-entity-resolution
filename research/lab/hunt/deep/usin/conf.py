import polars as pl
R = "/Users/adityarohilla/Documents/amazon-ml-2026/"; O = R + "v8_lab/hunt/deep/usin/out/"
tt = pl.read_parquet(R + "v8_lab/ce/setup/out/truth_train.parquet")
print("truth: pool recs with >1 S1:", tt.group_by("j").len().filter(pl.col("len") > 1).height, "of", tt["j"].n_unique())
h = pl.read_parquet(O + "holdout_cats.parquet"); t = pl.read_parquet(O + "test_cats.parquet")
hk = h.filter(pl.col("keep")); tk = t.filter(pl.col("keep"))
# holdout conflicts within holdout S1 only
for nm, d in [("holdout", hk), ("test", tk)]:
    c = d.group_by("j").len().filter(pl.col("len") > 1)
    print(nm, "kept pairs", d.height, "j kept by >1 S1:", c.height, "pairs involved", d.join(c.select("j"), on="j").height)
# per-S1 kept count distribution
ns = pl.read_parquet(R + "v8_lab/ce/setup/out/holdout_s1.parquet")
hs = ns.join(hk.group_by("i").len(), on="i", how="left").with_columns(pl.col("len").fill_null(0))
ts1 = pl.read_parquet(R + "v8_lab/ce/setup/out/test_s1.parquet").filter(pl.col("country").is_in(["us", "india"]))
ts = ts1.join(tk.group_by("i").len(), on="i", how="left").with_columns(pl.col("len").fill_null(0))
for c in ["us", "india"]:
    a = hs.filter(pl.col("country") == c); b = ts.filter(pl.col("country") == c)
    rows = []
    for k in range(10):
        rows.append((k, round((a["n_true"].clip(0, 9) == k).mean(), 4), round((a["len"].clip(0, 9) == k).mean(), 4), round((b["len"].clip(0, 9) == k).mean(), 4)))
    print(c, "k truth_hold keep_hold keep_test"); [print(r) for r in rows]
