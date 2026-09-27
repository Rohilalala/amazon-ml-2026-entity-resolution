import polars as pl, sys
R = "/Users/adityarohilla/Documents/amazon-ml-2026/"; O = R + "v8_lab/hunt/deep/usin/out/"
t = pl.read_parquet(O + "test_cats.parquet").filter(pl.col("keep"))
h = pl.read_parquet(O + "holdout_cats.parquet")
tt = pl.read_parquet(R + "v8_lab/ce/setup/out/text_test.parquet"); tr = pl.read_parquet(R + "v8_lab/ce/setup/out/text_train.parquet")
pl.Config.set_tbl_rows(60); pl.Config.set_fmt_str_lengths(90); pl.Config.set_tbl_width_chars(260)
for c, hr, nr in [("us", "diff", "fzlo"), ("us", "miss", "fz75"), ("india", "miss", "fz75")]:
    a = t.filter((pl.col("country") == c) & (pl.col("hr") == hr) & (pl.col("nr") == nr)).sample(18, seed=1)
    a = a.join(tt.rename({"row": "i", "text": "s1"}), on="i").join(tt.rename({"row": "j", "text": "cand"}), on="j")
    print("TEST", c, hr, nr, "p4 null frac", t.filter((pl.col("country") == c) & (pl.col("hr") == hr) & (pl.col("nr") == nr))["p4"].is_null().mean()); print(a.select("p4", "s1", "cand"))
    b = h.filter((pl.col("country") == c) & (pl.col("hr") == hr) & (pl.col("nr") == nr) & pl.col("keep") & (pl.col("y") == 0))
    b = b.join(tr.rename({"row": "i", "text": "s1"}), on="i").join(tr.rename({"row": "j", "text": "cand"}), on="j")
    print("HOLDOUT FP", b.height); print(b.select("s1", "cand").head(10))
