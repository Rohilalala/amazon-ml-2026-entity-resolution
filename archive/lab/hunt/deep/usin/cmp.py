import polars as pl
O = "/Users/adityarohilla/Documents/amazon-ml-2026/v8_lab/hunt/deep/usin/out/"
R = "/Users/adityarohilla/Documents/amazon-ml-2026/"
h = pl.read_parquet(O + "holdout_cats.parquet"); t = pl.read_parquet(O + "test_cats.parquet")
ns1h = pl.read_parquet(R + "v8_lab/ce/setup/out/holdout_s1.parquet").group_by("country").len()
ns1t = pl.read_parquet(R + "v8_lab/ce/setup/out/test_s1.parquet").group_by("country").len()
print(ns1h, ns1t)
H = h.group_by("country", "hr", "nr").agg(h_true=(pl.col("y") == 1).sum(), h_keep=pl.col("keep").sum(), h_tp=((pl.col("y") == 1) & pl.col("keep")).sum())
T = t.group_by("country", "hr", "nr").agg(t_keep=pl.col("keep").sum(), t_act=pl.col("p4").is_not_null().sum())
M = H.join(T, on=["country", "hr", "nr"], how="full", coalesce=True).fill_null(0)
M = M.join(ns1h.rename({"len": "nh"}), on="country").join(ns1t.rename({"len": "nt"}), on="country")
M = M.with_columns(truth_ps=pl.col("h_true") / pl.col("nh") * 1000, hkeep_ps=pl.col("h_keep") / pl.col("nh") * 1000,
                   tkeep_ps=pl.col("t_keep") / pl.col("nt") * 1000, prec=pl.col("h_tp") / pl.col("h_keep"), rec=pl.col("h_tp") / pl.col("h_true"))
M = M.with_columns(ratio_t_h=pl.col("tkeep_ps") / pl.col("hkeep_ps"))
pl.Config.set_tbl_rows(100); pl.Config.set_tbl_cols(20); pl.Config.set_tbl_width_chars(250)
for c in ["us", "india"]:
    print(M.filter(pl.col("country") == c).select("hr", "nr", "h_true", "h_keep", "prec", "rec", "truth_ps", "hkeep_ps", "tkeep_ps", "ratio_t_h", "t_keep").sort("t_keep", descending=True).with_columns(pl.col(pl.Float64).round(3)))
M.write_parquet(O + "cmp.parquet")
