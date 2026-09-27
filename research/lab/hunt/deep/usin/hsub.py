import polars as pl, re
R = "/Users/adityarohilla/Documents/amazon-ml-2026/"; O = R + "v8_lab/hunt/deep/usin/out/"
def sub(a, b):
    a, b = (a or "").lower(), (b or "").lower()
    if "-" in b and b.split("-")[0].lstrip("0") == a.lstrip("0"): return "range"
    if re.sub(r"[a-z/ ]+$", "", b) == a or re.sub(r"[a-z/ ]+$", "", a) == b: return "letter"
    da, db = re.sub(r"\D", "", a), re.sub(r"\D", "", b)
    if not da or not db: return "nonnum"
    if da == db: return "samedig"
    if len(db) == len(da) - 1 and da.startswith(db): return "trunc_end"
    if len(db) == len(da) - 1 and da.endswith(db): return "trunc_start"
    if len(db) > len(da) and (db.startswith(da) or db.endswith(da)): return "extra"
    d = abs(int(da[:12]) - int(db[:12]))
    return "pm2" if d <= 2 else "near20" if d <= 20 else "other"
def gib(nr): return nr in ("fzlo", "fz50")
ns1h = pl.read_parquet(R + "v8_lab/ce/setup/out/holdout_s1.parquet").group_by("country").len().rename({"len": "nh"})
ns1t = pl.read_parquet(R + "v8_lab/ce/setup/out/test_s1.parquet").group_by("country").len().rename({"len": "nt"})
out = {}
for side in ["holdout", "test"]:
    d = pl.read_parquet(O + side + "_cats.parquet").filter(pl.col("hr") == "diff")
    d = d.with_columns(pl.struct("h1", "h2").map_elements(lambda s: sub(s["h1"], s["h2"]), return_dtype=pl.String).alias("hs"),
                       pl.col("nr").is_in(["fzlo", "fz50"]).alias("lowname"))
    d.write_parquet(O + side + "_diff.parquet"); out[side] = d
H = out["holdout"].group_by("country", "lowname", "hs").agg(h_keep=pl.col("keep").sum(), h_true=(pl.col("y") == 1).sum(), h_tp=((pl.col("y") == 1) & pl.col("keep")).sum(), h_fp=((pl.col("y") == 0) & pl.col("keep")).sum())
T = out["test"].group_by("country", "lowname", "hs").agg(t_keep=pl.col("keep").sum())
M = H.join(T, on=["country", "lowname", "hs"], how="full", coalesce=True).fill_null(0).join(ns1h, on="country").join(ns1t, on="country")
M = M.with_columns(prec=pl.col("h_tp") / pl.col("h_keep"), exp_t=pl.col("h_keep") / pl.col("nh") * pl.col("nt")).with_columns(ratio=pl.col("t_keep") / pl.col("exp_t"))
pl.Config.set_tbl_rows(80); pl.Config.set_tbl_width_chars(220)
print(M.drop("nh", "nt").sort("country", "lowname", "t_keep", descending=True).with_columns(pl.col(pl.Float64).round(3)))
