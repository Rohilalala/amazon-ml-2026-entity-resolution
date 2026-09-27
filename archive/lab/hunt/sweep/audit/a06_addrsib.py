"""Address-sibling adds on the holdout: record r whose full address text equals the address text of a kept j of S1 i
(r not decided for i). Precision from truth_train."""
import polars as pl
R = "/Users/adityarohilla/Documents/amazon-ml-2026"
K = pl.read_parquet("hold_keeps.parquet"); A = pl.read_parquet("hold_adds.parquet")
S = pl.read_parquet(f"{R}/v8_lab/ce/setup/out/holdout_s1.parquet")
dec = pl.concat([K.filter("keep").select("i", "j", "p"), A.filter("add").select("i", "j", p="prob")])
TXL = pl.scan_parquet(f"{R}/v8_lab/ce/setup/out/text_train.parquet").select("row", addr=pl.col("text").str.splitn(" | ", 2).struct.field("field_1").fill_null("").str.strip_chars(),
                                                                            name=pl.col("text").str.splitn(" | ", 2).struct.field("field_0"))
src = pl.scan_parquet(f"{R}/cache/58f0a60c5c264fdd/recs.parquet").select("row", "src")
dj = TXL.filter(pl.col("row").is_in(pl.concat([dec["j"], dec["i"]]).unique().implode())).collect()
dec = dec.join(dj.select(j="row", jaddr="addr"), on="j").join(dj.select(i="row", iaddr="addr"), on="i")
dec = dec.filter((pl.col("jaddr").str.len_chars() >= 8) & pl.col("jaddr").str.contains(r"\d"))
addrs = dec["jaddr"].unique()
M = TXL.filter(pl.col("addr").is_in(addrs.implode())).join(src, on="row").filter(pl.col("src") != 1).collect()
print("records sharing a kept address", M.height)
C = dec.join(M.select(r="row", jaddr="addr", rname="name"), on="jaddr").filter(pl.col("r") != pl.col("j"))
C = C.join(dec.select("i", r="j").with_columns(done=pl.lit(True)), on=["i", "r"], how="left").filter(pl.col("done").is_null())
C = C.group_by("i", "r").agg(pmax=pl.col("p").max(), nsrc=pl.len(), same_as_s1=(pl.col("jaddr") == pl.col("iaddr")).any(), rname=pl.col("rname").first(), jaddr=pl.col("jaddr").first())
T = pl.read_parquet(f"{R}/v8_lab/ce/setup/out/truth_train.parquet").with_columns(y=pl.lit(1))
C = C.join(T, left_on=["i", "r"], right_on=["i", "j"], how="left").fill_null(0).join(S.select("i", "country"), on="i")
# how many S1-candidates does r have (any holdout i) - r claimed by >1 holdout S1 via this rule
C = C.with_columns(n_i=pl.len().over("r"), in_cand=pl.struct("i", "r").is_in(K.select("i", r="j").to_struct().implode()))
print(C.group_by("country", "same_as_s1", "in_cand").agg(pl.len(), prec=pl.col("y").mean()).sort("country", "same_as_s1", "in_cand"))
print(C.group_by("country", n_i=pl.col("n_i").clip(1, 3)).agg(pl.len(), prec=pl.col("y").mean()).sort("country", "n_i"))
C.write_parquet("addrsib_hold.parquet")
