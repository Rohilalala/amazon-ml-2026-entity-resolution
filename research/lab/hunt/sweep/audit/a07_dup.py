"""Within-S1 candidates with identical text to a kept candidate: y-rate of the unkept twin; and kept twins disagreeing."""
import polars as pl
R = "/Users/adityarohilla/Documents/amazon-ml-2026"
K = pl.read_parquet("hold_keeps.parquet").filter(pl.col("p") > 0.01).with_columns(pmax_j=pl.col("p").max().over("j"))
TX = pl.scan_parquet(f"{R}/v8_lab/ce/setup/out/text_train.parquet").filter(pl.col("row").is_in(K["j"].unique().implode())).collect()
K = K.join(TX.select(j="row", t="text"), on="j")
kept = K.filter("keep").select("i", "t", kp=pl.col("p"), ky=pl.col("y")).unique(["i", "t"])
U = K.filter(~pl.col("keep")).join(kept, on=["i", "t"])
U = U.with_columns(own=pl.col("p") >= pl.col("pmax_j"), emp=pl.col("t").str.strip_chars().str.ends_with("|"))
print(U.group_by("own", "emp").agg(pl.len(), yr=pl.col("y").mean(), kyr=pl.col("ky").mean(), pm=pl.col("p").mean()).sort("own", "emp"))
print(U.filter("own").group_by(pb=(pl.col("p") * 5).floor() / 5).agg(pl.len(), yr=pl.col("y").mean()).sort("pb"))
# kept twins with disagreeing labels
KK = K.filter("keep").group_by("i", "t").agg(n=pl.len(), ys=pl.col("y").sum()).filter(pl.col("n") > 1)
print("kept twin groups", KK.height, "mixed", KK.filter((pl.col("ys") > 0) & (pl.col("ys") < pl.col("n"))).height, "all-false", KK.filter(pl.col("ys") == 0).height)
