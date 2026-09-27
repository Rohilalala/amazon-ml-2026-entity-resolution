import polars as pl, numpy as np
R = "/Users/adityarohilla/Documents/amazon-ml-2026"
K = pl.read_parquet("hold_keeps.parquet").with_columns(pmax_j=pl.col("p").max().over("j"), nk=pl.col("keep").sum().over("i"))
TX = pl.read_parquet(f"{R}/v8_lab/ce/setup/out/text_train.parquet")
TX = TX.filter(pl.col("row").is_in(K.filter(pl.col("p") > 0.2)["j"].unique().implode())).select("row", jemp=pl.col("text").str.ends_with(" | ") | pl.col("text").str.ends_with("|"))
K = K.filter(pl.col("p") > 0.2).join(TX, left_on="j", right_on="row", how="left")
K = K.with_columns(bin=(pl.col("p") * 10).floor() / 10, own=pl.col("p") >= pl.col("pmax_j"))
print(K.filter(~pl.col("keep") & pl.col("own")).group_by("bin", "jemp").agg(pl.len(), yr=pl.col("y").mean()).sort("jemp", "bin"))
print(K.filter(pl.col("keep")).group_by("bin", "jemp").agg(pl.len(), yr=pl.col("y").mean()).sort("jemp", "bin"))
print(K.filter(~pl.col("own")).group_by("bin").agg(pl.len(), yr=pl.col("y").mean()).sort("bin"))
