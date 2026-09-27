"""All FN pairs of the best holdout configuration, with where they sit (candidate / dense-recall / nowhere), + texts."""
import polars as pl
R = "/Users/adityarohilla/Documents/amazon-ml-2026"
K = pl.read_parquet("hold_keeps.parquet"); A = pl.read_parquet("hold_adds.parquet")
S = pl.read_parquet(f"{R}/v8_lab/ce/setup/out/holdout_s1.parquet")
Tr = pl.read_parquet(f"{R}/v8_lab/ce/setup/out/truth_train.parquet").join(S.select("i", "country"), on="i")
print("truth pairs", Tr.height, "n_true sum", S["n_true"].sum())
dec = pl.concat([K.filter("keep").select("i", "j"), A.filter("add").select("i", "j")]).with_columns(d=pl.lit(True))
FN = Tr.join(dec, on=["i", "j"], how="anti")
FN = FN.join(K.select("i", "j", "p", "keep").with_columns(incand=pl.lit(True)), on=["i", "j"], how="left") \
       .join(A.select("i", "j", "prob").with_columns(indense=pl.lit(True)), on=["i", "j"], how="left").fill_null(False)
print(FN.group_by("incand", "indense", "country").len().sort("len", descending=True))
TX = pl.read_parquet(f"{R}/v8_lab/ce/setup/out/text_train.parquet")
rows = pl.concat([FN["i"], FN["j"]]).unique()
TX.filter(pl.col("row").is_in(rows.implode())).write_parquet("fn_text.parquet")
FN.write_parquet("fn_all.parquet")
