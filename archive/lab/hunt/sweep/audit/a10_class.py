"""Calibration by text class of j (unkept own candidates and kept pairs, p in [0.2, 0.98)); owner ties."""
import polars as pl
R = "/Users/adityarohilla/Documents/amazon-ml-2026"
K = pl.read_parquet("hold_keeps.parquet").with_columns(pmax_j=pl.col("p").max().over("j"), nkj=pl.col("keep").sum().over("j"))
print("j kept by >1 holdout S1:", K.filter(pl.col("keep") & (pl.col("nkj") > 1)).group_by("y").len())
U = K.filter((pl.col("p") >= 0.2) & (pl.col("p") < 0.98) & (pl.col("p") >= pl.col("pmax_j")))
TX = pl.scan_parquet(f"{R}/v8_lab/ce/setup/out/text_train.parquet").filter(pl.col("row").is_in(pl.concat([U["j"], U["i"]]).unique().implode())).collect()
U = U.join(TX.select(j="row", t="text"), on="j").join(TX.select(i="row", s="text"), on="i")
nm = pl.col("t").str.splitn(" | ", 2).struct.field("field_0")
U = U.with_columns(cls=pl.when(nm.str.contains(r"\.(com|net|org|in|co)\b|^@|www")).then(pl.lit("web"))
    .when(nm.str.contains(r"\b(dba|d/b/a|aka|a/k/a|f/k/a|fka|trading as|doing business as)\b")).then(pl.lit("alias"))
    .when(nm.str.contains(r"[^\x00-\x7F]") & ~nm.str.contains(r"[a-z]")).then(pl.lit("nonlatin"))
    .when(nm.str.contains(r"[^\x00-\x7F]")).then(pl.lit("mixed"))
    .otherwise(pl.lit("plain")), emp=pl.col("t").str.strip_chars().str.ends_with("|"))
print(U.group_by("keep", "cls", "emp", "country").agg(pl.len(), yr=pl.col("y").mean(), pm=pl.col("p").mean()).with_columns(gap=pl.col("yr") - pl.col("pm"))
      .filter(pl.col("len") >= 30).sort("gap", descending=True))
