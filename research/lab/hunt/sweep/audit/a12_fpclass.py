"""Class counts of all FP decisions (keeps + recall adds)."""
import polars as pl
R = "/Users/adityarohilla/Documents/amazon-ml-2026"
FP = pl.concat([pl.read_parquet("fp_keep.parquet").select("i", "j", "p", "country", src=pl.lit("keep")),
                pl.read_parquet("hold_adds.parquet").filter(pl.col("add") & (pl.col("y") == 0)).select("i", "j", p="prob")
                .join(pl.read_parquet(f"{R}/v8_lab/ce/setup/out/holdout_s1.parquet").select("i", "country"), on="i").with_columns(src=pl.lit("add"))], how="diagonal")
TX = pl.scan_parquet(f"{R}/v8_lab/ce/setup/out/text_train.parquet").filter(pl.col("row").is_in(pl.concat([FP["i"], FP["j"]]).unique().implode())).collect()
FP = FP.join(TX.select(j="row", t="text"), on="j").join(TX.select(i="row", s="text"), on="i").join(pl.read_parquet("s1_table.parquet").select("i", "n_true"), on="i")
num = lambda c: pl.col(c).str.splitn(" | ", 2).struct.field("field_1").str.extract(r"(\d+)")
FP = FP.with_columns(cls=pl.when(pl.col("t").str.strip_chars().str.ends_with("|")).then(pl.lit("empty_addr_chain"))
    .when(num("t") != num("s")).then(pl.lit("house_number_differs")).otherwise(pl.lit("same_house_name_variant")), nt0=pl.col("n_true") == 0)
print(FP.group_by("src", "cls", "nt0").agg(pl.len(), pm=pl.col("p").mean()).sort("len", descending=True))
