"""Class counts of all FN pairs (location x j text class)."""
import polars as pl
R = "/Users/adityarohilla/Documents/amazon-ml-2026"
FN = pl.read_parquet("fn_all.parquet"); TX = pl.read_parquet("fn_text.parquet")
K = pl.read_parquet("hold_keeps.parquet", columns=["i", "j", "p"]).with_columns(pmax_j=pl.col("p").max().over("j")).drop("p")
FN = FN.join(K, on=["i", "j"], how="left").join(TX.select(j="row", t="text"), on="j").join(TX.select(i="row", s="text"), on="i")
nm, snm = pl.col("t").str.splitn(" | ", 2).struct.field("field_0"), pl.col("s").str.splitn(" | ", 2).struct.field("field_0")
FN = FN.with_columns(where=pl.when(~pl.col("incand")).then(pl.when(pl.col("indense")).then(pl.lit("dense_not_added")).otherwise(pl.lit("not_retrieved")))
        .when(pl.col("p") < pl.col("pmax_j")).then(pl.lit("owner_conflict")).when(pl.col("p") >= 0.3).then(pl.lit("cutoff_p>=0.3")).otherwise(pl.lit("low_p<0.3")),
    jtype=pl.when(nm.str.contains(r"[^\x00-\x7F]") & ~nm.str.contains(r"[a-z]")).then(pl.lit("transliteration"))
        .when(nm.str.contains(r"\.(com|net|org|in|co)\b|^@|www")).then(pl.lit("website"))
        .when(nm.str.contains(r"\b(dba|d/b/a|aka|a/k/a|f/k/a|fka|trading as|doing business as)\b")).then(pl.lit("alias_dba"))
        .when(pl.col("t").str.strip_chars().str.ends_with("|")).then(pl.lit("empty_addr"))
        .when(nm.str.split(" ").list.set_intersection(snm.str.split(" ")).list.len() == 0).then(pl.lit("brand_no_name_overlap"))
        .otherwise(pl.lit("name_variant_with_addr")))
tab = FN.group_by("where", "jtype").agg(pl.len()).sort("len", descending=True)
with pl.Config(tbl_rows=40): print(tab)
tab.write_csv("fn_classes.csv")
