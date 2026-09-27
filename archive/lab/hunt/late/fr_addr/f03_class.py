import polars as pl
O='v8_lab/hunt/late/fr_addr/'
pr=pl.read_parquet(O+'fr_pairs.parquet')
c=pl.col
pr=pr.with_columns(
 hn=pl.when(c.a_hn.is_null()|c.b_hn.is_null()).then(pl.lit('miss')).when(c.a_hn==c.b_hn).then(pl.lit('eq')).otherwise(pl.lit('diff')),
 suf=pl.when(c.a_suf.fill_null('')==c.b_suf.fill_null('')).then(pl.lit('eq')).when(c.a_suf.is_null()|c.b_suf.is_null()).then(pl.lit('one')).otherwise(pl.lit('both_diff')),
 st=pl.when(c.a_st.is_null()|c.b_st.is_null()).then(pl.lit('miss')).when(c.a_st==c.b_st).then(pl.lit('eq')).otherwise(pl.lit('diff')),
 nm=pl.when(c.a_nm.is_null()|c.b_nm.is_null()).then(pl.lit('miss')).when(c.a_nm==c.b_nm).then(pl.lit('eq')).otherwise(pl.lit('diff')),
 pb=(c.p4.fill_null(-1)*10).floor().clip(-1,9))
pr.write_parquet(O+'fr_cls.parquet')
pl.Config.set_tbl_rows(80)
print(pr.group_by('hn','suf','st','nm').agg(n=pl.len(),kept=c.kept.sum(),hi=(c.p4>0.3).sum(),mp=c.p4.mean()).filter(c.n>200).sort('n',descending=True))
