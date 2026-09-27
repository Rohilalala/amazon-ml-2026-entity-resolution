import polars as pl,re,sys
B='/Users/adityarohilla/Documents/amazon-ml-2026/'
D=B+'v8_lab/hunt/late/fr_empty_recall/'
fr=pl.read_parquet(D+'fr_radd_cls.parquet')
def hn(a):
  if a is None: return None
  m=re.search(r'\d+',a); return None if m is None else int(m.group())
fr=fr.with_columns(pl.col('a1').map_elements(hn,return_dtype=pl.Int64).alias('h1'),pl.col('a2').map_elements(hn,return_dtype=pl.Int64).alias('h2'))
fr=fr.with_columns(pl.when(pl.col('h1').is_null()|pl.col('h2').is_null()).then(pl.lit('na')).when(pl.col('h1')==pl.col('h2')).then(pl.lit('same')).otherwise(pl.lit('diff')).alias('hs'))
fr.write_parquet(D+'fr_radd_cls.parquet')
print(fr.group_by('cls','hs').len().sort('cls','hs'))
print(fr['hs'].value_counts())
if len(sys.argv)>1:
  for r in fr.filter(pl.col('hs')=='diff').sample(int(sys.argv[1]),seed=5).iter_rows(named=True):
    print(f"[{r['cls']}] {r['n1']} | {r['a1']}\n    -> {r['m'][:2]} {r['n2']} | {r['a2']}")
