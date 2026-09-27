import polars as pl
D='/Users/adityarohilla/Documents/amazon-ml-2026/v8_lab/hunt/late/fr_empty_recall/'
fr=pl.read_parquet(D+'fr_radd_tag.parquet')
x=fr.filter(pl.col('sdiff')==True)
print(x['cls'].value_counts())
for r in x.sample(45,seed=7).iter_rows(named=True):
  print(f"[{r['cls']}] {r['n1']} | {r['a1']}\n    -> {r['m'][:2]} {r['n2']} | {r['a2']}")
