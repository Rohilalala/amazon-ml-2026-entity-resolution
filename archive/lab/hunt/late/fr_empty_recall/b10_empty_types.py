import polars as pl,re,unicodedata,difflib
D='/Users/adityarohilla/Documents/amazon-ml-2026/v8_lab/hunt/late/fr_empty_recall/'
ids=pl.read_parquet(D+'ids.parquet'); tp=pl.read_parquet(D+'empty_cands.parquet')
a=ids.select(pl.col('row').alias('i'),pl.col('business_name').alias('n1'),pl.col('business_address').alias('a1'))
b=ids.select(pl.col('row').alias('j'),pl.col('business_name').alias('n2'),pl.col('business_address').alias('a2'))
tp=tp.join(a,on='i').join(b,on='j')
def na(s): return unicodedata.normalize('NFKD',s or '').encode('ascii','ignore').decode().lower()
def hn(x):
  m=re.search(r'\d+',na(x)) if x else None; return None if m is None else int(m.group())
web=pl.col('n2').str.contains(r'(?i)\.com|^[@#]|com$') & ~pl.col('n2').str.strip_chars().str.contains(' ')
init=pl.col('n2').str.strip_chars().str.contains(r'^[A-Z]{1,4}$')
x=tp.filter((web|init) & pl.col('owner').is_null())
x=x.with_columns(pl.struct('a1','a2').map_elements(lambda r: hn(r['a1']) is not None and hn(r['a1'])==hn(r['a2']),return_dtype=pl.Boolean).alias('samehn'))
print('empty S1:',tp['i'].n_unique(),' unclaimed web/init cands:',x.height,' S1s:',x['i'].n_unique(),' same-house:',x.filter('samehn').height, x.filter('samehn')['i'].n_unique())
for r in x.filter('samehn').sample(min(40,x.filter('samehn').height),seed=1).iter_rows(named=True):
  print(f"p={r['prob']:.3f} p4={r['p4']} {r['n1']} | {r['a1']}\n    -> {r['n2']} | {r['a2']}")
