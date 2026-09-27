import polars as pl,re,unicodedata
D='/Users/adityarohilla/Documents/amazon-ml-2026/v8_lab/hunt/late/fr_empty_recall/'
ids=pl.read_parquet(D+'ids.parquet'); tp=pl.read_parquet(D+'empty_cands.parquet').filter(pl.col('owner').is_null())
a=ids.select(pl.col('row').alias('i'),pl.col('business_name').alias('n1'),pl.col('business_address').alias('a1'))
b=ids.select(pl.col('row').alias('j'),pl.col('business_name').alias('n2'),pl.col('business_address').alias('a2'))
tp=tp.join(a,on='i').join(b,on='j')
LEG={'sa','sas','sasu','sarl','eurl','sci','snc','ei','cie','et','de','du','des','la','le','les','france'}
def na(s): return unicodedata.normalize('NFKD',s or '').encode('ascii','ignore').decode().lower()
def cons(n1,n2):
  t=[x for x in re.findall(r'[a-z0-9]+',na(n1)) if x not in LEG]
  s2=re.sub(r'[^a-z0-9]','',na(n2)); raw=(n2 or '').strip()
  if not t: return False
  if re.fullmatch(r'[A-Z]{2,4}',raw):
    return raw.lower() in (''.join(x[0] for x in t), ''.join(x[0] for x in re.findall(r'[a-z0-9]+',na(n1))))
  if (' ' not in raw) and ('com' in s2 or raw[:1] in '@#'):
    return len(t)>=1 and all(x[:4] in s2 for x in t[:2])
  return False
x=tp.filter(pl.struct('n1','n2').map_elements(lambda r: cons(r['n1'],r['n2']),return_dtype=pl.Boolean))
print('empty S1 with an unclaimed name-consistent web/initials candidate:',x.height,x['i'].n_unique())
for r in x.sort('prob',descending=True).head(40).iter_rows(named=True):
  print(f"p={r['prob']:.3f} p4={r['p4']} {r['n1']} | {r['a1']}\n    -> {r['n2']} | {r['a2']}")
x.write_parquet(D+'empty_consistent.parquet')
