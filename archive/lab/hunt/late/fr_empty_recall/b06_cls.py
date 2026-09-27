import polars as pl,re,unicodedata,sys
B='/Users/adityarohilla/Documents/amazon-ml-2026/'
D=B+'v8_lab/hunt/late/fr_empty_recall/'
ids=pl.read_parquet(D+'ids.parquet')
fr=pl.read_parquet(D+'fr_radd.parquet').filter(pl.col('inbest')==1)
a=ids.select(pl.col('row').alias('i'),pl.col('business_name').alias('n1'),pl.col('business_address').alias('a1'))
b=ids.select(pl.col('row').alias('j'),pl.col('business_name').alias('n2'),pl.col('business_address').alias('a2'))
fr=fr.join(a,on='i').join(b,on='j')
LEG={'sa','sas','sasu','sarl','eurl','sci','snc','ei','et','cie','fils','associes','associés','france','groupe','holding','international','developpement','développement','services','s','a','r','l','u','c','i','e','n'}
def norm(s):
  s=unicodedata.normalize('NFKD',s or '').encode('ascii','ignore').decode().lower()
  return [t for t in re.split(r'[^a-z0-9]+',s) if t]
def cls(n1,n2):
  if re.search(r'\.com|com$',(n2 or '').lower()) and ' ' not in (n2 or '').strip(): return 'web'
  if re.fullmatch(r'[A-Z]{1,4}',(n2 or '').strip()): return 'init'
  t1=set(norm(n1))-LEG; t2=set(norm(n2))-LEG
  if t1==t2: return 'same_core'
  if not (t2-t1): return 'subset'
  if not (t1-t2): return 'superset'
  return 'diffword'
fr=fr.with_columns(pl.struct('n1','n2').map_elements(lambda r: cls(r['n1'],r['n2']),return_dtype=pl.Utf8).alias('cls'))
fr.write_parquet(D+'fr_radd_cls.parquet')
print(fr['cls'].value_counts().sort('count',descending=True))
if len(sys.argv)>1:
  for r in fr.filter(pl.col('cls')==sys.argv[1]).sample(min(int(sys.argv[2]),fr.filter(pl.col('cls')==sys.argv[1]).height),seed=3).iter_rows(named=True):
    print(f"{r['n1']} | {r['a1']}\n    -> {r['m'][:2]} {r['n2']} | {r['a2']}")
