import polars as pl,re,unicodedata,difflib
B='/Users/adityarohilla/Documents/amazon-ml-2026/'
D=B+'v8_lab/hunt/late/fr_empty_recall/'
STOP=set('rue r avenue av ave bd boulevard blvd allee all place pl impasse imp quai q chemin ch route rte cours de du des la le les bis ter road rd street st lane ln drive dr suite ste no floor unit shop plot sector near nr opp main cross the and sur cite square sq'.split())
def na(s): return unicodedata.normalize('NFKD',s or '').encode('ascii','ignore').decode().lower()
def seg(a):
  if not a: return None
  for p in na(a).split(','):
    if re.search(r'\d',p): return p
  return None
def hn(a):
  s=seg(a)
  if s is None: return None
  m=re.search(r'\d+',s); return int(m.group())
def toks(s): return [t for t in re.findall(r'[a-z]+',s or '') if len(t)>=3 and t not in STOP]
def sdiff(a1,a2):
  s=seg(a1)
  if s is None or not a2: return None
  t1=toks(s); t2=set(re.findall(r'[a-z]+',na(a2)))
  if not t1: return None
  ok=sum(any(difflib.SequenceMatcher(None,x,y).ratio()>=0.7 for y in t2) for x in t1)
  return ok/len(t1)<0.5
def tag(df,a1,a2):
  return df.with_columns(pl.struct(a1,a2).map_elements(lambda r: sdiff(r[a1],r[a2]),return_dtype=pl.Boolean).alias('sdiff'),
    pl.struct(a1,a2).map_elements(lambda r: (None if hn(r[a1]) is None or hn(r[a2]) is None else hn(r[a1])!=hn(r[a2])),return_dtype=pl.Boolean).alias('hdiff'))
fr=pl.read_parquet(D+'fr_radd_cls.parquet')
fr=tag(fr,'a1','a2'); fr.write_parquet(D+'fr_radd_tag.parquet')
print('FRANCE recall adds in best:',fr.height)
print(fr.group_by('sdiff','hdiff').len().sort('sdiff','hdiff'))
# holdout analog
h=pl.read_parquet(B+'v8_lab/hunt/sweep/audit/hold_adds.parquet').filter(pl.col('add'))
t=pl.read_parquet(B+'v8_lab/ce/setup/out/text_train.parquet')
h=h.join(t.rename({'row':'i','text':'t1'}),on='i').join(t.rename({'row':'j','text':'t2'}),on='j')
h=h.with_columns(pl.col('t1').str.split_exact(' | ',1).struct.field('field_1').alias('a1'),pl.col('t2').str.split_exact(' | ',1).struct.field('field_1').alias('a2'))
h=tag(h,'a1','a2'); h.write_parquet(D+'hold_adds_tag.parquet')
print('HOLD recall adds',h.height,'prec',h['y'].mean())
print(h.group_by('sdiff','hdiff').agg(pl.len(),pl.col('y').mean().alias('prec')).sort('sdiff','hdiff'))
