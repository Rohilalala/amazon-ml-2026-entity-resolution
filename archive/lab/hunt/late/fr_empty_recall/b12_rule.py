import polars as pl,re,unicodedata,difflib,sys
B='/Users/adityarohilla/Documents/amazon-ml-2026/'
D=B+'v8_lab/hunt/late/fr_empty_recall/'
exec(open(D+'b11_consistent.py').read().split("x=tp.filter")[0].split("ids=pl.read_parquet")[0])  # imports only
LEG={'sa','sas','sasu','sarl','eurl','sci','snc','ei','cie','et','de','du','des','la','le','les','france','inc','llc','ltd','limited','private','pvt','corp','co','the','and','of','m','s'}
def na(s): return unicodedata.normalize('NFKD',s or '').encode('ascii','ignore').decode().lower()
def cons(n1,n2):
  t=[x for x in re.findall(r'[a-z0-9]+',na(n1)) if x not in LEG]
  s2=re.sub(r'[^a-z0-9]','',na(n2)); raw=(n2 or '').strip()
  if not t: return None
  if re.fullmatch(r'[A-Za-z]{2,4}',raw) and raw.upper()==raw:
    return 'init' if raw.lower() in (''.join(x[0] for x in t), ''.join(x[0] for x in re.findall(r'[a-z0-9]+',na(n1)))) else None
  if (' ' not in raw) and ('com' in s2 or raw[:1] in '@#'):
    return 'web' if all(x[:4] in s2 for x in t[:2]) else None
  return None
STOP=set('rue r avenue av ave bd boulevard blvd allee all place pl impasse imp quai q chemin ch route rte cours de du des la le les bis ter road rd street st lane ln drive dr suite ste no floor unit shop plot sector near nr opp main cross the and sur cite square sq'.split())
def seg(a):
  if not a: return None
  for p in na(a).split(','):
    if re.search(r'\d',p): return p
  return None
def hn(a):
  s=seg(a); 
  if s is None: return None
  return int(re.search(r'\d+',s).group())
def addr_ok(a1,a2):
  if not a1 or not a2: return False
  s=seg(a1)
  if s is None: return False
  t1=[t for t in re.findall(r'[a-z]+',s) if len(t)>=3 and t not in STOP]; t2=set(re.findall(r'[a-z]+',na(a2)))
  if not t1: return False
  if sum(any(difflib.SequenceMatcher(None,x,y).ratio()>=0.7 for y in t2) for x in t1)/len(t1)<0.5: return False
  h1,h2=hn(a1),hn(a2)
  return h2 is None or h1==h2 or not re.search(r'\d',na(a2))
# ---- holdout analog
hk=pl.read_parquet(B+'v8_lab/hunt/sweep/audit/hold_keeps.parquet')
kept_j=hk.filter('keep')['j'].unique(); kept_i=hk.filter('keep')['i'].unique()
t=pl.read_parquet(B+'v8_lab/ce/setup/out/text_train.parquet')
c=hk.filter(~pl.col('keep') & ~pl.col('j').is_in(kept_j.implode()))
c=c.join(t.rename({'row':'j','text':'t2'}),on='j')
c=c.with_columns(pl.col('t2').str.split_exact(' | ',1).struct.field('field_0').alias('n2'),pl.col('t2').str.split_exact(' | ',1).struct.field('field_1').alias('a2'))
c=c.filter(pl.col('n2').str.contains(r'^\S+$'))
c=c.join(t.rename({'row':'i','text':'t1'}),on='i').with_columns(pl.col('t1').str.split_exact(' | ',1).struct.field('field_0').alias('n1'),pl.col('t1').str.split_exact(' | ',1).struct.field('field_1').alias('a1'))
# text_train is lowercased -> initials check on lowercase raw
def cons_lc(n1,n2):
  raw=(n2 or '').strip()
  if re.fullmatch(r'[a-z]{2,4}',raw):
    t=[x for x in re.findall(r'[a-z0-9]+',na(n1)) if x not in LEG]
    return 'init' if t and raw in (''.join(x[0] for x in t), ''.join(x[0] for x in re.findall(r'[a-z0-9]+',na(n1)))) else None
  return cons(n1,n2)
c=c.with_columns(pl.struct('n1','n2').map_elements(lambda r: cons_lc(r['n1'],r['n2']),return_dtype=pl.Utf8).alias('ct')).filter(pl.col('ct').is_not_null())
c=c.with_columns(pl.struct('a1','a2').map_elements(lambda r: addr_ok(r['a1'],r['a2']),return_dtype=pl.Boolean).alias('aok'))
c=c.with_columns(pl.col('i').is_in(kept_i.implode()).not_().alias('i_empty'))
c.write_parquet(D+'hold_rule_cands.parquet')
print('HOLDOUT analog (unkept, j unclaimed, name-consistent web/initials):')
print(c.group_by('ct','aok','i_empty').agg(pl.len(),pl.col('y').mean().alias('prec')).sort('ct','aok','i_empty'))
