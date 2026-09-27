import polars as pl, re, unicodedata
from collections import Counter
pr=pl.read_parquet('v8_lab/hunt/late/fr_addr/fr_cls.parquet')
REG={'nouvelle aquitaine','hauts de france','pays de la loire','gironde','nord','pas de calais','loire atlantique','france'}
def city(a):
    if a is None: return None
    s=''.join(ch for ch in unicodedata.normalize('NFKD',a) if not unicodedata.combining(ch)).lower()
    cs=[]
    for c in s.split(','):
        c=re.sub(r'[\-\.\']',' ',c); c=re.sub(r'\bst\b','saint',c); c=re.sub(r'\bste\b','sainte',c); c=re.sub(r'\d{5}','',c); c=re.sub(r'\s+',' ',c).strip()
        if not c or c in REG or re.search(r'\d',c): continue
        cs.append(c)
    return cs[-1] if cs else None  # ponytail: last non-region non-numeric component; street w/o number may leak in
u={}
for col in ('aa','ab'):
    for a in pr[col].unique().to_list(): u[a]=city(a)
pr=pr.with_columns(ca=pl.col('aa').replace_strict(u,default=None),cb=pl.col('ab').replace_strict(u,default=None))
KC=Counter(pr.filter(pl.col('kept'))['ca'].drop_nulls().to_list()); print(KC.most_common(25))
pr.select('i','j','kept','p4','hn','suf','st','nm','ca','cb','na','aa','nb','ab').write_parquet('v8_lab/hunt/late/fr_addr/fr_city.parquet')
