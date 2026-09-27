# keep rate of same-name same-address France pairs by the raw street-type token each side used
import polars as pl, re, unicodedata
from rapidfuzz import fuzz
c=pl.col
pr=pl.read_parquet('v8_lab/hunt/late/fr_addr/fr_city.parquet')
x=pr.filter((c.hn=='eq')&(c.suf=='eq')&(c.st=='eq')&(c.nm=='eq')&(c.ca==c.cb))
x=x.with_columns(ns=pl.Series([fuzz.token_set_ratio((a or '').lower(),(b or '').lower()) for a,b in zip(x['na'],x['nb'])]))
x=x.filter(c.ns>=90)
R=re.compile(r"(?:^|,)\s*(?:n[o°º]\.?\s*|#\s*)?\d+\s*(?:bis|ter|quater|[a-hqt])?\b\s*([a-z]+\.?)",re.I)
def tok(a):
    s=''.join(ch for ch in unicodedata.normalize('NFKD',a or '') if not unicodedata.combining(ch))
    m=R.search(s); return m.group(1) if m else None
x=x.with_columns(ta=pl.Series([tok(a) for a in x['aa']]),tb=pl.Series([tok(a) for a in x['ab']]))
x.write_parquet('v8_lab/hunt/late/fr_addr/fr_samename_sameaddr.parquet')
pl.Config.set_tbl_rows(80)
print('n',x.height,'kept',x['kept'].sum())
g=x.with_columns(tb=c.tb.str.to_lowercase()).group_by('tb').agg(n=pl.len(),k=c.kept.sum(),mp=c.p4.mean()).with_columns(rate=c.k/c.n).filter(c.n>=100).sort('rate')
print(g)
