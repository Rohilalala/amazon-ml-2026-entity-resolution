# France region/department conflict between the two sides of a pair (analog of US state conflict)
import polars as pl, re, unicodedata
c=pl.col
pr=pl.read_parquet('v8_lab/hunt/late/fr_addr/fr_city.parquet')
M={'nouvelle aquitaine':'NA','gironde':'NA','hauts de france':'HF','nord':'HF','pas de calais':'HF','pays de la loire':'PL','loire atlantique':'PL'}
CITY={'bordeaux':'NA','pessac':'NA','merignac':'NA','la teste de buch':'NA','lege cap ferret':'NA','nantes':'PL','saint nazaire':'PL','pornic':'PL','saint herblain':'PL','la baule escoublac':'PL','lille':'HF','tourcoing':'HF','dunkerque':'HF','roubaix':'HF','calais':'HF'}
def reg(a):
    if a is None: return None
    s=''.join(ch for ch in unicodedata.normalize('NFKD',a) if not unicodedata.combining(ch)).lower()
    r={M[x] for x in (re.sub(r'[\-\.]',' ',c).strip() for c in s.split(',')) if x in M}
    return r.pop() if len(r)==1 else None
u={a:reg(a) for col in ('aa','ab') for a in pr[col].unique().to_list()}
pr=pr.with_columns(ra=c.aa.replace_strict(u,default=None),rb=c.ab.replace_strict(u,default=None),rca=c.ca.replace_strict(CITY,default=None),rcb=c.cb.replace_strict(CITY,default=None))
pr=pr.with_columns(rA=pl.coalesce('ra','rca'),rB=pl.coalesce('rb','rcb'))
x=pr.filter(c.rA.is_not_null()&c.rB.is_not_null()&(c.rA!=c.rB))
print('region conflict pairs',x.height,'kept',x['kept'].sum())
# explicit region token conflict only (not city-derived)
y=pr.filter(c.ra.is_not_null()&c.rb.is_not_null()&(c.ra!=c.rb))
print('explicit region token conflict',y.height,'kept',y['kept'].sum())
# internal inconsistency on j side: its region token disagrees with its own city
z=pr.filter(c.rb.is_not_null()&c.rcb.is_not_null()&(c.rb!=c.rcb))
print('j self-inconsistent region vs city',z.height,'kept',z['kept'].sum())
pl.Config.set_fmt_str_lengths(60); pl.Config.set_tbl_rows(60); pl.Config.set_tbl_width_chars(250)
print(x.filter(c.kept).select('p4','na','aa','nb','ab').head(30))
print(z.filter(c.kept).select('p4','na','aa','nb','ab').head(20))
