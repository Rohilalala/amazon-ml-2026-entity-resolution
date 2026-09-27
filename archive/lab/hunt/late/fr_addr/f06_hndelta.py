# US/India analog: same street, different house number; label rate by numeric relation
import polars as pl, re
from rapidfuzz import fuzz
D='student_resource/dataset/train/'
rows=pl.concat([pl.read_csv(f'{D}train_source{k}.tsv',separator='\t',quote_char=None,infer_schema=False).select('business_name','business_address') for k in (1,2,3)]).with_row_index('row')
hk=pl.read_parquet('v8_lab/hunt/sweep/audit/hold_keeps.parquet')
rows=rows.filter(pl.col('row').is_in(pl.concat([hk['i'],hk['j']]).unique().implode()))
R=re.compile(r'^(?:no\.?\s*|#\s*)?(\d+)\s*[-/]?\s*([a-z])?\b\s*(.*)$')
def p(a):
    if a is None: return (None,None)
    for c in a.lower().split(','):
        m=R.match(c.strip())
        if m: return (m.group(1),re.sub(r'[^a-z0-9]','',m.group(3)))
    return (None,None)
P={r:p(a) for r,a in zip(rows['row'].to_list(),rows['business_address'].to_list())}
N=dict(zip(rows['row'].to_list(),rows['business_name'].to_list()))
def rel(a,b):
    ia,ib=int(a),int(b)
    if ia==ib: return 'zeropad'
    if a.lstrip('0') in b or b.lstrip('0') in a: return 'prefix/ins'
    d=abs(ia-ib)
    if len(a)==len(b) and sum(x!=y for x,y in zip(a,b))==1: return 'sub1'
    return 'near<=20' if d<=20 else 'far'
out=[]
for i,j,y,co,pp,kp in hk.iter_rows():
    a,b=P.get(i,(None,None)),P.get(j,(None,None))
    if a[0] is None or b[0] is None or a[0]==b[0] or a[1]!=b[1] or not a[1]: continue
    out.append((co,y,kp,pp,rel(a[0],b[0]),fuzz.token_set_ratio((N[i] or '').lower(),(N[j] or '').lower())>=90))
o=pl.DataFrame(out,schema=['country','y','keep','p','rel','ns90'],orient='row')
pl.Config.set_tbl_rows(60)
print(o.group_by('country','rel','ns90','keep').agg(n=pl.len(),pos=pl.col('y').sum()).with_columns(prec=pl.col('pos')/pl.col('n')).sort('country','rel','ns90','keep'))
