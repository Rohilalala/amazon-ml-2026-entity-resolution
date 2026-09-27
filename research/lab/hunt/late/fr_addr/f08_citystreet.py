# US analog for France kept pairs with different city AND different street name: false rate among holdout keeps
import polars as pl, re
from rapidfuzz import fuzz
D='student_resource/dataset/train/'
hk=pl.read_parquet('v8_lab/hunt/sweep/audit/hold_keeps.parquet').filter(pl.col('keep')&(pl.col('country')=='us'))
rows=pl.concat([pl.read_csv(f'{D}train_source{k}.tsv',separator='\t',quote_char=None,infer_schema=False).select('business_address') for k in (1,2,3)]).with_row_index('row')
rows=rows.filter(pl.col('row').is_in(pl.concat([hk['i'],hk['j']]).unique().implode()))
ST=re.compile(r'^(?:#\s*)?\d+\S*\s+(.*)$')
def p(a):
    if a is None: return (None,None)
    cs=[c.strip().lower() for c in a.split(',') if c.strip()]
    st=None; oth=[]
    for c in cs:
        m=ST.match(c)
        if st is None and m: st=re.sub(r'[^a-z ]','',m.group(1)).strip()
        elif not re.search(r'\d',c) and len(c)>2 and not re.match(r'^(unit|suite|ste|apt|fl|floor|po box|bldg)\b',c): oth.append(re.sub(r'^city of ','',c).replace(' city',''))
    return (st, oth[0] if oth else None)  # ponytail: first non-street text component = city (US S1 order: street, city, state)
P={r:p(a) for r,a in zip(rows['row'].to_list(),rows['business_address'].to_list())}
n=f=0; ex=[]
for i,j,y in hk.select('i','j','y').iter_rows():
    a,b=P.get(i),P.get(j)
    if not a or not b or None in a or None in b: continue
    if fuzz.token_set_ratio(a[0],b[0])<60 and fuzz.ratio(a[1],b[1])<70:
        n+=1; f+=(y==0)
print('US kept pairs with street<60 & city<70:',n,'false',f, 'false rate', f/max(n,1))
