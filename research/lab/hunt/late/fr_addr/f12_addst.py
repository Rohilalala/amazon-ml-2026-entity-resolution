# non-model France adds (p4 null) with different house number AND different street name; US/India analog on holdout recall adds
import polars as pl, re
from rapidfuzz import fuzz
c=pl.col
pr=pl.read_parquet('v8_lab/hunt/late/fr_addr/fr_city.parquet')
x=pr.filter(c.kept&c.p4.is_null()&(c.hn=='diff')&c.a_nm.is_not_null() if 'a_nm' in pr.columns else c.kept&c.p4.is_null()&(c.hn=='diff'))
cl=pl.read_parquet('v8_lab/hunt/late/fr_addr/fr_cls.parquet').select('i','j','a_nm','b_nm')
x=x.join(cl,on=['i','j'])
x=x.with_columns(ss=pl.Series([fuzz.token_set_ratio(a or '',b or '') for a,b in zip(x['a_nm'],x['b_nm'])]))
x=x.filter(c.ss<70)
print('FR non-model adds, hn diff & street sim<70:',x.height)
x.write_parquet('v8_lab/hunt/late/fr_addr/fr_nonmodel_diffstreet.parquet')
# analog
D='student_resource/dataset/train/'
ha=pl.read_parquet('v8_lab/hunt/sweep/audit/hold_adds.parquet').filter(c.add)
rows=pl.concat([pl.read_csv(f'{D}train_source{k}.tsv',separator='\t',quote_char=None,infer_schema=False).select('business_address') for k in (1,2,3)]).with_row_index('row')
rows=rows.filter(c.row.is_in(pl.concat([ha['i'],ha['j']]).unique().implode()))
R=re.compile(r'^(?:no\.?\s*|#\s*)?(\d+)\s*[-/]?\s*([a-z])?\b\s*(.*)$')
def p(a):
    if a is None: return (None,None)
    for cc in a.lower().split(','):
        m=R.match(cc.strip())
        if m: return (m.group(1).lstrip('0'),re.sub(r'[^a-z ]','',m.group(3)).strip())
    return (None,None)
P={r:p(a) for r,a in zip(rows['row'].to_list(),rows['business_address'].to_list())}
n=f=0
for i,j,y in ha.select('i','j','y').iter_rows():
    a,b=P.get(i,(None,None)),P.get(j,(None,None))
    if a[0] and b[0] and a[0]!=b[0] and a[1] and b[1] and fuzz.token_set_ratio(a[1],b[1])<70:
        n+=1; f+=(y==0)
print('holdout adds total',ha.height,'pos',ha['y'].sum(),'| analog hn diff & street<70:',n,'false',f)
