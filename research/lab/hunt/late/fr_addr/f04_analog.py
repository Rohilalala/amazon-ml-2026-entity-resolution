# US/India analog: house-number letter suffix one-sided / conflicting, same number, same rest of street
import polars as pl, re
D='student_resource/dataset/train/'
rows=pl.concat([pl.read_csv(f'{D}train_source{k}.tsv',separator='\t',quote_char=None,infer_schema=False).select('business_name','business_address') for k in (1,2,3)]).with_row_index('row')
hk=pl.read_parquet('v8_lab/hunt/sweep/audit/hold_keeps.parquet')
need=pl.concat([hk['i'],hk['j']]).unique()
rows=rows.filter(pl.col('row').is_in(need.implode()))
R=re.compile(r'^(?:no\.?\s*|#\s*)?(\d+)\s*[-/]?\s*([a-z])?\b\s*(.*)$')
def p(a):
    if a is None: return (None,None,None)
    s=a.lower().split(',')
    for k,c in enumerate(s):
        m=R.match(c.strip())
        if m: return (m.group(1).lstrip('0') or '0',m.group(2),re.sub(r'[^a-z0-9]','',m.group(3)))
    return (None,None,None)
P={r:p(a) for r,a in zip(rows['row'].to_list(),rows['business_address'].to_list())}
N=dict(zip(rows['row'].to_list(),rows['business_name'].to_list()))
A=dict(zip(rows['row'].to_list(),rows['business_address'].to_list()))
out=[]
for i,j,y,co,pp,kp in hk.iter_rows():
    a,b=P.get(i,(None,)*3),P.get(j,(None,)*3)
    if a[0] is None or b[0] is None or a[0]!=b[0] or a[2]!=b[2] or not a[2]: continue
    if a[1]==b[1]: continue
    cls='one' if (a[1] is None or b[1] is None) else 'both_diff'
    out.append((i,j,y,co,pp,kp,cls,(N[i] or '').lower().strip()==(N[j] or '').lower().strip(),N[i],A[i],N[j],A[j]))
o=pl.DataFrame(out,schema=['i','j','y','country','p','keep','cls','samename','na','aa','nb','ab'],orient='row')
o.write_parquet('v8_lab/hunt/late/fr_addr/analog_suffix.parquet')
pl.Config.set_fmt_str_lengths(50); pl.Config.set_tbl_rows(60); pl.Config.set_tbl_width_chars(250)
print(o.group_by('country','cls','keep').agg(n=pl.len(),pos=pl.col('y').sum(),mp=pl.col('p').mean()).sort('country','cls','keep'))
print(o.filter(pl.col('keep')).sample(min(30,o['keep'].sum()),seed=0).select('y','p','na','aa','nb','ab'))
