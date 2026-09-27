# US analog of France region conflict: kept holdout pairs whose state differs
import polars as pl, re
D='student_resource/dataset/train/'
hk=pl.read_parquet('v8_lab/hunt/sweep/audit/hold_keeps.parquet').filter(pl.col('country')=='us')
rows=pl.concat([pl.read_csv(f'{D}train_source{k}.tsv',separator='\t',quote_char=None,infer_schema=False).select('business_address') for k in (1,2,3)]).with_row_index('row')
rows=rows.filter(pl.col('row').is_in(pl.concat([hk['i'],hk['j']]).unique().implode()))
S={'alabama':'al','alaska':'ak','arizona':'az','arkansas':'ar','california':'ca','colorado':'co','connecticut':'ct','delaware':'de','florida':'fl','georgia':'ga','hawaii':'hi','idaho':'id','illinois':'il','indiana':'in','iowa':'ia','kansas':'ks','kentucky':'ky','louisiana':'la','maine':'me','maryland':'md','massachusetts':'ma','michigan':'mi','minnesota':'mn','mississippi':'ms','missouri':'mo','montana':'mt','nebraska':'ne','nevada':'nv','new hampshire':'nh','new jersey':'nj','new mexico':'nm','new york':'ny','north carolina':'nc','north dakota':'nd','ohio':'oh','oklahoma':'ok','oregon':'or','pennsylvania':'pa','rhode island':'ri','south carolina':'sc','south dakota':'sd','tennessee':'tn','texas':'tx','utah':'ut','vermont':'vt','virginia':'va','washington':'wa','west virginia':'wv','wisconsin':'wi','wyoming':'wy','district of columbia':'dc'}
AB=set(S.values())
def st(a):
    if a is None: return None
    r={S.get(c,c) for c in (x.strip().lower() for x in a.split(',')) if c in S or c in AB}
    return r.pop() if len(r)==1 else None
P={r:st(a) for r,a in zip(rows['row'].to_list(),rows['business_address'].to_list())}
o=pl.DataFrame([(y,k,P.get(i),P.get(j)) for i,j,y,k in hk.select('i','j','y','keep').iter_rows()],schema=['y','keep','a','b'],orient='row')
o=o.filter(pl.col('a').is_not_null()&pl.col('b').is_not_null()&(pl.col('a')!=pl.col('b')))
print(o.group_by('keep').agg(n=pl.len(),pos=pl.col('y').sum()))
