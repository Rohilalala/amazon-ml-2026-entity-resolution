import polars as pl, sys
sys.path.insert(0,'v8_lab/hunt/late/fr_addr'); from addr import parse
O='v8_lab/hunt/late/fr_addr/'
rows=pl.read_parquet(O+'test_rows.parquet')
fr=rows.filter(pl.col('country')=='France')
tp=pl.scan_parquet('v8_lab/ce/setup/out/test_pairs.parquet').filter((pl.col('country')=='france')&pl.col('active')).select('i','j','prob').collect()
p4=pl.scan_parquet('v8_lab/hunt/plan/s4ce/out_adapt2/test_ACNRST_3s_adapt2_prob.parquet').join(tp.lazy().select('i','j'),on=['i','j']).collect()
tp=tp.join(p4,on=['i','j'],how='left')
base=pl.read_parquet(O+'base_pairs.parquet').join(fr.select(pl.col('row').alias('i')),on='i').select('i','j').with_columns(pl.lit(True).alias('kept'))
pr=tp.join(base,on=['i','j'],how='full',coalesce=True).with_columns(pl.col('kept').fill_null(False))
print(pr.shape, pr['kept'].sum(), tp.shape)
P={r:parse(a) for r,a in zip(fr['row'].to_list(),fr['business_address'].to_list())}
def f(r):
    x=P.get(r); return x if x else (None,None,None,None,())
cols={}
for side,c in (('i','a'),('j','b')):
    v=[f(r) for r in pr[side].to_list()]
    cols[c+'_hn']=[x[0] for x in v]; cols[c+'_suf']=[x[1] for x in v]; cols[c+'_st']=[x[2] for x in v]
    cols[c+'_nm']=[' '.join(x[3]) if x[3] else None for x in v]; cols[c+'_rest']=['|'.join(x[4]) for x in v]
pr=pr.with_columns([pl.Series(k,v) for k,v in cols.items()])
nm=rows.select(pl.col('row'),pl.col('business_name'),pl.col('business_address'))
pr=pr.join(nm.rename({'row':'i','business_name':'na','business_address':'aa'}),on='i').join(nm.rename({'row':'j','business_name':'nb','business_address':'ab'}),on='j')
pr.write_parquet(O+'fr_pairs.parquet'); print(pr.shape)
