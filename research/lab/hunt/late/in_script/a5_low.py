import polars as pl
R='/Users/adityarohilla/Documents/amazon-ml-2026/'; O=R+'v8_lab/hunt/late/in_script/'
cls=pl.when(pl.col('nnl')&pl.col('anl')).then(pl.lit('both_nat')).when(pl.col('nnl')).then(pl.lit('name_nat')).when(pl.col('anl')).then(pl.lit('addr_nat')).otherwise(pl.lit('latin'))
t=pl.read_parquet(O+'test_india.parquet').with_columns(cls.alias('cls'))
low=t.filter(pl.col('kf'),pl.col('prob').fill_null(-1)<=0.02)
print(low.group_by('cls',pl.col('prob').is_null().alias('pnull'),pl.col('p4').is_null().alias('p4null')).len().sort('cls'))
# holdout same
h=pl.read_parquet(O+'hold_india.parquet')
hp=pl.scan_parquet(R+'v8_lab/ce/setup/out/holdout_pairs.parquet').filter(pl.col('country')=='india').select('i','j','prob').collect()
h=h.join(hp,on=['i','j'],how='left').with_columns(cls.alias('cls'))
hl=h.filter(pl.col('kf'),pl.col('prob').fill_null(-1)<=0.02)
print(hl.group_by('cls').agg(pl.len(),pl.col('y').mean()))
tx=pl.scan_parquet(R+'v8_lab/ce/setup/out/text_test.parquet')
s=low.filter(pl.col('cls')!='latin').sample(40,seed=1)
rows=pl.concat([s['i'],s['j']]).unique()
tt=tx.filter(pl.col('row').is_in(rows.implode())).collect()
s=s.join(tt.rename({'row':'i','text':'ti'}),on='i').join(tt.rename({'row':'j','text':'tj'}),on='j')
for r in s.iter_rows(named=True):
    print(r['cls'],r['p4'],'|',r['ti'][:90],'||',r['tj'][:110])
