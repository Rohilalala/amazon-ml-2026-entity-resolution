import polars as pl
R='/Users/adityarohilla/Documents/amazon-ml-2026/'; O=R+'v8_lab/hunt/late/in_script/'
cls=pl.when(pl.col('nnl')&pl.col('anl')).then(pl.lit('both_nat')).when(pl.col('nnl')).then(pl.lit('name_nat')).when(pl.col('anl')).then(pl.lit('addr_nat')).otherwise(pl.lit('latin'))
t=pl.read_parquet(O+'test_india.parquet').with_columns(cls.alias('cls'))
kc=t.filter('kf').group_by('i','src').len().rename({'len':'k_same_src'})
r=t.filter(~pl.col('kf'),pl.col('cls')=='addr_nat',pl.col('prob')>0.3).join(kc,on=['i','src'],how='left').with_columns(pl.col('k_same_src').fill_null(0))
print(r.group_by(pl.col('p4')>0.5,pl.col('k_same_src')>0).len())
tx=pl.scan_parquet(R+'v8_lab/ce/setup/out/text_test.parquet')
s=r.sample(40,seed=2)
kj=t.filter('kf').join(s.select('i','src'),on=['i','src']).group_by('i','src').agg(pl.col('j').first().alias('kj'))
s=s.join(kj,on=['i','src'],how='left')
rows=pl.concat([s['i'],s['j'],s['kj'].drop_nulls()]).unique()
tt=dict(tx.filter(pl.col('row').is_in(rows.implode())).collect().iter_rows())
for x in s.iter_rows(named=True):
    print(f"p4={x['p4']:.2f} v7={x['prob']:.2f} src{x['src']} ksame={x['k_same_src']} | {tt[x['i']][:80]}\n   CAND {tt[x['j']][:100]}\n   KEPT {tt.get(x['kj'],'-')[:100]}")
