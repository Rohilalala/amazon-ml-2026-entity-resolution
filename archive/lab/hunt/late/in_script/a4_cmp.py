import polars as pl
R='/Users/adityarohilla/Documents/amazon-ml-2026/'; O=R+'v8_lab/hunt/late/in_script/'
cls=pl.when(pl.col('nnl')&pl.col('anl')).then(pl.lit('both_nat')).when(pl.col('nnl')).then(pl.lit('name_nat')).when(pl.col('anl')).then(pl.lit('addr_nat')).otherwise(pl.lit('latin'))
hp=pl.scan_parquet(R+'v8_lab/ce/setup/out/holdout_pairs.parquet').filter(pl.col('country')=='india').select('i','j','prob').collect()
h=pl.read_parquet(O+'hold_india.parquet').join(hp,on=['i','j'],how='left').with_columns(cls.alias('cls'))
t=pl.read_parquet(O+'test_india.parquet').with_columns(cls.alias('cls'))
br=[0.02,0.1,0.3,0.5,0.7,0.9,0.98]
def b(df): return df.with_columns(pb=pl.col('prob').fill_null(-1).cut(br))
h=b(h); t=b(t)
y=pl.col('y').cast(pl.Boolean)
H=h.group_by('cls','pb').agg(pl.len().alias('hn'),y.mean().round(3).alias('h_true'),pl.col('kf').mean().round(3).alias('h_keep'))
T=t.group_by('cls','pb').agg(pl.len().alias('tn'),pl.col('kf').mean().round(3).alias('t_keep'),pl.col('p4').mean().round(3).alias('t_p4'))
pl.Config.set_tbl_rows(60)
print(H.join(T,on=['cls','pb'],how='full',coalesce=True).sort('cls','pb'))
