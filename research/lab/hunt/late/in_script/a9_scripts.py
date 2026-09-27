import polars as pl
R='/Users/adityarohilla/Documents/amazon-ml-2026/'; O=R+'v8_lab/hunt/late/in_script/'
S=[('deva','[ऀ-ॿ]'),('beng','[ঀ-৿]'),('guru','[਀-੿]'),('guj','[઀-૿]'),('oriya','[଀-୿]'),
   ('tamil','[஀-௿]'),('telugu','[ఀ-౿]'),('kann','[ಀ-೿]'),('mal','[ഀ-ൿ]')]
def sc(col):
    e=pl.lit('latin')
    for k,rx in reversed(S): e=pl.when(pl.col(col).str.contains(rx)).then(pl.lit(k)).otherwise(e)
    return e
def mix(col): return pl.col(col).str.split(' | ').list.first().str.count_matches('[a-z]{3,}')>0
nat=pl.col('nnl')|pl.col('anl')
y=pl.col('y').cast(pl.Boolean)
h=pl.read_parquet(O+'hold_india.parquet').filter(nat)
tx=pl.scan_parquet(R+'v8_lab/ce/setup/out/text_train.parquet').filter(pl.col('row').is_in(h['j'].unique().implode())).collect()
h=h.join(tx.rename({'row':'j'}),on='j').with_columns(sc('text').alias('scr'),(pl.col('nnl')&mix('text')).alias('mixname'))
H=h.group_by('scr').agg(pl.len().alias('hn'),y.mean().round(4).alias('h_true'),pl.col('kf').mean().round(4).alias('h_keep'),(~y&pl.col('kf')).sum().alias('FP'),(y&~pl.col('kf')).sum().alias('FN'))
t=pl.read_parquet(O+'test_india.parquet').filter(nat)
tx=pl.scan_parquet(R+'v8_lab/ce/setup/out/text_test.parquet').filter(pl.col('row').is_in(t['j'].unique().implode())).collect()
t=t.join(tx.rename({'row':'j'}),on='j').with_columns(sc('text').alias('scr'),(pl.col('nnl')&mix('text')).alias('mixname'))
T=t.group_by('scr').agg(pl.len().alias('tn'),pl.col('kf').mean().round(4).alias('t_keep'),pl.col('p4').mean().round(4).alias('t_p4'))
pl.Config.set_tbl_rows(40)
print(H.join(T,on='scr',how='full',coalesce=True).sort('scr'))
print(h.group_by('mixname').agg(pl.len(),y.mean().round(4),pl.col('kf').mean().round(4),(~y&pl.col('kf')).sum().alias('FP'),(y&~pl.col('kf')).sum().alias('FN')))
print(t.group_by('mixname').agg(pl.len(),pl.col('kf').mean().round(4)))
