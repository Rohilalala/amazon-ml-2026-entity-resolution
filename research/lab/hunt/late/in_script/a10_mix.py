import polars as pl, numpy as np
R='/Users/adityarohilla/Documents/amazon-ml-2026/'; O=R+'v8_lab/hunt/late/in_script/'
exec(open(O+'a9_scripts.py').read().split('nat=pl.col')[0])  # S, sc, mix
nat=pl.col('nnl')|pl.col('anl'); y=pl.col('y').cast(pl.Boolean)
h=pl.read_parquet(O+'hold_india.parquet').filter(nat)
tx=pl.scan_parquet(R+'v8_lab/ce/setup/out/text_train.parquet').filter(pl.col('row').is_in(h['j'].unique().implode())).collect()
h=h.join(tx.rename({'row':'j'}),on='j').with_columns((pl.col('nnl')&mix('text')).alias('mixname'))
m=h.filter('mixname')
print(m.with_columns(pb=pl.col('p').cut([0.02,0.1,0.3,0.5,0.7,0.9])).group_by('pb').agg(pl.len(),y.mean().round(3),pl.col('kf').mean().round(3)).sort('pb'))
tx2=pl.scan_parquet(R+'v8_lab/ce/setup/out/text_train.parquet').filter(pl.col('row').is_in(m['i'].unique().implode())).collect()
m=m.join(tx2.rename({'row':'i','text':'ti'}),on='i')
for r in m.filter(~pl.col('kf')).sort('p',descending=True).head(40).iter_rows(named=True):
    print(f"y={r['y']} p={r['p']:.2f} nsim={r['nsim']:.0f} | {r['ti'][:70]} || {r['text'][:90]}")
