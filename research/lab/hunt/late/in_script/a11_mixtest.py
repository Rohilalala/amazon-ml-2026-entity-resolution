import polars as pl
R='/Users/adityarohilla/Documents/amazon-ml-2026/'; O=R+'v8_lab/hunt/late/in_script/'
exec(open(O+'a9_scripts.py').read().split('nat=pl.col')[0])
nat=pl.col('nnl')|pl.col('anl')
t=pl.read_parquet(O+'test_india.parquet').filter(pl.col('nnl'))
tx=pl.scan_parquet(R+'v8_lab/ce/setup/out/text_test.parquet').filter(pl.col('row').is_in(t['j'].unique().implode())).collect()
t=t.join(tx.rename({'row':'j'}),on='j').with_columns(mix('text').alias('mixname')).filter('mixname')
print(t.with_columns(pb=pl.col('p4').fill_null(-1).cut([-0.5,0.02,0.1,0.3,0.5,0.7,0.9])).group_by('pb').agg(pl.len(),pl.col('kf').mean().round(3),pl.col('p4').mean().round(3)).sort('pb'))
# test mix kept share vs holdout: fraction with empty-ish address
t=t.with_columns(short=pl.col('text').str.split(' | ').list.get(1,null_on_oob=True).fill_null('').str.len_chars()<8)
print(t.group_by('short').agg(pl.len(),pl.col('kf').mean().round(4)))
h=pl.read_parquet(O+'hold_india.parquet').filter(pl.col('nnl'))
tx=pl.scan_parquet(R+'v8_lab/ce/setup/out/text_train.parquet').filter(pl.col('row').is_in(h['j'].unique().implode())).collect()
h=h.join(tx.rename({'row':'j'}),on='j').with_columns(mix('text').alias('mixname')).filter('mixname')
h=h.with_columns(short=pl.col('text').str.split(' | ').list.get(1,null_on_oob=True).fill_null('').str.len_chars()<8)
print(h.group_by('short').agg(pl.len(),pl.col('kf').mean().round(4),pl.col('y').mean().round(4)))
