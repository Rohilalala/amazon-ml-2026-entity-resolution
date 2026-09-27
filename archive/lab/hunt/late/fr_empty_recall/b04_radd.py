import polars as pl,sys
B='/Users/adityarohilla/Documents/amazon-ml-2026/'
D=B+'v8_lab/hunt/late/fr_empty_recall/'
ids=pl.read_parquet(D+'ids.parquet'); best=pl.read_parquet(D+'best_pairs.parquet')
e=ids.select('entity_id','row','country')
r=pl.read_parquet(D+'recall_adds_all.parquet').join(e.rename({'entity_id':'s1','row':'i'}),on='s1').join(e.select(pl.col('entity_id').alias('m'),pl.col('row').alias('j')),on='m')
print(r['country'].value_counts())
fr=r.filter(pl.col('country')=='France')
fr=fr.join(best.with_columns(pl.lit(1).alias('inbest')),on=['s1','m'],how='left')
print('France recall adds',fr.height,'still in best',fr['inbest'].sum())
p4=pl.scan_parquet(B+'v8_lab/hunt/plan/s4ce/out_adapt2/test_ACNRST_3s_adapt2_prob.parquet').filter(pl.col('i').is_in(fr['i'].unique().implode())).collect()
fr=fr.join(p4,on=['i','j'],how='left')
tp=pl.scan_parquet(B+'v8_lab/ce/setup/out/test_pairs.parquet').filter(pl.col('i').is_in(fr['i'].unique().implode())).select('i','j','prob').collect()
fr=fr.join(tp,on=['i','j'],how='left')
fr.write_parquet(D+'fr_radd.parquet')
print(fr.select(pl.col('p4').is_null().sum(),pl.col('prob').is_null().sum()))
