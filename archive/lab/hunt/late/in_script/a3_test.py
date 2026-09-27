import polars as pl
R='/Users/adityarohilla/Documents/amazon-ml-2026/'
O=R+'v8_lab/hunt/late/in_script/'
te=(pl.scan_parquet(R+'cache/3ee42f05d06efbcb/recs.parquet').filter(pl.col('country_key')=='india')
    .select('row','entity_id','src','name','addr','name_non_latin','addr_non_latin','house').collect())
ids=pl.scan_parquet(R+'cache/3ee42f05d06efbcb/recs.parquet').select('row','entity_id').collect()
m=(pl.read_csv(R+'v8_lab/hunt/final16c/matching_results.tsv',separator='\t',quote_char=None,infer_schema=False)
   .with_columns(pl.col('matched_entity_ids').str.split(',')).explode('matched_entity_ids')
   .filter(pl.col('matched_entity_ids').is_not_null() & (pl.col('matched_entity_ids')!=''))
   .join(ids.rename({'row':'i','entity_id':'source1_entity_id'}),on='source1_entity_id')
   .join(ids.rename({'row':'j','entity_id':'matched_entity_ids'}),on='matched_entity_ids')
   .select('i','j',pl.lit(True).alias('kf')))
del ids
indi=te.filter(pl.col('src')==1).select(pl.col('row').alias('i'))
m=m.join(indi,on='i')
print('final india kept pairs',m.height)
tp=pl.scan_parquet(R+'v8_lab/ce/setup/out/test_pairs.parquet').filter(pl.col('country')=='india').select('i','j','prob').collect()
p4=pl.read_parquet(R+'v8_lab/hunt/plan/s4ce/out_adapt2/test_ACNRST_3s_adapt2_prob.parquet')
tp=tp.join(p4,on=['i','j'],how='left').join(m,on=['i','j'],how='full',coalesce=True).with_columns(pl.col('kf').fill_null(False))
print('kept not in cand',tp.filter(pl.col('prob').is_null()).height)
tp=tp.filter((pl.col('p4').fill_null(0)>0.02)|pl.col('kf'))
tj=te.select(pl.col('row').alias('j'),'src',pl.col('name_non_latin').alias('nnl'),pl.col('addr_non_latin').alias('anl'))
tp=tp.join(tj,on='j',how='left')
tp.write_parquet(O+'test_india.parquet')
cls=pl.when(pl.col('nnl')&pl.col('anl')).then(pl.lit('both_nat')).when(pl.col('nnl')).then(pl.lit('name_nat')).when(pl.col('anl')).then(pl.lit('addr_nat')).otherwise(pl.lit('latin'))
tp=tp.with_columns(cls.alias('cls'))
print(tp.group_by('cls').agg(pl.len().alias('n'),pl.col('kf').sum().alias('kept'),pl.col('p4').filter(pl.col('kf')).sum().alias('p_kept'),
   pl.col('p4').filter(~pl.col('kf')).sum().alias('p_rej'),(pl.col('kf')&(pl.col('p4').fill_null(0)<0.5)).sum().alias('kept_lowp'),
   (~pl.col('kf')&(pl.col('p4')>0.5)).sum().alias('rej_hip')).sort('cls'))
