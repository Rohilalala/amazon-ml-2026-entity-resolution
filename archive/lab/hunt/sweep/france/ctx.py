"""France S1 candidate context: every active candidate of a France S1 with p4 and the final16b keep flag -> fr_ctx.parquet"""
import polars as pl
R='/Users/adityarohilla/Documents/amazon-ml-2026/'; O=R+'v8_lab/hunt/sweep/france/'
recs=pl.read_parquet(R+'cache/3ee42f05d06efbcb/recs.parquet',columns=['row','entity_id','src','country_key','name','addr','house'])
fr=recs.filter(pl.col('country_key')=='france')
f1=fr.filter(pl.col('src')==1)
p4=pl.scan_parquet(R+'v8_lab/hunt/plan/s4ce/out_adapt2/test_ACNRST_3s_adapt2_prob.parquet').filter(pl.col('i').is_in(f1['row'].implode())).collect()
b=pl.read_csv(R+'v8_lab/hunt/final16b/matching_results.tsv',separator='\t',quote_char=None,infer_schema=False)
kp=b.with_columns(pl.col('matched_entity_ids').fill_null('').str.split(',')).explode('matched_entity_ids').rename({'source1_entity_id':'s1','matched_entity_ids':'m'}).filter(pl.col('m')!='')
kp=kp.join(recs.select(pl.col('entity_id').alias('s1'),pl.col('row').alias('i')),on='s1').join(recs.select(pl.col('entity_id').alias('m'),pl.col('row').alias('j')),on='m').select('i','j',pl.lit(True).alias('kept'))
kp=kp.filter(pl.col('i').is_in(f1['row'].implode()))
d=p4.join(kp,on=['i','j'],how='full',coalesce=True).with_columns(pl.col('kept').fill_null(False))
d=d.join(f1.select(pl.col('row').alias('i'),pl.col('entity_id').alias('s1'),pl.col('name').alias('n1'),pl.col('addr').alias('a1'),pl.col('house').alias('h1')),on='i')
d=d.join(fr.select(pl.col('row').alias('j'),pl.col('entity_id').alias('m'),pl.col('name').alias('n2'),pl.col('addr').alias('a2'),pl.col('house').alias('h2')),on='j',how='left')
print(d.height, d.filter(pl.col('p4').is_null()).height, d.filter(pl.col('m').is_null()).height, d['kept'].sum())
d.write_parquet(O+'fr_ctx.parquet')
