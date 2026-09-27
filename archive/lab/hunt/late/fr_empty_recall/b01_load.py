import polars as pl
B='/Users/adityarohilla/Documents/amazon-ml-2026/'
D=B+'v8_lab/hunt/late/fr_empty_recall/'
rd=lambda f: pl.read_csv(f,separator='\t',quote_char=None,infer_schema=False)
src=[rd(B+f'student_resource/dataset/test/test_source{k}.tsv') for k in (1,2,3)]
ids=pl.concat([s.select('entity_id','business_name','business_address','country') for s in src]).with_row_index('row')
ids.write_parquet(D+'ids.parquet')
t=pl.read_parquet(B+'v8_lab/ce/setup/out/text_test.parquet')
print(ids.shape,t.shape); print(ids.head(3)); print(t.filter(pl.col('row').is_in([0,len(src[0]),len(src[0])+len(src[1])])))
print(ids.filter(pl.col('row').is_in([0,len(src[0]),len(src[0])+len(src[1])])))
def pairs(f):
  m=rd(f).filter(pl.col('matched_entity_ids').is_not_null() & (pl.col('matched_entity_ids')!=''))
  return m.with_columns(pl.col('matched_entity_ids').str.split(',')).explode('matched_entity_ids').rename({'source1_entity_id':'s1','matched_entity_ids':'m'})
best=pairs(B+'v8_lab/hunt/final16c/matching_results.tsv'); best.write_parquet(D+'best_pairs.parquet')
al=pairs(B+'v8_lab/recall/build/ALL/matching_results.tsv'); a2=pairs(B+'v8_lab/ce/v2_build/A_all_v2/matching_results.tsv')
fr=set(src[0].filter(pl.col('country')=='FR')['entity_id']) if 'FR' in set(src[0]['country']) else None
print(src[0]['country'].value_counts())
radd=al.join(a2,on=['s1','m'],how='anti'); radd.write_parquet(D+'recall_adds_all.parquet')
print('best',best.shape,'ALL',al.shape,'A2',a2.shape,'radd',radd.shape)
