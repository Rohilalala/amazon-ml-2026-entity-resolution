import polars as pl
D='student_resource/dataset/test/'
fr=[pl.read_csv(f'{D}test_source{k}.tsv',separator='\t',quote_char=None,infer_schema=False) for k in (1,2,3)]
allr=pl.concat(fr).with_row_index('row')
allr.write_parquet('v8_lab/hunt/late/fr_addr/test_rows.parquet')
m=pl.read_csv('v8_lab/hunt/final16c/matching_results.tsv',separator='\t',infer_schema=False)
m=m.with_columns(pl.col('matched_entity_ids').str.split(',')).explode('matched_entity_ids').rename({'source1_entity_id':'s1','matched_entity_ids':'m'}).filter(pl.col('m').is_not_null() & (pl.col('m')!=''))
idx=allr.select('row','entity_id')
m=m.join(idx.rename({'row':'i','entity_id':'s1'}),on='s1').join(idx.rename({'row':'j','entity_id':'m'}),on='m')
m.write_parquet('v8_lab/hunt/late/fr_addr/base_pairs.parquet')
print(m.shape)
