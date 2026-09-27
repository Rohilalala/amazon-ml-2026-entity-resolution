# Recheck the 3 France region-conflict drops against raw test records and the base file
import polars as pl
c=pl.col
D='student_resource/dataset/test/'
d=pl.read_parquet('v8_lab/hunt/late/fr_addr/DROP_fr_region_conflict.parquet'); print(d)
fr=pl.concat([pl.read_csv(f'{D}test_source{k}.tsv',separator='\t',quote_char=None,infer_schema=False).filter(c.country.is_in(['France','FR'])) for k in (1,2,3)])
print('france rows',fr.height, fr['country'].unique().to_list())
fr.write_parquet('/tmp/fr_test.parquet')
base=pl.read_csv('v8_lab/hunt/final16c/matching_results.tsv',separator='\t',quote_char=None,infer_schema=False)
b=base.with_columns(m=c.matched_entity_ids.str.split(',')).explode('m').filter(c.m.is_not_null()&(c.m!='')).rename({'source1_entity_id':'s1'}).select('s1','m')
pl.Config.set_fmt_str_lengths(80); pl.Config.set_tbl_rows(80); pl.Config.set_tbl_width_chars(250)
for s1,m in d.iter_rows():
    print('=====',s1,m,'in base:',b.filter((c.s1==s1)&(c.m==m)).height)
    ids=set(b.filter(c.s1==s1)['m'].to_list())|{s1,m}|set(b.filter(c.m==m)['s1'].to_list())
    print(fr.filter(c.entity_id.is_in(list(ids))).select('entity_id','business_name','business_address'))
    print('base matches of s1:',b.filter(c.s1==s1)['m'].to_list())
