"""Keep-count profile per S1 by country (final16b) vs train truth (US/India)."""
import polars as pl
R='/Users/adityarohilla/Documents/amazon-ml-2026/'
recs=pl.read_parquet(R+'cache/3ee42f05d06efbcb/recs.parquet',columns=['entity_id','src','country_key'])
b=pl.read_csv(R+'v8_lab/hunt/final16b/matching_results.tsv',separator='\t',quote_char=None,infer_schema=False)
p=b.with_columns(pl.col('matched_entity_ids').fill_null('').str.split(',')).explode('matched_entity_ids').rename({'source1_entity_id':'s1','matched_entity_ids':'m'})
p=p.with_columns((pl.col('m')!='').alias('v'),pl.col('m').str.slice(0,2).alias('ms'))
s1c=recs.filter(pl.col('src')==1).select(pl.col('entity_id').alias('s1'),'country_key')
g=p.group_by('s1').agg(pl.col('v').sum().alias('n'),((pl.col('ms')=='S2')&pl.col('v')).sum().alias('n2'),((pl.col('ms')=='S3')&pl.col('v')).sum().alias('n3')).join(s1c,on='s1')
print(g.group_by('country_key','n').len().sort('country_key','n').with_columns((pl.col('len')/pl.col('len').sum().over('country_key')).round(4).alias('frac')))
print(g.group_by('country_key','n2','n3').len().with_columns((pl.col('len')/pl.col('len').sum().over('country_key')).round(4).alias('frac')).filter(pl.col('frac')>0.01).sort('country_key','len',descending=[False,True]))
print(g.group_by('country_key').agg(pl.col('n').mean(),pl.col('n2').mean(),pl.col('n3').mean()))
pl.Config.set_tbl_rows(80)
t=g.group_by('country_key','n').len().with_columns((pl.col('len')/pl.col('len').sum().over('country_key')).round(4).alias('frac')).pivot(on='country_key',index='n',values='frac').sort('n')
T=R+'student_resource/dataset/train/'
gt=pl.read_csv(T+'train_ground_truth.tsv',separator='\t',quote_char=None,infer_schema=False)
s1t=pl.read_csv(T+'train_source1.tsv',separator='\t',quote_char=None,infer_schema=False).select(pl.col('entity_id').str.strip_chars().alias('s1'),pl.col('country').str.strip_chars().str.to_lowercase().alias('c'))
q=gt.rename({'source1_entity_id':'s1','matched_entity_ids':'m'}).with_columns(pl.col('s1').str.strip_chars(),pl.col('m').fill_null('').str.split(',').list.eval(pl.element().filter(pl.element()!='')).list.len().alias('n')).join(s1t,on='s1',how='right')
q=q.with_columns(pl.col('n').fill_null(0))
tt=q.group_by('c','n').len().with_columns((pl.col('len')/pl.col('len').sum().over('c')).round(4).alias('frac')).pivot(on='c',index='n',values='frac').sort('n')
print(t.join(tt.rename({c:'TRUTH_'+c for c in tt.columns if c!='n'}),on='n',how='full',coalesce=True).sort('n'))
print(q.group_by('c').agg(pl.col('n').mean(),pl.len()))
