import polars as pl
from rapidfuzz import fuzz
R='/Users/adityarohilla/Documents/amazon-ml-2026/'; O=R+'v8_lab/hunt/late/in_script/'
cls=pl.when(pl.col('nnl')&pl.col('anl')).then(pl.lit('both_nat')).when(pl.col('nnl')).then(pl.lit('name_nat')).when(pl.col('anl')).then(pl.lit('addr_nat')).otherwise(pl.lit('latin'))
tr=(pl.scan_parquet(R+'cache/58f0a60c5c264fdd/recs.parquet').select('row','country_key','name','name_non_latin','addr_non_latin').collect())
a=pl.read_parquet(R+'v8_lab/hunt/sweep/audit/hold_adds.parquet').filter(pl.col('add'))
a=a.join(tr.select(pl.col('row').alias('i'),'country_key',pl.col('name').alias('ni')),on='i').join(tr.select(pl.col('row').alias('j'),pl.col('name').alias('nj'),pl.col('name_non_latin').alias('nnl'),pl.col('addr_non_latin').alias('anl')),on='j')
del tr
a=a.with_columns(cls.alias('cls'),nsim=pl.struct('ni','nj').map_elements(lambda s: fuzz.token_sort_ratio(s['ni'],s['nj']),return_dtype=pl.Float64))
print(a.group_by('country_key','cls').agg(pl.len(),pl.col('y').mean().round(3)).sort('country_key','cls'))
print(a.filter(pl.col('country_key')=='india').group_by((pl.col('nsim')//20*20).alias('nb')).agg(pl.len(),pl.col('y').mean().round(3)).sort('nb'))
t=pl.read_parquet(O+'test_india.parquet').filter(pl.col('kf'),pl.col('prob').is_null()).with_columns(cls.alias('cls'))
print('test not-in-cand India by class'); print(t.group_by('cls').len().sort('cls'))
