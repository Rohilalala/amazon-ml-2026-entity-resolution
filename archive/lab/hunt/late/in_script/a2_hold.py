import polars as pl
from rapidfuzz import fuzz
R='/Users/adityarohilla/Documents/amazon-ml-2026/'
O=R+'v8_lab/hunt/late/in_script/'
def flags(path):
    return (pl.scan_parquet(path).filter(pl.col('country_key')=='india')
            .select('row','src','name','addr','name_non_latin','addr_non_latin','house').collect())
tr=flags(R+'cache/58f0a60c5c264fdd/recs.parquet')
h=pl.read_parquet(R+'v8_lab/hunt/sweep/audit/hold_keeps.parquet').filter(pl.col('country')=='india')
a=pl.read_parquet(R+'v8_lab/hunt/sweep/audit/hold_adds.parquet').filter(pl.col('add')).select('i','j',pl.lit(True).alias('addk'))
h=h.join(a,on=['i','j'],how='left').with_columns(kf=pl.col('keep')|pl.col('addk').fill_null(False))
# adds not in candidate set
extra=a.join(h.select('i','j'),on=['i','j'],how='anti'); print('adds outside cand',extra.height)
ti=tr.select(pl.col('row').alias('i'),pl.col('name').alias('ni'),pl.col('addr').alias('ai'),pl.col('house').alias('hi'))
tj=tr.select(pl.col('row').alias('j'),pl.col('src'),pl.col('name').alias('nj'),pl.col('addr').alias('aj'),pl.col('house').alias('hj'),pl.col('name_non_latin').alias('nnl'),pl.col('addr_non_latin').alias('anl'))
h=h.filter((pl.col('p')>0.02)|pl.col('y').cast(pl.Boolean)|pl.col('kf')).join(ti,on='i').join(tj,on='j')
h=h.with_columns(pl.struct('ni','nj').map_elements(lambda s: fuzz.token_sort_ratio(s['ni'] or '',s['nj'] or ''),return_dtype=pl.Float64).alias('nsim'),
                 pl.struct('ai','aj').map_elements(lambda s: fuzz.token_set_ratio(s['ai'] or '',s['aj'] or ''),return_dtype=pl.Float64).alias('asim'))
h.write_parquet(O+'hold_india.parquet')
cls=pl.when(pl.col('nnl')&pl.col('anl')).then(pl.lit('both_nat')).when(pl.col('nnl')).then(pl.lit('name_nat')).when(pl.col('anl')).then(pl.lit('addr_nat')).otherwise(pl.lit('latin'))
h=h.with_columns(cls.alias('cls'))
y=pl.col('y').cast(pl.Boolean)
print(h.group_by('cls').agg(pl.len().alias('n'),y.sum().alias('true'),pl.col('kf').sum().alias('kept'),(y&pl.col('kf')).sum().alias('TP'),
   (~y&pl.col('kf')).sum().alias('FP'),(y&~pl.col('kf')).sum().alias('FN'),pl.col('p').filter(~pl.col('kf')).sum().alias('p_rej'),pl.col('p').filter(pl.col('kf')).sum().alias('p_kept')).sort('cls'))
