# DROP: France kept pairs whose two records sit in different regions (region token or city-implied region)
import polars as pl
exec(open('v8_lab/hunt/late/fr_addr/f09_region.py').read().split("x=pr.filter")[0])
d=pr.filter(c.kept&c.rA.is_not_null()&c.rB.is_not_null()&(c.rA!=c.rB)).select('i','j')
rows=pl.read_parquet('v8_lab/hunt/late/fr_addr/test_rows.parquet').select('row','entity_id')
d=d.join(rows.rename({'row':'i','entity_id':'s1'}),on='i').join(rows.rename({'row':'j','entity_id':'m'}),on='j')
base=pl.read_parquet('v8_lab/hunt/late/fr_addr/base_pairs.parquet')
assert d.join(base,on=['s1','m']).height==d.height
d.select('s1','m').write_parquet('v8_lab/hunt/late/fr_addr/DROP_fr_region_conflict.parquet')
pl.DataFrame({'s1':[],'m':[]},schema={'s1':pl.Utf8,'m':pl.Utf8}).write_parquet('v8_lab/hunt/late/fr_addr/ADD_none.parquet')
print(d.select('s1','m'))
