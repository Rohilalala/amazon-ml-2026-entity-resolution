# Merge accepted TEST lists: drops = France region conflict (3); adds = none accepted
import polars as pl
S={'s1':pl.Utf8,'m':pl.Utf8}
d=pl.read_parquet('../fr_addr/DROP_fr_region_conflict.parquet').select('s1','m').cast(S).unique(maintain_order=True)
a=pl.DataFrame(schema=S)
assert d.join(a,on=['s1','m']).height==0
d.write_parquet('drops.parquet'); a.write_parquet('adds.parquet'); print('drops',d.height,'adds',a.height)
