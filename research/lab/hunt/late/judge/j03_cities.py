# Coverage check: France candidate-pair cities not in the 15-city region map, and kept pairs whose cities differ where a region is unknown
import polars as pl
c=pl.col
pr=pl.read_parquet('v8_lab/hunt/late/fr_addr/fr_city.parquet')
print(pr.columns, pr.height)
CITY={'bordeaux','pessac','merignac','la teste de buch','lege cap ferret','nantes','saint nazaire','pornic','saint herblain','la baule escoublac','lille','tourcoing','dunkerque','roubaix','calais'}
v=pl.concat([pr.select(x=c.ca),pr.select(x=c.cb)]).group_by('x').len().sort('len',descending=True)
print(v.filter(~c.x.is_in(list(CITY))).head(20)); print('mapped share',v.filter(c.x.is_in(list(CITY)))['len'].sum()/v['len'].sum())
