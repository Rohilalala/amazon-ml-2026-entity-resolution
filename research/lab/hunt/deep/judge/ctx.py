import polars as pl
from pairs import pairs
R='/Users/adityarohilla/Documents/amazon-ml-2026/'
H=R+'v8_lab/hunt/'
recs=pl.read_parquet(R+'cache/3ee42f05d06efbcb/recs.parquet',columns=['row','entity_id'])
tt=pl.read_parquet(R+'v8_lab/ce/setup/out/text_test.parquet')
tp=pl.read_parquet(R+'v8_lab/ce/setup/out/test_pairs.parquet',columns=['i','j','prob','active'])
pa=tp.filter(pl.col('active')).with_row_index('k')
p4=pl.read_parquet(H+'plan/s4ce/out_adapt/test_ACNRST_3s_adapt_prob.parquet')
def enrich(d):
    d=d.join(recs.rename({'row':'i','entity_id':'s1'}),on='s1').join(recs.rename({'row':'j','entity_id':'m'}),on='m')
    d=d.join(tt.rename({'row':'i','text':'t1'}),on='i').join(tt.rename({'row':'j','text':'t2'}),on='j')
    d=d.join(tp.select('i','j','prob'),on=['i','j'],how='left').join(p4.select('i','j','p4'),on=['i','j'],how='left')
    return d
