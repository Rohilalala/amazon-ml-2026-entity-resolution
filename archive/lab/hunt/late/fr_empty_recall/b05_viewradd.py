import polars as pl,sys
B='/Users/adityarohilla/Documents/amazon-ml-2026/'
D=B+'v8_lab/hunt/late/fr_empty_recall/'
ids=pl.read_parquet(D+'ids.parquet'); best=pl.read_parquet(D+'best_pairs.parquet')
fr=pl.read_parquet(D+'fr_radd.parquet').filter(pl.col('inbest')==1)
def s(r):
  x=ids.row(int(r)); return f"{x[1]} | {x[2]}"
e2r=dict(zip(ids.filter(pl.col('entity_id').is_in(best.filter(pl.col('s1').is_in(fr['s1'].implode()))['m'].implode()))['entity_id'],ids.filter(pl.col('entity_id').is_in(best.filter(pl.col('s1').is_in(fr['s1'].implode()))['m'].implode()))['row']))
sub=fr.sample(int(sys.argv[2]),seed=int(sys.argv[1]))
for row in sub.iter_rows(named=True):
  print('S1',s(row['i']))
  print('  ADD',row['m'][:2],s(row['j']))
  for m in best.filter((pl.col('s1')==row['s1'])&(pl.col('m')!=row['m']))['m']:
    print('  kept',m[:2],s(e2r[m]))
