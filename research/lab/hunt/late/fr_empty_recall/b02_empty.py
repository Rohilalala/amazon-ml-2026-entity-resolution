import polars as pl
B='/Users/adityarohilla/Documents/amazon-ml-2026/'
D=B+'v8_lab/hunt/late/fr_empty_recall/'
ids=pl.read_parquet(D+'ids.parquet'); best=pl.read_parquet(D+'best_pairs.parquet')
e=ids.select('entity_id','row')
bj=best.join(e.rename({'entity_id':'m','row':'j'}),on='m').join(e.rename({'entity_id':'s1','row':'owner'}),on='s1')
bj.select('owner','j').write_parquet(D+'best_rows.parquet')
s1fr=ids.filter((pl.col('entity_id').str.starts_with('S1-'))&(pl.col('country')=='France'))
emp=s1fr.join(best.select(pl.col('s1').alias('entity_id')).unique(),on='entity_id',how='anti')
print('France S1',s1fr.height,'empty',emp.height,flush=True)
tp=pl.scan_parquet(B+'v8_lab/ce/setup/out/test_pairs.parquet').filter(pl.col('i').is_in(emp['row'].implode())).collect()
p4=pl.scan_parquet(B+'v8_lab/hunt/plan/s4ce/out_adapt2/test_ACNRST_3s_adapt2_prob.parquet').filter(pl.col('i').is_in(emp['row'].implode())).collect()
tp=tp.join(p4,on=['i','j'],how='left')
claim=bj.group_by('j').agg(pl.col('owner').first(),pl.len().alias('nown'))
tp=tp.join(claim,on='j',how='left')
tp.write_parquet(D+'empty_cands.parquet')
print('empties with any cand',tp['i'].n_unique(),'cand rows',tp.height)
top=tp.sort('prob',descending=True).group_by('i',maintain_order=True).head(1)
for thr in [0.05,0.1,0.3,0.5,0.7]:
  x=top.filter(pl.col('prob')>thr); print(thr,x.height,'unclaimed',x.filter(pl.col('owner').is_null()).height)
for thr in [0.1,0.3,0.5]:
  x=top.filter(pl.col('p4')>thr); print('p4',thr,x.height,'unclaimed',x.filter(pl.col('owner').is_null()).height)
