import polars as pl,sys
B='/Users/adityarohilla/Documents/amazon-ml-2026/'
D=B+'v8_lab/hunt/late/fr_empty_recall/'
ids=pl.read_parquet(D+'ids.parquet'); tp=pl.read_parquet(D+'empty_cands.parquet')
top=tp.sort('prob',descending=True).group_by('i',maintain_order=True).head(1)
mode=sys.argv[1]
if mode=='rand': sel=top.sample(40,seed=1)['i']
else: sel=top.filter((pl.col('prob')>0.3)&pl.col('owner').is_null()).sample(40,seed=2)['i']
nm=lambda r: ids[int(r)]
def s(r):
  x=ids.row(int(r)); return f"{x[1]} | {x[2]} | {x[3]}"
for i in sel:
  print('S1',s(i))
  c=tp.filter(pl.col('i')==i).sort('prob',descending=True).head(3)
  for row in c.iter_rows(named=True):
    own='' if row['owner'] is None else ' OWN='+ids.row(int(row['owner']))[1]
    p4='-' if row['p4'] is None else f"{row['p4']:.3f}"
    print(f"   p={row['prob']:.3f} p4={p4} {s(row['j'])}{own}")
