import polars as pl, numpy as np
R='/Users/adityarohilla/Documents/amazon-ml-2026/'; O=R+'v8_lab/hunt/late/in_script/'
cls=pl.when(pl.col('nnl')&pl.col('anl')).then(pl.lit('both_nat')).when(pl.col('nnl')).then(pl.lit('name_nat')).when(pl.col('anl')).then(pl.lit('addr_nat')).otherwise(pl.lit('latin'))
h=pl.read_parquet(O+'hold_india.parquet').with_columns(cls.alias('cls'),nat=pl.col('nnl')|pl.col('anl'))
hp=pl.scan_parquet(R+'v8_lab/ce/setup/out/holdout_pairs.parquet').filter(pl.col('country')=='india').select('i','j','prob').collect()
h=h.join(hp,on=['i','j'],how='left')
y=pl.col('y').cast(pl.Boolean)
# calibration of stacker p by nat
print(h.filter(pl.col('p').is_between(0.05,0.95)).with_columns(pb=pl.col('p').cut([0.2,0.4,0.6,0.8])).group_by('nat','pb')
      .agg(pl.len(),pl.col('p').mean().round(3).alias('p'),y.mean().round(3).alias('true'),pl.col('kf').mean().round(3).alias('keep')).sort('nat','pb'))
# baseline per S1
s1=pl.read_parquet(R+'v8_lab/ce/setup/out/holdout_s1.parquet')
full=pl.read_parquet(R+'v8_lab/hunt/sweep/audit/hold_keeps.parquet').filter(pl.col('country')=='india')
a=pl.read_parquet(R+'v8_lab/hunt/sweep/audit/hold_adds.parquet').filter(pl.col('add')).select('i','j','y')
ks=pl.concat([full.filter('keep').select('i','j','y'),a]).unique(['i','j'])
base=ks.group_by('i').agg(pl.len().alias('K'),pl.col('y').cast(pl.Int64).sum().alias('TP'))
def f(tp,k,nt):
    fp=k-tp; fn=nt-tp; d=1.25*tp+0.25*fn+fp
    return np.where(d==0,1.0,1.25*tp/np.maximum(d,1e-9))
N=s1.height
def evalrule(mask,name):
    ch=h.filter(mask & ~pl.col('kf'))   # adds only
    if ch.height==0: print(name,'none'); return
    d=ch.group_by('i').agg(pl.len().alias('dk'),pl.col('y').cast(pl.Int64).sum().alias('dtp'))
    t=d.join(base,on='i',how='left').join(s1.select('i','n_true'),on='i').fill_null(0)
    f0=f(t['TP'].to_numpy(),t['K'].to_numpy(),t['n_true'].to_numpy()); f1=f(t['TP'].to_numpy()+t['dtp'].to_numpy(),t['K'].to_numpy()+t['dk'].to_numpy(),t['n_true'].to_numpy())
    dd=f1-f0; rng=np.random.default_rng(0)
    bs=[dd[rng.integers(0,len(dd),len(dd))].sum()/N for _ in range(2000)]
    print(f"{name}: adds={ch.height} prec={ch['y'].mean():.3f} S1={len(dd)} dF={dd.sum()/N:+.6f} CI=({np.percentile(bs,2.5):+.6f},{np.percentile(bs,97.5):+.6f})")
for c in ['addr_nat','name_nat','both_nat']:
  for t in [0.2,0.3,0.4,0.5]:
    evalrule((pl.col('cls')==c)&(pl.col('p')>t),f'{c} p>{t}')
for t in [0.3,0.5,0.7]:
    evalrule(pl.col('nat')&(pl.col('prob')>t),f'nat v7prob>{t}')
