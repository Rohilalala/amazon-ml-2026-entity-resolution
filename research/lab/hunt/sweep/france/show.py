import polars as pl, sys
O='/Users/adityarohilla/Documents/amazon-ml-2026/v8_lab/hunt/sweep/france/'
d=pl.read_parquet(O+'fr_ctx.parquet')
k=d.group_by('s1').agg(pl.col('kept').sum().alias('nk'))
mode,n,seed=sys.argv[1],int(sys.argv[2]),int(sys.argv[3])
if mode=='rand': ss=k.sample(n,seed=seed)['s1']
else: ss=k.filter(pl.col('nk')==int(mode)).sample(n,seed=seed)['s1']
for s in ss:
    x=d.filter(pl.col('s1')==s).sort('p4',descending=True,nulls_last=False)
    r=x.row(0,named=True); print(f"\n## {s} | {r['n1']} | {r['a1']}")
    x=x.filter(pl.col('kept')|(pl.col('p4')>0.02)).head(12)
    for r in x.iter_rows(named=True): print(f"  {'K' if r['kept'] else '.'} {r['p4'] if r['p4'] is None else round(r['p4'],3)} {r['m']} | {r['n2']} | {r['a2']}")
