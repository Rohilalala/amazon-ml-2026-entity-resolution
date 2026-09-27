import polars as pl, sys
O='/Users/adityarohilla/Documents/amazon-ml-2026/v8_lab/hunt/sweep/france/'
pl.Config.set_tbl_rows(60); pl.Config.set_tbl_width_chars(250); pl.Config.set_fmt_str_lengths(60)
d=pl.read_parquet(O+'fr_ctx.parquet').join(pl.read_parquet(O+'fr_cats.parquet').select('i','j','nm','ad'),on=['i','j'])
nm,ad,kept,lo,hi,n=sys.argv[1],sys.argv[2],sys.argv[3]=='1',float(sys.argv[4]),float(sys.argv[5]),int(sys.argv[6])
x=d.filter((pl.col('nm')==nm)&(pl.col('ad')==ad)&(pl.col('kept')==kept)&pl.col('p4').is_between(lo,hi))
print(x.height); print(x.sample(min(n,x.height),seed=3).select(pl.col('p4').round(3),'n1','a1','n2','a2'))
