"""What separates true gibberish renames from gibberish decoys (holdout truth), and does it carry to France."""
import polars as pl, sys
sys.path.insert(0,'/Users/adityarohilla/Documents/amazon-ml-2026/v8_lab/hunt/sweep/france'); from gib import gib
R='/Users/adityarohilla/Documents/amazon-ml-2026/'; O=R+'v8_lab/hunt/sweep/france/'
pl.Config.set_tbl_rows(60)
tr=pl.read_parquet(R+'cache/58f0a60c5c264fdd/recs.parquet',columns=['row','src','core'])
h=gib(pl.read_parquet(O+'ho_cats.parquet').filter(pl.col('active')&(pl.col('nm')=='none')&pl.col('ad').is_in(['aset','sameh']))).filter(pl.col('gib'))
cnt=tr.filter(pl.col('core').is_in(h['c2'].implode())).group_by('core').agg(pl.len().alias('ncore'))
h=h.join(tr.select(pl.col('row').alias('j'),'src'),on='j').join(cnt.rename({'core':'c2'}),on='c2',how='left').with_columns((pl.col('a1')==pl.col('a2')).alias('aexact'))
for k in ['src','aexact',pl.col('ncore').clip(1,3)]: print(h.group_by('ad',k).agg(pl.len(),pl.col('y').mean(),pl.col('kept').mean()).sort(pl.all()))
