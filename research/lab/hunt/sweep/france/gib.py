"""Gibberish-rename candidates at the S1's own address (single invented token, no S1 token inside): holdout truth vs France keeps."""
import polars as pl
O='/Users/adityarohilla/Documents/amazon-ml-2026/v8_lab/hunt/sweep/france/'
def gib(d):
    return d.with_columns(((pl.col('c2').str.split(' ').list.len()==1)&(pl.col('c2').str.len_chars()>=6)&~pl.col('c2').str.contains(r'\d')&
        ~pl.struct('c1','c2').map_elements(lambda s: any(len(t)>=3 and t in s['c2'] for t in s['c1'].split()) or s['c2'].startswith(''.join(t[0] for t in s['c1'].split())),return_dtype=pl.Boolean)).alias('gib'))
if __name__=='__main__':
    h=pl.read_parquet(O+'ho_cats.parquet').filter(pl.col('active')&(pl.col('nm')=='none')&pl.col('ad').is_in(['aset','sameh']))
    h=gib(h).filter(pl.col('gib'))
    print('holdout', h.group_by('country','ad').agg(pl.len(),pl.col('y').mean(),pl.col('kept').mean(),((pl.col('y')*pl.col('kept')).sum()/pl.col('kept').sum()).alias('prec_kept')))
    print(h.sample(20,seed=2).select('country','y','kept','c1','c2'))
    import sys
    R='/Users/adityarohilla/Documents/amazon-ml-2026/'
    recs=pl.read_parquet(R+'cache/3ee42f05d06efbcb/recs.parquet',columns=['row','core'])
    d=pl.read_parquet(O+'fr_ctx.parquet').join(pl.read_parquet(O+'fr_cats.parquet').select('i','j','nm','ad'),on=['i','j']).filter((pl.col('nm')=='none')&pl.col('ad').is_in(['aset','sameh']))
    d=d.join(recs.select(pl.col('row').alias('i'),pl.col('core').alias('c1')),on='i').join(recs.select(pl.col('row').alias('j'),pl.col('core').alias('c2')),on='j'); del recs
    d=gib(d).filter(pl.col('gib')); d.write_parquet(O+'fr_gib.parquet')
    print('france', d.group_by('ad').agg(pl.len(),pl.col('kept').mean(),pl.col('p4').mean()))
    print(d.with_columns(pl.col('p4').cut([0.05,0.1,0.2,0.3,0.5,0.7,0.9]).alias('b')).group_by('b').agg(pl.len(),pl.col('kept').mean()).sort('b'))
