"""Structural (name relation x address relation) categories: France test keep rate vs US/India holdout truth rate."""
import polars as pl
R='/Users/adityarohilla/Documents/amazon-ml-2026/'; O=R+'v8_lab/hunt/sweep/france/'
pl.Config.set_tbl_rows(100); pl.Config.set_tbl_width_chars(200)
def cats(d):
    t1=pl.col('c1').str.split(' ').list.unique(); t2=pl.col('c2').str.split(' ').list.unique()
    a1=pl.col('a1').fill_null('').str.split(' ').list.unique(); a2=pl.col('a2').fill_null('').str.split(' ').list.unique()
    inter=t1.list.set_intersection(t2).list.len()
    nm=(pl.when(pl.col('c1')==pl.col('c2')).then(pl.lit('eq')).when((inter==t1.list.len())&(inter==t2.list.len())).then(pl.lit('perm'))
        .when(inter==t2.list.len()).then(pl.lit('c_sub')).when(inter==t1.list.len()).then(pl.lit('c_sup')).when(inter==0).then(pl.lit('none')).otherwise(pl.lit('part')))
    ai=a1.list.set_intersection(a2).list.len()
    ad=(pl.when(pl.col('a2').fill_null('').str.strip_chars()=='').then(pl.lit('empty')).when((ai==a1.list.len())&(ai==a2.list.len())).then(pl.lit('aset'))
        .when(pl.col('h1')==pl.col('h2')).then(pl.lit('sameh')).when(pl.col('h1').is_null()|pl.col('h2').is_null()).then(pl.lit('noh')).otherwise(pl.lit('diffh')))
    return d.with_columns(nm.alias('nm'),ad.alias('ad'))
if __name__=='__main__':
    recs=pl.read_parquet(R+'cache/3ee42f05d06efbcb/recs.parquet',columns=['row','core'])
    d=pl.read_parquet(O+'fr_ctx.parquet').join(recs.select(pl.col('row').alias('i'),pl.col('core').alias('c1')),on='i').join(recs.select(pl.col('row').alias('j'),pl.col('core').alias('c2')),on='j'); del recs
    d=cats(d); d.select('i','j','s1','m','nm','ad').write_parquet(O+'fr_cats.parquet')
    ft=d.group_by('nm','ad').agg(pl.len().alias('fr_n'),pl.col('kept').mean().round(3).alias('fr_kept'),pl.col('p4').mean().round(3).alias('fr_p4'))
    tr=pl.read_parquet(R+'cache/58f0a60c5c264fdd/recs.parquet',columns=['row','core','addr','house'])
    h=pl.read_parquet(R+'v8_lab/ce/setup/out/holdout_pairs.parquet',columns=['i','j','country','y','kept','active'])
    h=h.join(tr.select(pl.col('row').alias('i'),pl.col('core').alias('c1'),pl.col('addr').alias('a1'),pl.col('house').alias('h1')),on='i').join(tr.select(pl.col('row').alias('j'),pl.col('core').alias('c2'),pl.col('addr').alias('a2'),pl.col('house').alias('h2')),on='j'); del tr
    h=cats(h); h.select('i','j','country','y','kept','active','nm','ad','c1','c2','a1','a2').write_parquet(O+'ho_cats.parquet')
    ht=h.filter(pl.col('active')).group_by('nm','ad').agg(pl.len().alias('ho_n'),pl.col('y').mean().round(3).alias('ho_y'),pl.col('kept').mean().round(3).alias('ho_kept_v7'))
    print(ft.join(ht,on=['nm','ad'],how='full',coalesce=True).sort('fr_n',descending=True,nulls_last=True))
