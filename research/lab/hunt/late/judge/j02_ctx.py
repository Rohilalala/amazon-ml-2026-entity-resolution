# Context around the 3 region-conflict j records: who else sits at those addresses / shares the name
import polars as pl
c=pl.col
fr=pl.read_parquet('/tmp/fr_test.parquet')
pl.Config.set_fmt_str_lengths(80); pl.Config.set_tbl_rows(80); pl.Config.set_tbl_width_chars(250)
n=c.business_name.str.to_lowercase(); a=c.business_address.str.to_lowercase()
for lab,f in [('amicale tulipes', a.str.contains('tulipes')&n.str.contains('amicale')),
              ('pessac tulipes 23', a.str.contains('tulipes')&a.str.contains('pessac')&a.str.contains('23')),
              ('deleves', n.str.replace_all(r'[^a-z]','').str.contains('deleves')),
              ('dunkerque jaures no/any', a.str.contains('dunkerque')&a.str.contains('jean jaures')&(n.str.contains('club')|n.str.contains('delev'))),
              ('alccatmeud', n.str.contains('alccatmeud')),
              ('tourcoing jules verne 19', a.str.contains('tourcoing')&a.str.contains('jules verne')&a.str.contains('19')),
              ('artistique jules verne', n.str.contains('artistique')&a.str.contains('jules verne'))]:
    print('==',lab); print(fr.filter(f).select('entity_id','business_name','business_address').head(40))
