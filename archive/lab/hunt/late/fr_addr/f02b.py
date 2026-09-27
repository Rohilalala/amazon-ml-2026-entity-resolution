import polars as pl, sys
sys.path.insert(0,'.'); from addr import parse
pr=pl.read_parquet('fr_pairs.parquet')
for side,c in (('aa','a'),('ab','b')):
    u={a:parse(a) or (None,None,None,None,()) for a in pr[side].unique().to_list()}
    v=[u[a] for a in pr[side].to_list()]
    pr=pr.with_columns(pl.Series(c+'_hn',[x[0] for x in v]),pl.Series(c+'_suf',[x[1] for x in v]),pl.Series(c+'_st',[x[2] for x in v]),pl.Series(c+'_nm',[' '.join(x[3]) if x[3] else None for x in v],dtype=pl.Utf8))
pr.write_parquet('fr_pairs.parquet')
