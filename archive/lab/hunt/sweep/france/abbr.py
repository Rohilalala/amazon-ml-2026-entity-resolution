"""France pairs whose names are equal as token multisets after cie<->compagnie / et,&<->and canonicalisation, same address, not kept."""
import polars as pl, re
O='/Users/adityarohilla/Documents/amazon-ml-2026/v8_lab/hunt/sweep/france/'
pl.Config.set_tbl_rows(60); pl.Config.set_tbl_width_chars(250); pl.Config.set_fmt_str_lengths(60)
LF={"sarl","sas","sasu","sa","eurl","snc","sci","scop","selarl","ei"}
MAP={"compagnie":"cie","et":"and","&":"and"}
def canon(n): return ' '.join(sorted(MAP.get(t,t) for t in re.sub(r"[^a-z0-9& ]"," ",n or "").split() if t not in LF))
def raw(n): return ' '.join(sorted(t for t in re.sub(r"[^a-z0-9& ]"," ",n or "").split() if t not in LF))
d=pl.read_parquet(O+'fr_ctx.parquet').join(pl.read_parquet(O+'fr_cats.parquet').select('i','j','nm','ad'),on=['i','j'])
d=d.filter(pl.col('ad').is_in(['aset','sameh']))
d=d.with_columns(pl.col('n1').map_elements(canon,return_dtype=pl.String).alias('k1'),pl.col('n2').map_elements(canon,return_dtype=pl.String).alias('k2'),
                 pl.col('n1').map_elements(raw,return_dtype=pl.String).alias('r1'),pl.col('n2').map_elements(raw,return_dtype=pl.String).alias('r2'))
x=d.filter((pl.col('k1')==pl.col('k2'))&(pl.col('r1')!=pl.col('r2')))
print(x.group_by('ad','kept').agg(pl.len(),pl.col('p4').mean()))
print(x.filter(~pl.col('kept')).sample(min(25,x.filter(~pl.col('kept')).height),seed=1).select(pl.col('p4').round(3),'n1','n2','a1','a2'))
x.write_parquet(O+'abbr_pairs.parquet')
