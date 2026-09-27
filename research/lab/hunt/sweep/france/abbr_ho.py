"""US/India analog of cie<->compagnie: holdout pairs whose names differ only by an abbreviation expansion (co/company, corp/corporation,
& /and, intl/international, mfg/manufacturing, assoc/associates, bros/brothers, svc/services), same address. Truth rate + v7 keep."""
import polars as pl, re
R='/Users/adityarohilla/Documents/amazon-ml-2026/'; O=R+'v8_lab/hunt/sweep/france/'
MAP={"company":"co","corporation":"corp","and":"&","international":"intl","manufacturing":"mfg","associates":"assoc","brothers":"bros","services":"svc","limited":"ltd","private":"pvt","incorporated":"inc","enterprises":"ent"}
def tok(n): return re.sub(r"[^a-z0-9& ]"," ",(n or "").lower()).split()
def canon(n): return ' '.join(MAP.get(t,t) for t in tok(n))
def raw(n): return ' '.join(tok(n))
h=pl.read_parquet(O+'ho_cats.parquet').filter(pl.col('active')&pl.col('ad').is_in(['aset','sameh']))
tr=pl.scan_parquet(R+'cache/58f0a60c5c264fdd/recs.parquet').select('row','entity_id').filter(pl.col('row').is_in(pl.concat([h['i'],h['j']]).unique().implode())).collect()
T=R+'student_resource/dataset/train/'
nm=pl.concat([pl.read_csv(T+f'train_source{k}.tsv',separator='\t',quote_char=None,infer_schema=False,columns=['entity_id','business_name']) for k in (1,2,3)]).with_columns(pl.col('entity_id').str.strip_chars())
tr=tr.join(nm,on='entity_id').select('row',pl.col('business_name').alias('name')); del nm
h=h.join(tr.rename({'row':'i','name':'n1'}),on='i').join(tr.rename({'row':'j','name':'n2'}),on='j')
h=h.with_columns(pl.col('n1').map_elements(canon,return_dtype=pl.String).alias('k1'),pl.col('n2').map_elements(canon,return_dtype=pl.String).alias('k2'),pl.col('n1').map_elements(raw,return_dtype=pl.String).alias('r1'),pl.col('n2').map_elements(raw,return_dtype=pl.String).alias('r2'))
x=h.filter((pl.col('k1')==pl.col('k2'))&(pl.col('r1')!=pl.col('r2')))
pl.Config.set_tbl_rows(40); pl.Config.set_tbl_width_chars(220); pl.Config.set_fmt_str_lengths(50)
print(x.group_by('country','ad').agg(pl.len(),pl.col('y').mean(),pl.col('kept').mean()))
x=x.with_columns(pl.struct('r1','r2').map_elements(lambda s: '/'.join(sorted(set(s['r1'].split())^set(s['r2'].split()))),return_dtype=pl.String).alias('diff'))
print(x.group_by('diff').agg(pl.len(),pl.col('y').mean(),pl.col('kept').mean()).sort('len',descending=True).head(15))
print(x.sample(12,seed=1).select('country','y','kept','n1','n2'))
x=x.with_columns(pl.struct('r1','r2').map_elements(lambda s: ' / '.join(['-'+t for t in sorted(set(s['r1'].split())-set(s['r2'].split()))]+['+'+t for t in sorted(set(s['r2'].split())-set(s['r1'].split()))]),return_dtype=pl.String).alias('dd'))
print(x.group_by('dd').agg(pl.len(),pl.col('y').mean(),pl.col('kept').mean()).sort('len',descending=True).head(14))
