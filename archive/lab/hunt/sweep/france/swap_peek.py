import polars as pl, re, collections, sys
O='/Users/adityarohilla/Documents/amazon-ml-2026/v8_lab/hunt/sweep/france/'
LF={"sarl","sas","sasu","sa","eurl","snc","sci","scop","selarl","ei"}
def tok(n): return [t for t in re.sub(r"[^a-z0-9& ]"," ",n or "").replace('&',' and ').split() if t not in LF]
W1,W2,n=sys.argv[1],sys.argv[2],int(sys.argv[3])
d=pl.read_parquet(O+'fr_ctx.parquet').join(pl.read_parquet(O+'fr_cats.parquet').select('i','j','ad'),on=['i','j']).filter(pl.col('ad').is_in(['aset','sameh']))
def f(a,b):
    a,b=tok(a),tok(b)
    if len(a)!=len(b): return False
    sa,sb=collections.Counter(a),collections.Counter(b); return list((sa-sb).elements())==[W1] and list((sb-sa).elements())==[W2]
x=d.filter(pl.struct('n1','n2').map_elements(lambda s:f(s['n1'],s['n2']),return_dtype=pl.Boolean))
print(x.group_by('kept').agg(pl.len(),pl.col('p4').mean()))
for s in x.sample(min(n,x.height),seed=4)['s1']:
    y=d.filter(pl.col('s1')==s).sort('p4',descending=True,nulls_last=False); r=y.row(0,named=True); print(f"## {r['n1']} | {r['a1']}")
    for r in y.filter(pl.col('kept')|(pl.col('p4')>0.01)).head(8).iter_rows(named=True): print(f"  {'K' if r['kept'] else '.'} {r['p4'] if r['p4'] is None else round(r['p4'],3)} {r['m']} | {r['n2']} | {r['a2']}")
