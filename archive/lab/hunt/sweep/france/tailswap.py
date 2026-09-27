"""France single-word swaps at the S1's own address (address token set equal or same house): kept rate by (dropped word, added tail),
split by legal-form agreement and by the dropped word's position. Writes fr_swaps.parquet."""
import polars as pl, re, collections
O='/Users/adityarohilla/Documents/amazon-ml-2026/v8_lab/hunt/sweep/france/'
pl.Config.set_tbl_rows(80); pl.Config.set_tbl_width_chars(200)
LF={"sarl","sas","sasu","sa","eurl","snc","sci","scop","selarl","ei"}
TAILS={"fils","cie","compagnie","services","developpement","groupe","france","associes"}
def parts(n): t=re.sub(r"[^a-z0-9& ]"," ",n or "").replace('&',' and ').split(); return [x for x in t if x not in LF], sorted(set(x for x in t if x in LF))
d=pl.read_parquet(O+'fr_ctx.parquet').join(pl.read_parquet(O+'fr_cats.parquet').select('i','j','ad'),on=['i','j']).filter(pl.col('ad').is_in(['aset','sameh']))
rows=[]
for s1,m,n1,n2,k,p,ad in d.select('s1','m','n1','n2','kept','p4','ad').iter_rows():
    (a,la),(b,lb)=parts(n1),parts(n2)
    if len(a)!=len(b) or len(a)<2: continue
    sa,sb=collections.Counter(a),collections.Counter(b); x,z=list((sa-sb).elements()),list((sb-sa).elements())
    if len(x)==1 and len(z)==1 and z[0] in TAILS:
        pos='and' if x[0]=='and' else ('last' if a[-1]==x[0] else 'mid')
        rows.append((s1,m,x[0],z[0],pos,la==lb,b[-1]==z[0],k,p,ad))
s=pl.DataFrame(rows,schema=['s1','m','drop','add','pos','lf_ok','tail_last','kept','p4','ad'],orient='row')
s.write_parquet(O+'fr_swaps.parquet')
print(s.group_by('pos','add','lf_ok').agg(pl.len(),pl.col('kept').mean().round(3)).filter(pl.col('len')>=20).sort('pos','add','lf_ok'))
