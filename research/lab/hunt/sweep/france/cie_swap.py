"""Structural check: at the S1's own address, what does the S1 word 'cie' (and, as a control, other words) get replaced by?
A random sister-swap decoy would spread over many words; an abbreviation map concentrates on one."""
import polars as pl, re, collections
O='/Users/adityarohilla/Documents/amazon-ml-2026/v8_lab/hunt/sweep/france/'
LF={"sarl","sas","sasu","sa","eurl","snc","sci","scop","selarl","ei"}
def tok(n): return [t for t in re.sub(r"[^a-z0-9& ]"," ",n or "").replace('&',' and ').split() if t not in LF]
d=pl.read_parquet(O+'fr_ctx.parquet').join(pl.read_parquet(O+'fr_cats.parquet').select('i','j','ad'),on=['i','j']).filter(pl.col('ad').is_in(['aset','sameh']))
C=collections.Counter(); K=collections.defaultdict(list); W=collections.Counter()
for n1,n2,k,p in d.select('n1','n2','kept','p4').iter_rows():
    a,b=tok(n1),tok(n2)
    if len(a)!=len(b): continue
    sa,sb=collections.Counter(a),collections.Counter(b)
    x,y=list((sa-sb).elements()),list((sb-sa).elements())
    if len(x)==1 and len(y)==1:
        W[x[0]]+=1
        if x[0] in ('cie','club','and'): C[(x[0],y[0])]+=1; K[(x[0],y[0])].append(k)
for w in ('cie','club','and'):
    tot=sum(v for (a,b),v in C.items() if a==w)
    print(w, 'swaps', tot, [(b,v,round(sum(K[(a,b)])/v,2)) for (a,b),v in C.most_common() if a==w][:10])
