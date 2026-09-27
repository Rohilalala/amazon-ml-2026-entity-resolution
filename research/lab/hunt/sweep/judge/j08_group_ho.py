"""US/India analog: single-word swaps at the S1's own address in the holdout; truth rate by (dropped word, added word)."""
import polars as pl, re, collections
R='/Users/adityarohilla/Documents/amazon-ml-2026/'; O=R+'v8_lab/hunt/sweep/france/'
LF={"inc","llc","ltd","limited","pvt","private","corp","co","company","corporation","incorporated","llp","pllc","pc","lp"}
def tok(n): return [t for t in re.sub(r"[^a-z0-9& ]"," ",(n or "").lower()).replace('&',' and ').split() if t not in LF]
h=pl.read_parquet(O+'ho_cats.parquet').filter(pl.col('active')&pl.col('ad').is_in(['aset','sameh']))
tr=pl.scan_parquet(R+'cache/58f0a60c5c264fdd/recs.parquet').select('row','entity_id').filter(pl.col('row').is_in(pl.concat([h['i'],h['j']]).unique().implode())).collect()
T=R+'student_resource/dataset/train/'
nm=pl.concat([pl.read_csv(T+f'train_source{k}.tsv',separator='\t',quote_char=None,infer_schema=False,columns=['entity_id','business_name']) for k in (1,2,3)]).with_columns(pl.col('entity_id').str.strip_chars())
tr=tr.join(nm,on='entity_id').select('row',pl.col('business_name').alias('name')); del nm
h=h.join(tr.rename({'row':'i','name':'n1'}),on='i').join(tr.rename({'row':'j','name':'n2'}),on='j')
C=collections.defaultdict(list); A=collections.defaultdict(list)
for n1,n2,y in h.select('n1','n2','y').iter_rows():
    a,b=tok(n1),tok(n2)
    if len(a)!=len(b) or len(a)<2: continue
    sa,sb=collections.Counter(a),collections.Counter(b); x,z=list((sa-sb).elements()),list((sb-sa).elements())
# US/India: the word 'group' anywhere in the candidate but not in the S1 (same address): append, swap-at-end, other; truth rate.
P = collections.defaultdict(list)
for n1, n2, y in h.select('n1', 'n2', 'y').iter_rows():
    a, b = tok(n1), tok(n2)
    if 'group' in a or 'group' not in b: continue
    rest = [t for t in b if t != 'group']
    kind = 'append' if rest == a else ('swap_end' if len(b) == len(a) and b[-1] == 'group' else 'other')
    P[kind].append(y)
[print('group', k, len(v), round(sum(v) / len(v), 4)) for k, v in sorted(P.items())]
