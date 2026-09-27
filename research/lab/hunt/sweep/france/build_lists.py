"""France ADD lists (columns s1, m) for pairs the round-2 file rejects although they follow the France true-copy generator.
Common guards: France S1; j is an active test candidate of the S1; same address (address token sets equal, or equal house number
and digit-stripped address token_set_ratio >= 85); legal-form tokens equal; pair not kept; j kept by no S1 in final16b; j offered
to one S1 only.
 A adds_cie.parquet    'cie' -> 'compagnie' expansion: candidate core = S1 core with 'cie' replaced by 'compagnie' (et/&/and
                       treated as one word, order free). compagnie->cie is kept 0.99; cie->compagnie 0.13. 'cie' is replaced by
                       'compagnie' 695 times at the same address vs <= 12 for any sister word -> an abbreviation map, not a
                       sister swap. US/India holdout analog (co/company, inc/incorporated, ltd/limited, corp/corporation, &/and,
                       both directions, same address): 8.3k pairs, truth 1.000.
 B adds_tailmid.parquet tail swap on a non-last word: candidate core = S1 core minus one word that is 'and' or not the last word,
                       plus 'groupe' or 'france' as the final word. Same-slot counts are uniform across the five tails
                       (mid: cie 553, developpement 613, services 574, groupe 624, france 493; and: 168/188/178/186/173) but
                       the file keeps cie/developpement/services 0.98-1.0 and groupe/france 0.00-0.24. US/India holdout analog
                       (center/services/service/partners tail at end, same address, dropped word not last): mid 2,537 pairs
                       0.9965 true, 'and' 469 pairs 0.9936 true.
 adds_all.parquet = A + B."""
import polars as pl, re, collections
from rapidfuzz import fuzz
R='/Users/adityarohilla/Documents/amazon-ml-2026/'; O=R+'v8_lab/hunt/sweep/france/'
LF={"sarl","sas","sasu","sa","eurl","snc","sci","scop","selarl","ei"}; D=re.compile(r"\d+\w*")
def parts(n):
    t=re.sub(r"[^a-z0-9& ]"," ",n or "").replace('&',' and ').split()
    return ['and' if x=='et' else x for x in t if x not in LF], sorted(set(x for x in t if x in LF))
def rule(n1,n2):
    (a,la),(b,lb)=parts(n1),parts(n2)
    if la!=lb or len(a)!=len(b) or len(a)<2: return None
    sa,sb=collections.Counter(a),collections.Counter(b); x,z=list((sa-sb).elements()),list((sb-sa).elements())
    if x==['cie'] and z==['compagnie']: return 'A'
    if len(x)==1 and z in (['groupe'],['france']) and b[-1]==z[0] and (x[0]=='and' or a[-1]!=x[0]): return 'B'
    return None
d=pl.read_parquet(O+'fr_ctx.parquet').join(pl.read_parquet(O+'fr_cats.parquet').select('i','j','ad'),on=['i','j']).filter(pl.col('ad').is_in(['aset','sameh'])&~pl.col('kept'))
d=d.with_columns(pl.struct('n1','n2').map_elements(lambda s: rule(s['n1'],s['n2']),return_dtype=pl.String).alias('rule')).filter(pl.col('rule').is_not_null())
d=d.with_columns(pl.struct('a1','a2').map_elements(lambda s: int(fuzz.token_set_ratio(D.sub(' ',s['a1'] or ''),D.sub(' ',s['a2'] or ''))),return_dtype=pl.Int64).alias('st'))
d=d.filter((pl.col('ad')=='aset')|(pl.col('st')>=85))
b=pl.read_csv(R+'v8_lab/hunt/final16b/matching_results.tsv',separator='\t',quote_char=None,infer_schema=False)
claimed=b.with_columns(pl.col('matched_entity_ids').fill_null('').str.split(',')).explode('matched_entity_ids')['matched_entity_ids']
d=d.filter(~pl.col('m').is_in(claimed.implode())).filter(pl.col('m').is_unique())
print(d.group_by('rule','ad').agg(pl.len(),pl.col('p4').mean().round(3),pl.col('s1').n_unique().alias('s1')).sort('rule','ad'))
d.select('s1','m','rule','ad','st',pl.col('p4').round(3),'n1','n2','a1','a2').write_parquet(O+'adds_detail.parquet')
for name,flt in (('adds_cie',pl.col('rule')=='A'),('adds_tailmid',pl.col('rule')=='B'),('adds_all',pl.lit(True))): d.filter(flt).select('s1','m').write_parquet(O+name+'.parquet')
pl.DataFrame(schema={'s1':pl.String,'m':pl.String}).write_parquet(O+'drops_none.parquet')
