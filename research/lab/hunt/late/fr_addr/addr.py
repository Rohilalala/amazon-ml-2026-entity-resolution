import re, unicodedata
TYPES={'r':'rue','rue':'rue','av':'avenue','ave':'avenue','avenue':'avenue','aevnue':'avenue','avnue':'avenue','bd':'boulevard','boulevard':'boulevard','blvd':'boulevard','bld':'boulevard','pl':'place','place':'place','ch':'chemin','chemin':'chemin','chem':'chemin','imp':'impasse','impasse':'impasse','rte':'route','route':'route','all':'allee','allee':'allee','allees':'allee','fbg':'faubourg','faubourg':'faubourg','qu':'quai','quai':'quai','crs':'cours','cours':'cours','sq':'square','square':'square','pass':'passage','passage':'passage','res':'residence','residence':'residence','cite':'cite','sente':'sente','voie':'voie','esplanade':'esplanade','prom':'promenade','promenade':'promenade','parvis':'parvis','hameau':'hameau','ham':'hameau','lot':'lotissement','lotissement':'lotissement','zac':'zac','za':'za','zi':'zi','rpt':'rondpoint','rond':'rondpoint','mail':'mail','villa':'villa','ruelle':'ruelle'}
SUF={'bis':'b','ter':'t','quater':'q','quatdr':'q'}
ART={'de','du','des','la','le','les','l','d','et','a','au','aux'}
HN=re.compile(r'^(?:(?:no|n|numero)\s*\.?\s*|#\s*)?(\d+)\s*(bis|ter|quater|quatdr|[a-hqt])?(?:\s*[-/]\s*(\d+))?\b\s*(.*)$')
def strip(s):
    return ''.join(c for c in unicodedata.normalize('NFKD',s) if not unicodedata.combining(c)).lower()
def parse(a):
    """return (hn, suffix, stype, sname_tokens, other_components) for the street component"""
    if a is None: return None
    comps=[c.strip() for c in strip(a).split(',') if c.strip()]
    street=None; rest=[]
    for c in comps:
        c2=re.sub(r"[\.']",' ',c); c2=re.sub(r'\s+',' ',c2).strip()
        m=HN.match(c2)
        toks=c2.split()
        if street is None and (m or (toks and toks[0] in TYPES)):
            street=(c2,m)
        else: rest.append(c2)
    if street is None: return (None,None,None,None,tuple(rest))
    c2,m=street
    if m: hn,suf,rng,body=m.group(1),SUF.get(m.group(2),m.group(2)),m.group(3),m.group(4)
    else: hn,suf,rng,body=None,None,None,c2
    toks=body.split()
    # suffix written as separate token after number e.g. '24 bis av'
    st=None
    if toks and toks[0] in TYPES: st=TYPES[toks[0]]; toks=toks[1:]
    # 'grande r' etc: search type anywhere in first 3 tokens
    elif any(t in TYPES for t in toks[:3]):
        k=next(k for k,t in enumerate(toks[:3]) if t in TYPES); st=TYPES[toks[k]]; toks=toks[:k]+toks[k+1:]
    name=tuple(t for t in toks if t not in ART)
    return (hn,suf,st,name,tuple(rest))
if __name__=='__main__':
    for a in ["24 BIS AV SINT EXUPÉRY, LA TESTE-DE-BUCH, Gironde","168 Ter Route De La Côte D'amour, Saint-nazaire","NO 190 BD. JULES VERNE, NANTES","# 7 R De La Vieille Comédie, Lille","266 B RUE MANDRON, Bordeaux","Nord, ROUBAIX, 40 R. ROLLIN","RUE LEFEVRE-UTILE, NANTES","1016 GRANDE R DU PETIT COURGAIN, CALAIS","12-14 rue x, Lille"]:
        print(a,'->',parse(a))
