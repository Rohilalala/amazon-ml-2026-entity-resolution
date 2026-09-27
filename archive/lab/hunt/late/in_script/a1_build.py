import polars as pl, re, unicodedata
R='/Users/adityarohilla/Documents/amazon-ml-2026/'
O=R+'v8_lab/hunt/late/in_script/'
def script(s):
    if s is None: return 'none'
    c={}
    for ch in s:
        o=ord(ch)
        if o<128:
            if ch.isalpha(): c['lat']=c.get('lat',0)+1
            continue
        if 0x900<=o<=0x97F: k='deva'
        elif 0x980<=o<=0x9FF: k='beng'
        elif 0xA00<=o<=0xA7F: k='guru'
        elif 0xA80<=o<=0xAFF: k='guj'
        elif 0xB00<=o<=0xB7F: k='oriya'
        elif 0xB80<=o<=0xBFF: k='tamil'
        elif 0xC00<=o<=0xC7F: k='telugu'
        elif 0xC80<=o<=0xCFF: k='kann'
        elif 0xD00<=o<=0xD7F: k='mal'
        elif 0x600<=o<=0x6FF: k='arab'
        elif ch.isalpha(): k='olat' if o<0x250 else 'oth'
        else: continue
        c[k]=c.get(k,0)+1
    nat={k:v for k,v in c.items() if k not in('lat','olat')}
    if not nat: return 'latin'
    top=max(nat,key=nat.get)
    return top if c.get('lat',0)<=2 else 'mix_'+top
def recs(path):
    return (pl.scan_parquet(path).filter(pl.col('country_key')=='india')
            .select('row','entity_id','src','name','addr').collect())
# holdout
tr=recs(R+'cache/58f0a60c5c264fdd/recs.parquet')
tr=tr.with_columns(pl.col('name').map_elements(script,return_dtype=pl.Utf8).alias('ns'),
                   pl.col('addr').map_elements(script,return_dtype=pl.Utf8).alias('as'))
tr.write_parquet(O+'recs_train_india.parquet')
te=recs(R+'cache/3ee42f05d06efbcb/recs.parquet')
te=te.with_columns(pl.col('name').map_elements(script,return_dtype=pl.Utf8).alias('ns'),
                   pl.col('addr').map_elements(script,return_dtype=pl.Utf8).alias('as'))
te.write_parquet(O+'recs_test_india.parquet')
print(tr['ns'].value_counts().sort('count',descending=True).head(20))
print(te['ns'].value_counts().sort('count',descending=True).head(20))
