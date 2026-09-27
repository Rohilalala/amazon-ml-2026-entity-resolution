import re, difflib, polars as pl
from cls import norm
ST = {"rue","r","avenue","av","ave","boulevard","bd","blvd","bouelvard","place","pl","impasse","imp","allee","all","chemin","chem","route","rte","quai","q","cours","crs","square","sq","street","st","road","rd","drive","dr","lane","ln","court","ct","circle","cir","way","trail","trl","de","du","des","la","le","l","d","bis","ter","no","n","unit","apartment","saint","sainte"}
def street(a):
    for p in (a or "").split(","):
        if re.search(r"\d", p):
            return " ".join(t for t in norm(p).split() if not t.isdigit() and t not in ST)
    return None
def srel(a1, a2):
    s1, s2 = street(a1), street(a2)
    if not s1 or not s2: return None
    r = difflib.SequenceMatcher(None, s1, s2).ratio()
    return "st_same" if r >= 0.75 else ("st_near" if r >= 0.5 else "st_diff")
def add_s(df): return df.with_columns(pl.struct("ad1","ad2").map_elements(lambda s: srel(s["ad1"], s["ad2"]), return_dtype=pl.String).alias("sr"))
