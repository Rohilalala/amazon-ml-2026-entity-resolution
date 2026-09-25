"""Vectorised normalisation of business names and addresses (polars).

Every record from every source goes through the same functions, so the
abbreviation maps only need to be consistent, not exhaustive: if both sides
map "Rd" and "Road" to "road", the comparison sees the same token.

Non-Latin scripts (Devanagari, Malayalam, Kannada, ...) are transliterated to
ASCII first; the phonetic `skeleton` then absorbs most spelling differences
between a transliteration and the English spelling
("helthkeyr" / "healthcare" -> "ltkr").
"""
import re

import polars as pl
from anyascii import anyascii

# Token-level expansions for business names.
NAME_ABBREV = {
    "corp": "corporation", "co": "company", "cos": "companies",
    "inc": "incorporated", "incorp": "incorporated",
    "ltd": "limited", "lmt": "limited", "pvt": "private", "pte": "private",
    "intl": "international", "mfg": "manufacturing", "mfrs": "manufacturers",
    "svc": "services", "svcs": "services", "srvcs": "services",
    "bros": "brothers", "assoc": "associates", "assocs": "associates",
    "natl": "national", "mgmt": "management", "grp": "group",
    "ent": "enterprises", "ents": "enterprises", "entp": "enterprises",
    "tech": "technologies", "techs": "technologies", "sys": "systems",
    "engg": "engineering", "mktg": "marketing", "univ": "university",
    "hosp": "hospital", "ctr": "center", "centre": "center", "cntr": "center",
    "trdg": "trading", "dept": "department", "ets": "etablissements",
}

# Legal forms, honorifics and filler words dropped to get a "core" name.
NAME_STOP = [
    "private", "limited", "llp", "llc", "lp", "plc", "incorporated", "corporation",
    "company", "companies", "the", "and", "of", "formerly", "dba", "aka",
    "com", "www", "net", "org", "in", "ms", "m", "s", "shri", "sri", "shree", "sree", "smt",
    # French legal forms and articles (France appears only in the test set).
    "sarl", "sas", "sasu", "sa", "eurl", "snc", "sci", "et", "de", "la", "le",
    "les", "du", "des", "d", "l",
]

# Token-level expansions for addresses.
ADDR_ABBREV = {
    "rd": "road", "st": "street", "str": "street", "ave": "avenue",
    "av": "avenue", "blvd": "boulevard", "bd": "boulevard", "ln": "lane",
    "dr": "drive", "ct": "court", "pl": "place", "sq": "square",
    "hwy": "highway", "pkwy": "parkway", "expy": "expressway", "cir": "circle",
    "ter": "terrace", "trl": "trail", "rte": "route", "ste": "suite",
    "fl": "floor", "flr": "floor", "bldg": "building", "apt": "apartment",
    "nr": "near", "opp": "opposite", "mkt": "market", "ngr": "nagar",
    "sec": "sector", "sect": "sector", "clny": "colony", "soc": "society",
    "hsg": "housing", "dist": "district", "distt": "district",
    "chem": "chemin", "imp": "impasse", "fbg": "faubourg", "fg": "faubourg",
    "mt": "mount", "ft": "fort", "pt": "point",
    # French street types (France appears only in the test set).
    "r": "rue", "all": "allee", "bld": "boulevard", "crs": "cours", "pass": "passage",
    "qu": "quai", "sq": "square", "rpt": "rondpoint",
}

# State / union-territory names -> postal codes, so "Arizona" and "AZ" (or
# "Uttar Pradesh" and "UP") compare equal. Applied to whole phrases before the
# token expansions above; both sides pass through the same maps.
STATES = {
    "alabama": "al", "alaska": "ak", "arizona": "az", "arkansas": "ar", "california": "ca",
    "colorado": "co", "connecticut": "ct", "delaware": "de", "florida": "fl", "georgia": "ga",
    "hawaii": "hi", "idaho": "id", "illinois": "il", "indiana": "in", "iowa": "ia",
    "kansas": "ks", "kentucky": "ky", "louisiana": "la", "maine": "me", "maryland": "md",
    "massachusetts": "ma", "michigan": "mi", "minnesota": "mn", "mississippi": "ms",
    "missouri": "mo", "montana": "mt", "nebraska": "ne", "nevada": "nv", "new hampshire": "nh",
    "new jersey": "nj", "new mexico": "nm", "new york": "ny", "north carolina": "nc",
    "north dakota": "nd", "ohio": "oh", "oklahoma": "ok", "oregon": "or", "pennsylvania": "pa",
    "rhode island": "ri", "south carolina": "sc", "south dakota": "sd", "tennessee": "tn",
    "texas": "tx", "utah": "ut", "vermont": "vt", "virginia": "va", "washington": "wa",
    "west virginia": "wv", "wisconsin": "wi", "wyoming": "wy", "district of columbia": "dc",
    "andhra pradesh": "ap", "arunachal pradesh": "arp", "assam": "as", "bihar": "br",
    "chhattisgarh": "cg", "chattisgarh": "cg", "goa": "ga", "gujarat": "gj", "haryana": "hr",
    "himachal pradesh": "hp", "jharkhand": "jh", "karnataka": "ka", "kerala": "kl",
    "keralam": "kl", "madhya pradesh": "mp", "maharashtra": "mh", "manipur": "mn",
    "meghalaya": "ml", "mizoram": "mz", "nagaland": "nl", "odisha": "od", "orissa": "od",
    "punjab": "pb", "rajasthan": "rj", "sikkim": "sk", "tamil nadu": "tn", "tamilnadu": "tn",
    "telangana": "ts", "tripura": "tr", "uttar pradesh": "up", "uttarakhand": "uk",
    "uttaranchal": "uk", "west bengal": "wb", "delhi": "dl", "jammu and kashmir": "jk",
    "ladakh": "la", "chandigarh": "ch", "puducherry": "py", "pondicherry": "py",
    "andaman and nicobar islands": "an", "lakshadweep": "ld",
}

_PLACEHOLDER = r"(?i)<null>|\bn/a\b|^\s*(none|null|nan|na)\s*$"


# Indic word -> English word, learned from training data (see indic.py). Empty until
# set_native_words() is called; then Indic words it covers are translated before
# the generic anyascii transliteration handles everything else.
_NATIVE: dict = {}
_INDIC_RUN = re.compile(r"[\u0900-\u0DFF\u200c\u200d]+(?:\s+[\u0900-\u0DFF\u200c\u200d]+)*")
_INDIC_WORD = re.compile(r"[\u0900-\u0DFF\u200c\u200d]+")


def set_native_words(words: dict) -> None:
    """Install the Indic -> English dictionary used by transliterate()."""
    global _NATIVE
    _NATIVE = dict(words)


def _native_word(m: re.Match) -> str:
    return f" {_NATIVE[m.group(0)]} " if m.group(0) in _NATIVE else m.group(0)


def _native_run(m: re.Match) -> str:
    run = m.group(0)  # whole runs first: multi-word state names such as उत्तर प्रदेश
    return f" {_NATIVE[run]} " if run in _NATIVE else _INDIC_WORD.sub(_native_word, run)


def _to_ascii(x: str) -> str:
    return anyascii(_INDIC_RUN.sub(_native_run, x) if _NATIVE else x)


def transliterate(s: pl.Series) -> pl.Series:
    """ASCII transliteration; only non-ASCII strings pay the Python-level cost."""
    idx = s.str.contains(r"[^\x00-\x7F]").arg_true()
    if len(idx) == 0:
        return s
    return s.scatter(idx, s.gather(idx).map_elements(_to_ascii, return_dtype=pl.String))


def is_non_latin(s: pl.Series) -> pl.Series:
    """True for strings containing letters outside Latin/Latin-1 (e.g. Indic scripts)."""
    return s.str.contains(r"[^\x00-\x{024F}]")


def clean(s: pl.Series) -> pl.Series:
    """Lower-case ASCII, placeholders ('<NULL>', 'N/A', 'None') removed, '&' -> 'and',
    apostrophes dropped, other punctuation -> single spaces."""
    s = transliterate(s.fill_null("")).str.to_lowercase()
    s = s.str.replace_all(_PLACEHOLDER, " ")
    s = s.str.replace_all("&", " and ").str.replace_all("'", "")
    # Glue digit runs split by noise ("1-59", "560 001" stay distinct, "22-d" -> "22d").
    s = s.str.replace_all(r"(\d)[-/.](\d)", "$1$2")
    s = s.str.replace_all(r"(\d)(st|nd|rd|th)\b", "$1")  # ordinals: 14th -> 14
    s = s.str.replace_all(r"[^a-z0-9]+", " ").str.strip_chars()
    return _join_initials(s)


def _glue(text: str) -> str:
    out, run = [], []
    for tok in text.split(" "):
        if len(tok) == 1 and tok.isalpha():
            run.append(tok)
            continue
        if run:
            out.append("".join(run))
            run = []
        out.append(tok)
    if run:
        out.append("".join(run))
    return " ".join(out)


def _join_initials(s: pl.Series) -> pl.Series:
    """Glue runs of single-letter tokens: 's a r l' -> 'sarl', 'l l c' -> 'llc',
    'm g road' -> 'mg road', so dotted forms (S.A.R.L., M.G.) equal undotted ones.
    A lone letter ('a 203') is left alone."""
    idx = s.str.contains(r"(^| )[a-z] [a-z]( |$)").arg_true()
    if len(idx) == 0:
        return s
    return s.scatter(idx, s.gather(idx).map_elements(_glue, return_dtype=pl.String))


def expand(s: pl.Series, mapping: dict) -> pl.Series:
    """Replace whole tokens using `mapping` (e.g. 'rd' -> 'road')."""
    return s.str.split(" ").list.eval(pl.element().replace(mapping)).list.join(" ")


def replace_phrases(s: pl.Series, mapping: dict) -> pl.Series:
    """Replace whole multi-word phrases (longest first). Spaces are doubled so two
    adjacent phrases ('ohio texas') each keep their own boundary."""
    padded = pl.select(pl.concat_str([pl.lit(" "), pl.lit(s).str.replace_all(" ", "  "), pl.lit(" ")])).to_series()
    keys = sorted(mapping, key=len, reverse=True)  # leftmost-first: longest phrase wins
    out = padded.str.replace_many([f" {k.replace(' ', '  ')} " for k in keys],
                                  [f" {mapping[k]} " for k in keys], leftmost=True)
    return out.str.replace_all(" +", " ").str.strip_chars()


def clean_name(s: pl.Series) -> pl.Series:
    return expand(clean(s), NAME_ABBREV)


def clean_address(s: pl.Series) -> pl.Series:
    return expand(replace_phrases(clean(s), STATES), ADDR_ABBREV)


def core_name(name: pl.Series) -> pl.Series:
    """Name without legal forms, honorifics and fillers (falls back to the full name).
    Stop words are also matched by skeleton, which catches transliterated forms
    ('praivet' -> 'prvt' = skeleton of 'private')."""
    e = pl.element()
    core = name.str.split(" ").list.eval(
        e.filter((e != "") & ~e.is_in(NAME_STOP) & ~skeleton(e).is_in(_STOP_SKELETONS))).list.join(" ")
    return pl.select(pl.when(core == "").then(name).otherwise(core)).to_series()


def skeleton(e: pl.Expr) -> pl.Expr:
    """Phonetic skeleton of a token: c/q->k, z->s, w->v, x->ks, ph->f, 'h' and vowels
    dropped, repeated consonants collapsed. Digits pass through unchanged."""
    e = (e.str.replace_all("ph", "f").str.replace_all("[cq]", "k").str.replace_all("z", "s")
         .str.replace_all("w", "v").str.replace_all("x", "ks").str.replace_all("h", "")
         .str.replace_all("[aeiouy]", ""))
    for ch in "bdfgjklmnprstv":
        e = e.str.replace_all(ch + "+", ch)
    return e


# Skeletons of long legal words only; short ones ('smt' for Smt., 't' for 'the')
# would also remove real words such as 'Smith'.
_STOP_SKELETONS = pl.select(skeleton(pl.lit(pl.Series(
    ["private", "limited", "company", "companies", "corporation", "incorporated", "formerly"]
)))).to_series().to_list()


def skeleton_string(s: pl.Series) -> pl.Series:
    """Skeleton applied token by token to a whole string."""
    return s.str.split(" ").list.eval(skeleton(pl.element())).list.join(" ")


def concat_skeleton(core: pl.Series) -> pl.Series:
    """Skeleton of the core name written without spaces; equals the skeleton of a
    website-style name ('qualityaceculture.com') built from the same words."""
    return pl.select(skeleton(pl.lit(core).str.replace_all(" ", ""))).to_series()


def acronym(core: pl.Series) -> pl.Series:
    """Initials of multi-word names ('tata consultancy services' -> 'tcs'), else ''."""
    toks = core.str.split(" ")
    return pl.select(pl.when(toks.list.len() >= 2)
                     .then(toks.list.eval(pl.element().str.slice(0, 1)).list.join(""))
                     .otherwise(pl.lit(""))).to_series()
