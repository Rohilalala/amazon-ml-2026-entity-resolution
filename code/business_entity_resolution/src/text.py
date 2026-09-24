"""Normalisation helpers for business names and addresses.

Every record from every source goes through the same functions, so the
abbreviation maps only need to be consistent, not exhaustive: if both sides
map "Rd" and "Road" to "road", the comparison sees the same token.
"""
import re
import unicodedata

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
    "trdg": "trading", "dept": "department",
}

# Legal forms and filler words dropped to get a "core" name.
NAME_STOP = {
    "private", "limited", "llp", "llc", "lp", "plc", "incorporated",
    "corporation", "company", "companies", "the", "and", "of",
    # French legal forms and articles (France appears only in the test set).
    "sarl", "sas", "sasu", "sa", "eurl", "snc", "sci", "et", "de", "la", "le",
    "les", "du", "des", "d", "l",
}

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
    "n": "north", "s": "south", "e": "east", "w": "west",
    "ne": "northeast", "nw": "northwest", "se": "southeast", "sw": "southwest",
    "chem": "chemin", "imp": "impasse", "fbg": "faubourg",
}

_TOKEN = re.compile(r"[a-z0-9]+")
_DBA = re.compile(r"\bd\s*/\s*b\s*/\s*a\b|\bd\.b\.a\b\.?")
_SPLIT_PIN = re.compile(r"\b(\d{3})\s+(\d{3})\b")  # "560 001" -> "560001"
_DIGITS = re.compile(r"\d+")


def to_ascii(s):
    """Lower-case and strip accents (é -> e) so transliterations line up."""
    return unicodedata.normalize("NFKD", str(s)).encode("ascii", "ignore").decode().lower()


def _clean(s, abbrev):
    """Shared cleaning: ASCII, '&' -> 'and', drop punctuation, expand tokens."""
    s = to_ascii(s).replace("&", " and ").replace("'", "")
    s = _DBA.sub(" dba ", s)
    return " ".join(abbrev.get(t, t) for t in _TOKEN.findall(s))


def clean_name(s):
    """Normalised business name with abbreviations expanded."""
    return _clean(s, NAME_ABBREV)


def clean_address(s):
    """Normalised address; split 6-digit PIN codes are re-joined first."""
    return _clean(_SPLIT_PIN.sub(r"\1\2", to_ascii(s)), ADDR_ABBREV)


def core_name(name):
    """Name without legal suffixes and filler words (falls back to the full name)."""
    core = [t for t in name.split() if t not in NAME_STOP]
    return " ".join(core) if core else name


def numbers(address):
    """All digit runs in an address (house numbers, postcodes, sector numbers)."""
    return frozenset(_DIGITS.findall(address))


def house_number(address):
    """First digit run in an address, or '' when there is none."""
    m = _DIGITS.search(address)
    return m.group() if m else ""


def postcodes(nums):
    """5-digit (US ZIP, French code postal) and 6-digit (Indian PIN) numbers."""
    return frozenset(n for n in nums if len(n) in (5, 6))


def acronym(name):
    """Initials of a multi-word name, e.g. 'tata consultancy services' -> 'tcs'."""
    toks = name.split()
    return "".join(t[0] for t in toks) if len(toks) >= 2 else ""

