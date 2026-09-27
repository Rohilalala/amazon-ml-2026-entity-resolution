"""The rules behind the pair lists of steps E-G (lists.parquet), as code over a decision.

Each function reads the test records and a submission frame (postprocess) and returns the pairs
(s1, m) the rule drops or adds on that frame; postprocess.apply_lists then applies them with the
chain's shared semantics. The leaderboard run derived each list on an earlier decision of the
same run (named per rule below) and applied the fixed list later in the chain, so `final` replays
lists.parquet by default; `derive` recomputes the lists on the current frame instead.

  E drops  street_drops        France: same house number, different street (a decoy business).
  E adds   groupe_swap_adds    France: the Source 1 name with its last core word swapped for
                               'groupe' at the same address (a generator true-copy pattern the
                               encoder rejects in France).
  F adds   cie_tailmid_adds    France: 'cie' written out as 'compagnie', or a non-last word
                               replaced by a final 'groupe' / 'france', at the same address.
  F adds   india_copy_adds     India: an unscored Source 2/3 record with the same core name and
                               address as a kept record (a pool copy), with its source records.
  G drops  sister2_drops       France: a sister-organisation word swap at any position (the step C
                               rule without its position constraint).
  G drops  us_state_conflict_drops  US: the two addresses carry different state codes.
  H drops  region_conflict_drops  France: the two addresses name (or have cities in) different regions.
Lists without code here (lists.parquet only, see README): G source-cap drops and France
no-legal-form groupe adds; H near-tie ownership abstention.
"""
import collections
import difflib
import re
import unicodedata

import polars as pl
from rapidfuzz import fuzz
from rapidfuzz.distance import Levenshtein

from . import postprocess

S1, M = postprocess.S1, postprocess.M
# Street-type words and articles ignored when comparing street words.
STREET_TYPES = {"rue", "r", "avenue", "av", "ave", "boulevard", "bd", "blvd", "place", "pl", "impasse", "imp", "allee", "all",
                "chemin", "chem", "route", "rte", "quai", "q", "cours", "crs", "square", "sq", "passage", "pas", "psg", "mail",
                "cite", "sentier", "sent", "esplanade", "promenade", "parvis", "rond", "voie"}
STREET_STOP = {"rue", "r", "avenue", "av", "ave", "boulevard", "bd", "blvd", "bouelvard", "place", "pl", "impasse", "imp",
               "allee", "all", "chemin", "chem", "route", "rte", "quai", "q", "cours", "crs", "square", "sq", "street", "st",
               "road", "rd", "drive", "dr", "lane", "ln", "court", "ct", "circle", "cir", "way", "trail", "trl", "de", "du",
               "des", "la", "le", "l", "d", "bis", "ter", "no", "n", "unit", "apartment", "saint", "sainte"}
REGION_WORDS = {"hauts", "france", "pays", "loire", "nouvelle", "aquitaine", "nord", "gironde", "atlantique", "calais"}
LEGAL_FORMS = {"sarl", "sas", "sasu", "sa", "eurl", "snc", "sci", "scop", "selarl", "ei"}
_DIGITS = re.compile(r"\d+\w*")


def _norm(s: str | None) -> str:
    s = unicodedata.normalize("NFKD", s or "").encode("ascii", "ignore").decode().lower()
    return re.sub(r"[^a-z0-9 ]", " ", s)


def _similar(a: str, b: str) -> bool:
    return difflib.SequenceMatcher(None, a, b).ratio() >= 0.6


def _street_words(a: str | None, component_only: bool) -> list | None:
    """Words (4+ letters, no digit, no street type) of the first address component with a digit,
    or of the whole address (minus region words) when not `component_only`."""
    if not component_only:
        return [t for t in _norm(a).split() if not re.search(r"\d", t) and t not in STREET_STOP and len(t) >= 4
                and t not in REGION_WORDS]
    for p in (a or "").split(","):
        if re.search(r"\d", p):
            return [t for t in _norm(p).split() if not re.search(r"\d", t) and t not in STREET_STOP and len(t) >= 4]
    return None


def _has_street_type(a: str | None) -> bool:
    """The first address component with a digit names a street type (skips 'appt 18' style parts)."""
    for p in (a or "").split(","):
        if re.search(r"\d", p):
            return bool(set(_norm(p).split()) & STREET_TYPES)
    return False


def _numbers(a: str | None) -> list:
    return [int(x) for x in re.findall(r"\d+", _norm(a))][:3]


def different_street(a1: str, a2: str) -> bool:
    """Same first house number, both numbered parts name a street type, and no street word of the
    Source 1 address resembles (ratio >= 0.6) a word of the candidate's numbered part nor any word
    of the candidate's whole address."""
    h1, h2 = _numbers(a1), _numbers(a2)
    if not (h1 and h2 and h1[0] == h2[0] and _has_street_type(a1) and _has_street_type(a2)):
        return False
    w1, w2 = _street_words(a1, True), _street_words(a2, True)
    if not w1 or not w2 or any(_similar(x, y) for x in w1 for y in w2):
        return False
    w2 = _street_words(a2 or "", False)
    return bool(w2) and not any(_similar(x, y) for x in w1 for y in w2)


def _france_pairs(sub: pl.DataFrame, recs: postprocess.TestRecords) -> list:
    s1 = recs.s1_of("france").select("entity_id", "business_name", pl.col("business_address").fill_null(""))
    S = {r[0]: (r[1], r[2]) for r in s1.iter_rows()}
    return [(a, b, *S[a]) for a, m in sub.select(S1, M).iter_rows() if a in S and m for b in m.split(",")]


def street_drops(sub: pl.DataFrame, recs: postprocess.TestRecords) -> pl.DataFrame:
    """E drops: France kept pairs at the Source 1 record's house number on a different street."""
    out = [(a, b) for a, b, _, a1 in _france_pairs(sub, recs) if different_street(a1, recs.addr.get(b, ""))]
    return pl.DataFrame(out, schema={"s1": pl.String, "m": pl.String}, orient="row")


def _sister2_core(n: str | None) -> list:
    s = unicodedata.normalize("NFKD", n or "").encode("ascii", "ignore").decode().lower()
    s = re.sub(r"\bs\.?\s?a\.?\s?r\.?\s?l\b\.?", "sarl", s)
    s = re.sub(r"\bs\.a\.s\.?(u)?\b", r"sas\1", s)
    return [t for t in re.sub(r"[^a-z0-9 ]", " ", s).split() if t not in LEGAL_FORMS]


def sister2_swap(n1: str, n2: str, freq: collections.Counter) -> bool:
    """One core word of n1 replaced, at any position, by a common unrelated France word in n2
    (trailing tail words of n2 ignored)."""
    c1, c2 = _sister2_core(n1), _sister2_core(n2)
    while len(c2) > len(c1) and c2 and c2[-1] in postprocess.TAIL_WORDS:
        c2 = c2[:-1]
    a, b = collections.Counter(c1) - collections.Counter(c2), collections.Counter(c2) - collections.Counter(c1)
    if sum(a.values()) != 1 or sum(b.values()) != 1 or len(c1) < 2:
        return False
    x, y = next(iter(a)), next(iter(b))
    if not (x.isalpha() and y.isalpha() and len(x) >= 3 and len(y) >= 3):
        return False
    if _similar(x, y) or y in postprocess.TAIL_WORDS or {x, y} <= {"cie", "compagnie"}:
        return False
    return freq[x] >= postprocess.SISTER_MIN_FREQ and freq[y] >= postprocess.SISTER_MIN_FREQ


def sister2_drops(sub: pl.DataFrame, recs: postprocess.TestRecords) -> pl.DataFrame:
    """G drops (sister2): France kept pairs that are position-free sister swaps and not street drops."""
    freq = collections.Counter(t for n in recs.raw.filter((pl.col("src") != 1) & (pl.col("country_key") == "france"))
                               ["business_name"].to_list() for t in _sister2_core(n))
    out = [(a, b) for a, b, n1, a1 in _france_pairs(sub, recs)
           if not different_street(a1, recs.addr.get(b, "")) and sister2_swap(n1, recs.name.get(b, ""), freq)]
    return pl.DataFrame(out, schema={"s1": pl.String, "m": pl.String}, orient="row")


def _kept_rows(sub: pl.DataFrame, ids: pl.DataFrame) -> pl.DataFrame:
    return (postprocess.pairs(sub).join(ids.rename({"entity_id": "s1", "row": "i"}), on="s1")
            .join(ids.rename({"entity_id": "m", "row": "j"}), on="m").select("i", "j"))


def _ids(prep: pl.DataFrame, rows: pl.DataFrame) -> pl.DataFrame:
    """rows (i, j) -> + s1, m ids."""
    ids = prep.select("row", "entity_id")
    return (rows.join(ids.rename({"row": "i", "entity_id": "s1"}), on="i")
            .join(ids.rename({"row": "j", "entity_id": "m"}), on="j"))


def _swap_last(a: list, b: list) -> str | None:
    """The word b puts in place of a's last word when exactly that word changed into an unrelated word
    (rel.ntype 'swaplast'), else None."""
    if a == b or not a or not b or sorted(a) == sorted(b) or len(a) != len(b):
        return None
    d = [(x, y) for x, y in zip(a, b) if x != y]
    dis = [(x, y) for x, y in d if Levenshtein.normalized_similarity(x, y) < 0.6]
    if len(dis) == 1 and len(d) == 1 and a[-1] != b[-1]:
        return dis[0][1]
    return None


def _house_int(h: str | None) -> int | None:
    m = re.match(r"\d+", h or "")
    return int(m.group()) if m else None


def _street_ratio(a1: str | None, a2: str | None) -> int:
    return int(fuzz.token_set_ratio(_DIGITS.sub(" ", a1 or ""), _DIGITS.sub(" ", a2 or ""))) if a2 else -1


def _legal(n: str) -> list:
    return sorted({t for t in n.split(" ") if t in LEGAL_FORMS})


def groupe_swap_adds(sub: pl.DataFrame, prep: pl.DataFrame, cand: pl.DataFrame) -> pl.DataFrame:
    """E adds: France (i, j), j a candidate of i or kept, whose core is i's core (2+ words) with the
    last word swapped for 'groupe', raw name ending in ' groupe', same integer house number and
    non-empty address, digit-stripped address token-set ratio >= 85, equal legal forms; pair not
    kept, j kept by nobody, j qualifying for one Source 1 record only. `prep`: prepared test
    records (row, entity_id, src, country_key, name, core, addr, house); `cand`: candidate pairs."""
    fr = prep.filter(pl.col("country_key") == "france")
    kp = _kept_rows(sub, prep.select("row", "entity_id"))
    s1_rows = fr.filter(pl.col("src") == 1)["row"].implode()
    g_rows = fr.filter(pl.col("name").str.ends_with(" groupe"))["row"].implode()
    u = (pl.concat([cand.select("i", "j"), kp]).filter(pl.col("i").is_in(s1_rows) & pl.col("j").is_in(g_rows)).unique()
         .join(fr.select(pl.col("row").alias("i"), n1="name", c1="core", a1="addr", h1="house"), on="i")
         .join(fr.select(pl.col("row").alias("j"), n2="name", c2="core", a2="addr", h2="house"), on="j"))
    keep = []
    for n1, c1, a1, h1, n2, c2, a2, h2 in u.select("n1", "c1", "a1", "h1", "n2", "c2", "a2", "h2").iter_rows():
        w = _swap_last((c1 or "").split(), (c2 or "").split())
        n_1, n_2 = _house_int(h1), _house_int(h2)
        keep.append(w == "groupe" and bool(a2 and a2.strip()) and n_1 is not None and n_1 == n_2
                    and _street_ratio(a1, a2) >= 85 and len(c1.split(" ")) >= 2 and _legal(n1) == _legal(n2))
    c = (u.filter(pl.Series(keep, dtype=pl.Boolean)).join(kp, on=["i", "j"], how="anti")
         .join(kp.select("j").unique(), on="j", how="anti").filter(pl.col("j").count().over("j") == 1))
    return _ids(prep, c.select("i", "j")).select("s1", "m")


def _name_parts(n: str | None) -> tuple:
    t = re.sub(r"[^a-z0-9& ]", " ", n or "").replace("&", " and ").split()
    return ["and" if x == "et" else x for x in t if x not in LEGAL_FORMS], sorted({x for x in t if x in LEGAL_FORMS})


def cie_tailmid_rule(n1: str, n2: str) -> str | None:
    """'cie' (A: cie written out as compagnie) or 'tailmid' (B: a word that is 'and' or not the last
    one replaced by a final 'groupe' / 'france'); None otherwise."""
    (a, la), (b, lb) = _name_parts(n1), _name_parts(n2)
    if la != lb or len(a) != len(b) or len(a) < 2:
        return None
    sa, sb = collections.Counter(a), collections.Counter(b)
    x, z = list((sa - sb).elements()), list((sb - sa).elements())
    if x == ["cie"] and z == ["compagnie"]:
        return "fr_cie"
    if len(x) == 1 and z in (["groupe"], ["france"]) and b[-1] == z[0] and (x[0] == "and" or a[-1] != x[0]):
        return "fr_tailmid"
    return None


def cie_tailmid_adds(sub: pl.DataFrame, prep: pl.DataFrame, active: pl.DataFrame) -> pl.DataFrame:
    """F adds (fr_cie, fr_tailmid): France (i, j), j an active candidate of i, rule above, same address
    (equal address word sets, or equal house and digit-stripped token-set ratio >= 85); pair not
    kept, j kept by nobody and offered to one Source 1 record only. Columns s1, m, rule."""
    fr = prep.filter(pl.col("country_key") == "france")
    kp = _kept_rows(sub, prep.select("row", "entity_id"))
    s1_rows = fr.filter(pl.col("src") == 1)["row"].implode()
    u = (active.select("i", "j").filter(pl.col("i").is_in(s1_rows))
         .join(kp.filter(pl.col("i").is_in(s1_rows)).with_columns(kept=pl.lit(True)), on=["i", "j"], how="full", coalesce=True)
         .with_columns(pl.col("kept").fill_null(False)).filter(~pl.col("kept"))
         .join(fr.select(pl.col("row").alias("i"), n1="name", a1="addr", h1="house"), on="i")
         .join(fr.select(pl.col("row").alias("j"), n2="name", a2="addr", h2="house"), on="j", how="left")
         .filter(pl.col("n2").str.contains("compagnie|groupe|france")))
    t1 = pl.col("a1").fill_null("").str.split(" ").list.unique()
    t2 = pl.col("a2").fill_null("").str.split(" ").list.unique()
    inter = t1.list.set_intersection(t2).list.len()
    u = u.with_columns(ad=pl.when(pl.col("a2").fill_null("").str.strip_chars() == "").then(pl.lit("empty"))
                       .when((inter == t1.list.len()) & (inter == t2.list.len())).then(pl.lit("aset"))
                       .when(pl.col("h1") == pl.col("h2")).then(pl.lit("sameh")).otherwise(pl.lit("other")))
    u = u.filter(pl.col("ad").is_in(["aset", "sameh"]))
    u = u.with_columns(rule=pl.Series([cie_tailmid_rule(a, b) for a, b in zip(u["n1"].to_list(), u["n2"].to_list())],
                                      dtype=pl.String)).filter(pl.col("rule").is_not_null())
    u = u.filter((pl.col("ad") == "aset") | pl.Series([_street_ratio(a, b) >= 85 for a, b in
                                                        zip(u["a1"].to_list(), u["a2"].to_list())], dtype=pl.Boolean))
    u = u.filter(~pl.col("j").is_in(kp["j"].implode())).filter(pl.col("j").is_unique())
    return _ids(prep, u.select("i", "j", "rule")).select("s1", "m", "rule")


def india_copy_adds(sub: pl.DataFrame, prep: pl.DataFrame, cand: pl.DataFrame, recall_pairs: pl.DataFrame,
                    country: str = "india") -> pl.DataFrame:
    """F adds (in_copy): for a kept pair (i, j), every other Source 2/3 record j2 with j's core name
    and (non-empty) address, kept by nobody, never scored for i (neither a candidate nor a recall
    pair) and claimed through one Source 1 record only. Columns s1, m, src (the kept records of s1
    that m copies; the add holds only while one of them is still kept)."""
    rec = prep.select("row", "src", "country_key", key=pl.concat_str("core", "addr", separator="|").hash(),
                      e=pl.col("addr").str.len_chars() == 0)
    B = _kept_rows(sub, prep.select("row", "entity_id"))
    kj = rec.filter((pl.col("src") > 1) & ~pl.col("e")).select(j="row", key="key")
    cop = (B.join(kj, on="j").join(kj.rename({"j": "j2"}), on="key").filter(pl.col("j2") != pl.col("j"))
           .select("i", "j2").unique())
    cop = cop.filter(~pl.col("j2").is_in(B["j"].implode()))
    scored = pl.concat([cand.select("i", pl.col("j").alias("j2")),
                        recall_pairs.select(pl.col("s1_row").alias("i"), pl.col("j_row").alias("j2"))])
    cop = cop.join(scored.filter(pl.col("i").is_in(cop["i"].unique().implode())).unique(), on=["i", "j2"], how="anti")
    cop = cop.filter(pl.col("j2").is_unique()).join(rec.select(i="row", country="country_key"), on="i")
    A = _ids(prep, cop.filter(pl.col("country") == country).select("i", pl.col("j2").alias("j"))).select("s1", "m")
    return with_sources(A, sub, prep)


def with_sources(adds: pl.DataFrame, sub: pl.DataFrame, prep: pl.DataFrame) -> pl.DataFrame:
    """adds (s1, m) + src: the records s1 keeps in `sub` with m's core name and address."""
    rec = prep.filter(pl.col("src") > 1).select("entity_id", key=pl.concat_str("core", "addr", separator="|").hash(),
                                                 e=pl.col("addr").str.len_chars() == 0).filter(~pl.col("e")).drop("e")
    B = postprocess.pairs(sub).rename({"m": "src"})
    S = (adds.join(rec.rename({"entity_id": "m"}), on="m").join(B, on="s1").join(rec.rename({"entity_id": "src", "key": "k2"}), on="src")
         .filter(pl.col("key") == pl.col("k2")).group_by("s1", "m").agg(pl.col("src").unique().sort()))
    return adds.join(S, on=["s1", "m"], how="left", maintain_order="left")


US_STATES = set("al ak az ar ca co ct de fl ga hi id il in ia ks ky la me md ma mi mn ms mo mt ne nv nh nj nm ny nc nd oh ok "
                "or pa ri sc sd tn tx ut vt va wa wv wi wy dc pr".split())


def us_state_conflict_drops(sub: pl.DataFrame, prep: pl.DataFrame) -> pl.DataFrame:
    """G drops (stateconf): US kept pairs whose two prepared addresses both carry state codes and
    share none."""
    us = prep.filter(pl.col("country_key") == "us")
    k = (_kept_rows(sub, us.select("row", "entity_id"))
         .join(us.select(pl.col("row").alias("i"), ai="addr"), on="i").join(us.select(pl.col("row").alias("j"), aj="addr"), on="j"))
    hit = [bool(si and sj and not (si & sj)) for si, sj in
           (({w for w in (a or "").split() if w in US_STATES}, {w for w in (b or "").split() if w in US_STATES})
            for a, b in zip(k["ai"].to_list(), k["aj"].to_list()))]
    return _ids(prep, k.filter(pl.Series(hit, dtype=pl.Boolean)).select("i", "j")).select("s1", "m")


# France regions named in addresses (as region or department) and the cities of the test pool that imply one.
REGIONS = {"nouvelle aquitaine": "NA", "gironde": "NA", "hauts de france": "HF", "nord": "HF", "pas de calais": "HF",
           "pays de la loire": "PL", "loire atlantique": "PL"}
REGION_CITIES = {"bordeaux": "NA", "pessac": "NA", "merignac": "NA", "la teste de buch": "NA", "lege cap ferret": "NA",
                 "nantes": "PL", "saint nazaire": "PL", "pornic": "PL", "saint herblain": "PL", "la baule escoublac": "PL",
                 "lille": "HF", "tourcoing": "HF", "dunkerque": "HF", "roubaix": "HF", "calais": "HF"}
_REGION_WORDS = {"nouvelle aquitaine", "hauts de france", "pays de la loire", "gironde", "nord", "pas de calais",
                 "loire atlantique", "france"}


def _fold(a: str) -> str:
    return "".join(ch for ch in unicodedata.normalize("NFKD", a) if not unicodedata.combining(ch)).lower()


def region(a: str | None) -> str | None:
    """Region of a France address: the one region its comma parts name, else the region of its city
    (last comma part that is not a region and has no digit)."""
    if not a:
        return None
    s = _fold(a)
    named = {REGIONS[x] for x in (re.sub(r"[\-\.]", " ", c).strip() for c in s.split(",")) if x in REGIONS}
    if len(named) == 1:
        return named.pop()
    cities = []
    for c in s.split(","):
        c = re.sub(r"[\-\.']", " ", c)
        c = re.sub(r"\bst\b", "saint", c)
        c = re.sub(r"\bste\b", "sainte", c)
        c = re.sub(r"\s+", " ", re.sub(r"\d{5}", "", c)).strip()
        if c and c not in _REGION_WORDS and not re.search(r"\d", c):
            cities.append(c)
    return REGION_CITIES.get(cities[-1]) if cities else None


def region_conflict_drops(sub: pl.DataFrame, recs: postprocess.TestRecords) -> pl.DataFrame:
    """H drops: France kept pairs whose two addresses sit in different regions."""
    out = []
    for a, b, _, a1 in _france_pairs(sub, recs):
        r1, r2 = region(a1), region(recs.addr.get(b, ""))
        if r1 and r2 and r1 != r2:
            out.append((a, b))
    return pl.DataFrame(out, schema={"s1": pl.String, "m": pl.String}, orient="row")


def derive(step: str, sub: pl.DataFrame, recs: postprocess.TestRecords, prep: pl.DataFrame, cand: pl.DataFrame,
           active: pl.DataFrame, recall_pairs: pl.DataFrame, shipped: dict) -> tuple:
    """(drops, adds, sources) of one step computed on the current frame; lists without code come
    from `shipped` ({(step, action): frame})."""
    empty = pl.DataFrame(schema={"s1": pl.String, "m": pl.String})
    if step == "E":
        return street_drops(sub, recs), groupe_swap_adds(sub, prep, cand), None
    if step == "F":
        copies = india_copy_adds(sub, prep, cand, recall_pairs)
        adds = pl.concat([cie_tailmid_adds(sub, prep, active).select("s1", "m"), copies.select("s1", "m")])
        return empty, adds, copies.filter(pl.col("src").is_not_null())
    if step == "G":
        rest = shipped["G", "drop"].filter(~pl.col("rule").is_in(["sister2", "stateconf"])).select("s1", "m")
        return (pl.concat([sister2_drops(sub, recs), us_state_conflict_drops(sub, prep), rest]),
                shipped["G", "add"].select("s1", "m"), None)
    if step == "H":
        rest = shipped["H", "drop"].filter(pl.col("rule") != "fr_region_conflict").select("s1", "m")
        return pl.concat([region_conflict_drops(sub, recs), rest]), shipped["H", "add"].select("s1", "m"), None
    return shipped[step, "drop"], shipped[step, "add"], None
