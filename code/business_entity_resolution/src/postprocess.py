"""Post-decision corrections at the id level, applied in a fixed order to the submission frame.

The frame has one row per test Source 1 record, in test_source1 order, and two columns:
`source1_entity_id` and `matched_entity_ids` (comma-joined, "" when empty). Every step keeps the
row order and the order of the ids inside a row (kept ids first, added ids appended), so the
same inputs always give the same bytes.

Steps (the leaderboard run, `final.py`):
  C  sister_filter        France sister-organisation decoys: one core word swapped for a common,
                          unrelated word (v2 rule).
  D  restore_scrambles    re-add C's drops that are generator typo scrambles or abbreviations.
  E-H apply_lists         pair lists of the later France / India / US rules (lists.parquet in
                          the bundle; see rules.py for the code that derives each list).
Shared semantics of every add: applied only if the Source 1 row exists and the candidate is kept
by no Source 1 record in the current frame; drops apply where present.
"""
import collections
import difflib
import re
import unicodedata

import polars as pl

S1, M = "source1_entity_id", "matched_entity_ids"
# Legal forms ignored when comparing French names, and the tail words the test generator appends
# to true copies (never a sister-organisation swap).
LEGAL_FORMS = {"sarl", "sas", "sasu", "sa", "eurl", "snc", "sci", "scop", "selarl"}
TAIL_WORDS = {"fils", "cie", "compagnie", "services", "developpement", "groupe", "france", "associes", "et", "and"}
SISTER_MIN_FREQ = 20  # both swapped words occur at least this often as France name words
SISTER_MAX_SIMILARITY = 0.6  # SequenceMatcher ratio below which a swapped word is not a typo
SCRAMBLE_MAX_FREQ = 100  # a restored word is rarer than this in France (a generator scramble)


def frame(ids: pl.Series, lists: pl.Series) -> pl.DataFrame:
    """Submission frame from Source 1 ids and their comma-joined match lists."""
    return pl.DataFrame({S1: ids, M: lists.fill_null("")})


def pairs(sub: pl.DataFrame) -> pl.DataFrame:
    """(s1, m) rows of a submission frame, in frame order."""
    return (sub.select(pl.col(S1).alias("s1"), pl.col(M).fill_null("").str.split(",").alias("m"))
            .explode("m").filter(pl.col("m") != ""))


def _rebuild(sub: pl.DataFrame, ex: pl.DataFrame) -> pl.DataFrame:
    """Frame with `sub`'s rows and the (s1, m) rows of `ex`, grouped in `ex` order."""
    out = (sub.select(S1).join(ex.group_by("s1", maintain_order=True).agg(pl.col("m").str.join(","))
                               .rename({"s1": S1, "m": M}), on=S1, how="left", maintain_order="left")
           .with_columns(pl.col(M).fill_null("")))
    assert out.height == sub.height and out[S1].equals(sub[S1])
    return out


def _norm(s: str | None) -> str:
    """ASCII-folded, lowercased text; characters other than a-z, 0-9 and space become spaces."""
    s = unicodedata.normalize("NFKD", s or "").encode("ascii", "ignore").decode().lower()
    return re.sub(r"[^a-z0-9 ]", " ", s)


def core_words(name: str | None) -> list[str]:
    """ASCII-folded, lowercased name words without French legal forms."""
    return [t for t in _norm(name).split() if t not in LEGAL_FORMS]


def _house(address: str | None) -> str | None:
    """First number of an address without leading zeros, or None."""
    m = re.search(r"\b(\d+)\b", _norm(address))
    return str(int(m.group(1))) if m else None


class TestRecords:
    """Names, addresses and countries of the test records, by entity id (data.load_split order)."""

    def __init__(self, raw: pl.DataFrame):
        """Index names and addresses by entity id and count France name words over Source 2/3."""
        country = raw["country"].str.strip_chars().str.to_lowercase()
        self.raw = raw.with_columns(country_key=country)
        s23 = self.raw.filter(pl.col("src") != 1)
        self.name = dict(zip(s23["entity_id"].to_list(), s23["business_name"].to_list()))
        self.addr = dict(zip(s23["entity_id"].to_list(), s23["business_address"].fill_null("").to_list()))
        s1 = self.raw.filter(pl.col("src") == 1)
        self.s1_name = dict(zip(s1["entity_id"].to_list(), s1["business_name"].to_list()))
        fr = s23.filter(pl.col("country_key") == "france")
        # France name-word frequencies over the Source 2/3 pool.
        self.fr_freq = collections.Counter(t for n in fr["business_name"].to_list() for t in core_words(n))

    def s1_of(self, country: str) -> pl.DataFrame:
        """Source 1 records of one country."""
        return self.raw.filter((pl.col("src") == 1) & (pl.col("country_key") == country))


def _sister_swap(c1: list, c2: list, freq: collections.Counter) -> tuple | None:
    """The one swapped word pair when c2 is c1 with one word replaced by a common, unrelated word."""
    if len(c1) != len(c2) or len(c1) < 2:
        return None
    d = [(a, b) for a, b in zip(c1, c2) if a != b]
    if len(d) != 1:
        return None
    a, b = d[0]
    if not (a.isalpha() and b.isalpha() and len(a) >= 3 and len(b) >= 3):
        return None
    if difflib.SequenceMatcher(None, a, b).ratio() >= SISTER_MAX_SIMILARITY or b in TAIL_WORDS or {a, b} <= {"cie", "compagnie"}:
        return None
    if freq[a] < SISTER_MIN_FREQ or freq[b] < SISTER_MIN_FREQ:
        return None
    return a, b


def sister_filter(sub: pl.DataFrame, recs: TestRecords) -> tuple[pl.DataFrame, dict]:
    """Step C. Drop a France kept pair whose candidate name is the Source 1 name with one core word
    replaced by a different, common France name word that is not a typo of it (similarity < 0.6),
    not a generator tail word and not cie <-> compagnie; the comparison also tries the candidate
    with trailing tail words removed. A record whose kept pairs are all such swaps is emptied: it
    looks like a true singleton that only matched sister organisations."""
    s1 = recs.s1_of("france").select(pl.col("entity_id"), "business_name")
    fr = sub.rename({S1: "entity_id"}).join(s1, on="entity_id")
    drops, n_emptied = {}, 0
    for r in fr.iter_rows(named=True):
        ids = [x for x in (r[M] or "").split(",") if x]
        if not ids:
            continue
        c1 = core_words(r["business_name"])
        bad = []
        for i in ids:
            c2 = core_words(recs.name.get(i, ""))
            s = _sister_swap(c1, c2, recs.fr_freq)
            if s is None:
                c2t = list(c2)
                while len(c2t) > len(c1) and c2t and c2t[-1] in TAIL_WORDS:
                    c2t.pop()
                s = _sister_swap(c1, c2t, recs.fr_freq)
            if s:
                bad.append(i)
        if bad:
            drops[r["entity_id"]] = set(bad)
            n_emptied += len(bad) == len(ids)
    out = sub.with_columns(pl.struct(S1, M).map_elements(
        lambda s: ",".join(x for x in (s[M] or "").split(",") if x and x not in drops.get(s[S1], ())),
        return_dtype=pl.String).alias(M))
    return out, {"drop_pairs": sum(len(v) for v in drops.values()), "drop_s1": len(drops), "emptied_s1": n_emptied}


def _scramble(recs: TestRecords, s: str, m: str) -> bool:
    """Step D test: exactly one core word differs, and the candidate's word is a rare word with the
    same first letter (a generator scramble) or one word is a prefix of the other."""
    d = [(a, b) for a, b in zip(core_words(recs.s1_name.get(s, "")), core_words(recs.name.get(m, ""))) if a != b]
    if len(d) != 1:
        return False
    a, b = d[0]
    return (recs.fr_freq[b] < SCRAMBLE_MAX_FREQ and a[:1] == b[:1]) or bool(a and b and (a.startswith(b) or b.startswith(a)))


def restore_scrambles(pre: pl.DataFrame, post: pl.DataFrame, recs: TestRecords) -> tuple[pl.DataFrame, dict]:
    """Step D. Re-add pairs that `pre` kept and `post` (the sister filter's output) dropped when
    exactly one core word differs and the candidate's word is a generator typo scramble (rare in
    France, same first letter) or one word abbreviates the other, if the candidate is unclaimed."""
    a, b = pairs(pre).rename({"s1": "s"}), pairs(post).rename({"s1": "s"})
    dropped = a.join(b, on=["s", "m"], how="anti")
    claimed = set(b["m"].to_list())
    add = [(s, m) for s, m in dropped.iter_rows() if _scramble(recs, s, m) and m not in claimed]
    ex = pl.concat([b, pl.DataFrame({"s": [x for x, _ in add], "m": [y for _, y in add]},
                                    schema={"s": pl.String, "m": pl.String})])
    assert ex["m"].n_unique() == ex.height
    return _rebuild(post, ex.rename({"s": "s1"})), {"sister_drops": dropped.height, "restored": len(add)}


def apply_lists(sub: pl.DataFrame, drops: pl.DataFrame, adds: pl.DataFrame, sources: pl.DataFrame | None = None,
                unique_adds: bool = False) -> tuple[pl.DataFrame, dict]:
    """Steps E-G. Remove `drops` (s1, m) where present; append each add (s1, m), in list order, when
    its Source 1 row exists and m is kept by no Source 1 record. `sources` (s1, m, src list): an add
    listed there also needs the frame to still keep one of its source records for that s1 (India
    copies). `unique_adds`: keep only the first add of each m."""
    b = pairs(sub)
    assert b["m"].is_unique().all(), "a candidate is kept twice"
    kept = b.join(drops.select("s1", "m"), on=["s1", "m"], how="anti")
    adds = adds.select("s1", "m")
    if sources is None:
        ok = adds.filter(~pl.col("m").is_in(kept["m"].implode()) & pl.col("s1").is_in(sub[S1].implode()))
        if unique_adds:
            ok = ok.unique(subset=["m"], keep="first")
    else:
        ok = adds.filter(pl.col("s1").is_in(sub[S1].implode()) & ~pl.col("m").is_in(b["m"].implode()))
        src = sources.explode("src")
        held = src.join(b.rename({"m": "src"}), on=["s1", "src"]).select("s1", "m").unique()
        ok = pl.concat([ok.join(src.select("s1", "m").unique(), on=["s1", "m"], how="anti"), ok.join(held, on=["s1", "m"])])
    ex = pl.concat([kept, ok])
    assert ex["m"].is_unique().all(), "a candidate is kept by two Source 1 records"
    return _rebuild(sub, ex), {"dropped": b.height - kept.height, "added": ok.height}


def apply_late(sub: pl.DataFrame, drops: pl.DataFrame, adds: pl.DataFrame) -> tuple[pl.DataFrame, dict]:
    """Step H: apply_lists semantics with plain per-row lists (adds appended to their row)."""
    rows = sub[S1].to_list()
    ms = [[m for m in (x or "").split(",") if m] for x in sub[M].to_list()]
    drop = set(drops.select("s1", "m").iter_rows())
    kept = {m for lst in ms for m in lst}
    pos = {s: k for k, s in enumerate(rows)}
    nd = na = 0
    for k, s in enumerate(rows):
        new = [m for m in ms[k] if (s, m) not in drop]
        nd += len(ms[k]) - len(new)
        ms[k] = new
    for s, m in adds.select("s1", "m").iter_rows():
        if s in pos and m not in kept:
            ms[pos[s]].append(m)
            kept.add(m)
            na += 1
    flat = [m for lst in ms for m in lst]
    assert len(flat) == len(set(flat)), "a candidate is kept twice"
    return pl.DataFrame({S1: rows, M: [",".join(lst) for lst in ms]}), {"dropped": nd, "added": na}
