"""Label-free corrections, applied after the decision, for a country absent from training.

A model that never saw a country's vocabulary misreads some of the test generator's
patterns there. Three corrections use only names, house numbers and streets:

1. Legal-form conflict drop: a kept pair at a nearby house number (one edit, or within
   20) whose names carry different legal forms (sarl vs sas) is a sister company when
   the Source 1 record already keeps a record at its own house number.
2. True-variant word add: a candidate whose core name is the Source 1 core plus one of
   TRUE_VARIANT_WORDS, at the same house number and without conflicting legal forms.
3. Generator-tail add: a candidate named like the Source 1 record with its last core
   word replaced by a tail (swap) or with a tail appended (append), at the same house
   number and street.

Adds keep the exclusive owner rule: the Source 2/3 record is kept by no Source 1 record,
the pair is its most likely owner, and the Source 1 record already keeps a pair.
"""
import numpy as np
import polars as pl
from rapidfuzz import process
from rapidfuzz.distance import Levenshtein

COLUMNS = ("name", "core", "addr", "house")
LEGAL_FORMS = ["private", "limited", "llp", "llc", "lp", "plc", "incorporated", "corporation", "company", "companies",
               "sarl", "sas", "sasu", "sa", "eurl", "snc", "sci", "inc", "corp", "co", "ltd", "pvt"]
# Words that, among the test pool's candidates whose core name adds exactly one word to
# the Source 1 core, occur at least 1,000 times with at most 45% at a different house
# and at least 40% at the same house: variants of the same business, not decoys.
TRUE_VARIANT_WORDS = ["fils", "associes", "cie", "services"]
TRUE_VARIANT_MIN_PROB = 0.15
# Name tails the generator uses for true copies, found at equal house and street with
# near-uniform counts. The model rejects most of them in a country it never saw.
SWAP_TAILS = ["and associes", "et associes", "associes", "france"]
APPEND_TAILS = ["groupe", "france", "and associes", "et associes", "associes"]
# Street types, articles and directions ignored when comparing street words.
STREET_STOP = ["rue", "avenue", "boulevard", "place", "chemin", "impasse", "route", "allee", "cours", "quai", "passage",
               "square", "faubourg", "rondpoint", "residence", "lieu", "dit", "zone", "za", "zi", "zac", "bis", "ter",
               "cedex", "de", "la", "le", "les", "du", "des", "d", "l", "street", "road", "lane", "drive", "court",
               "suite", "floor", "building", "apartment", "near", "opposite", "nagar", "sector", "colony", "highway",
               "parkway", "circle", "terrace", "trail", "way", "north", "south", "east", "west", "n", "s", "e", "w",
               "unit", "no"]


def house_relation(hi: pl.Series, hj: pl.Series, aj: pl.Series) -> np.ndarray:
    """no_addr, no_house, eq, substr, edit1 (one edit), off20 (numbers within 20) or far."""
    def num(h):
        return h.str.extract(r"(\d+)", 1).str.slice(0, 9).cast(pl.Int64, strict=False)
    lev = process.cpdist(hi.to_list(), hj.to_list(), scorer=Levenshtein.distance, workers=-1, dtype=np.int32)
    d = pl.DataFrame({"hi": hi, "hj": hj, "aj": aj, "lev": lev, "nd": (num(hi) - num(hj)).abs()})
    return d.select(pl.when(pl.col("aj") == "").then(pl.lit("no_addr"))
                    .when((pl.col("hi") == "") | (pl.col("hj") == "")).then(pl.lit("no_house"))
                    .when(pl.col("hi") == pl.col("hj")).then(pl.lit("eq"))
                    .when(pl.col("hi").str.contains(pl.col("hj"), literal=True)
                          | pl.col("hj").str.contains(pl.col("hi"), literal=True)).then(pl.lit("substr"))
                    .when(pl.col("lev") <= 1).then(pl.lit("edit1"))
                    .when(pl.col("nd") <= 20).then(pl.lit("off20"))
                    .otherwise(pl.lit("far")))[:, 0].to_numpy()


def legal_form_conflict(ni: pl.Series, nj: pl.Series) -> np.ndarray:
    """Both names carry legal forms and the sets differ."""
    def forms(n):
        return (n.str.to_lowercase().str.replace_all(r"[^a-z0-9 ]", " ").str.split(" ")
                .list.eval(pl.element().filter(pl.element().is_in(LEGAL_FORMS))).list.unique().list.sort().list.join(" "))
    a, b = forms(ni), forms(nj)
    return ((a != b) & (a != "") & (b != "")).to_numpy()


def same_street(ai: pl.Series, aj: pl.Series) -> np.ndarray:
    """Street words (two letters or more, no digit, minus STREET_STOP) equal, one side a
    subset of the other, or differing by one word each with a typo (similarity >= 0.75)."""
    def words(a):
        return (a.str.split(" ").list.eval(pl.element().filter((pl.element().str.len_chars() >= 2)
                                                               & ~pl.element().str.contains(r"\d")))
                .list.unique().list.set_difference(pl.lit(STREET_STOP)))
    d = pl.DataFrame({"a": ai, "b": aj}).select(words(pl.col("a")).alias("a"), words(pl.col("b")).alias("b"))
    d = d.with_columns(pl.col("a").list.set_difference("b").alias("x1"), pl.col("b").list.set_difference("a").alias("x2"))
    l1, l2 = d["x1"].list.len().to_numpy(), d["x2"].list.len().to_numpy()
    sim = process.cpdist(d["x1"].list.first().fill_null("").to_list(), d["x2"].list.first().fill_null("").to_list(),
                         scorer=Levenshtein.normalized_similarity, workers=-1, dtype=np.float32)
    both = (d["a"].list.len() > 0).to_numpy() & (d["b"].list.len() > 0).to_numpy()
    return both & ((l1 == 0) | (l2 == 0) | ((l1 == 1) & (l2 == 1) & (sim >= 0.75)))


def _tail_match(n1: str, c1: str, n2: str) -> bool:
    t1, core = n1.split(" "), c1.split(" ")
    last = max((q for q, w in enumerate(t1) if w == core[-1]), default=-1)
    base = " ".join(w for q, w in enumerate(t1) if q != last)
    if n2.startswith(n1 + " ") and n2[len(n1) + 1:] in APPEND_TAILS:
        return True
    return n2.startswith(base + " ") and n2[len(base) + 1:] in SWAP_TAILS and len(core) >= 2


def correct(recs: pl.DataFrame, i: np.ndarray, j: np.ndarray, prob: np.ndarray, keep: np.ndarray
            ) -> tuple[np.ndarray, dict]:
    """Corrected keep flags for one unseen country's candidate pairs. `i`, `j` index the
    rows of `recs` (which has COLUMNS); `prob` is the probability the country is decided
    on and `keep` the decision."""
    def col(c, idx):
        return recs[c].gather(idx)
    keep = keep.copy()
    k = np.flatnonzero(keep)
    hrel = house_relation(col("house", i[k]), col("house", j[k]), col("addr", j[k]))
    anchored = np.isin(i[k], i[k][hrel == "eq"])
    drop = k[np.isin(hrel, ["edit1", "off20"]) & anchored & legal_form_conflict(col("name", i[k]), col("name", j[k]))]
    keep[drop] = False
    top = pl.DataFrame({"j": j, "p": prob}).select(pl.col("p").max().over("j"))["p"].to_numpy()

    def free(keep):
        """Unkept pairs that are their Source 2/3 record's most likely owner, whose Source 2/3
        record nobody keeps and whose Source 1 record keeps a pair."""
        s1, s23 = np.zeros(len(recs), dtype=bool), np.zeros(len(recs), dtype=bool)
        s1[i[keep]], s23[j[keep]] = True, True
        return s1[i] & ~s23[j] & (prob >= top)

    c = np.flatnonzero(free(keep) & (prob >= TRUE_VARIANT_MIN_PROB))
    ti, tj = col("core", i[c]).str.split(" "), col("core", j[c]).str.split(" ")
    extra = tj.list.set_difference(ti)
    one_word = ((tj.list.unique().list.len() == ti.list.unique().list.len() + 1) & (extra.list.len() == 1)
                & (ti.list.set_difference(tj).list.len() == 0) & extra.list.first().is_in(TRUE_VARIANT_WORDS))
    c = c[one_word.fill_null(False).to_numpy()]
    c = c[(house_relation(col("house", i[c]), col("house", j[c]), col("addr", j[c])) == "eq")
          & ~legal_form_conflict(col("name", i[c]), col("name", j[c]))]
    tv = c[np.unique(j[c], return_index=True)[1]]
    keep[tv] = True

    tails = "(?: " + "| ".join(sorted(set(SWAP_TAILS + APPEND_TAILS), key=len, reverse=True)) + ")$"
    c = np.flatnonzero(free(keep))
    c = c[col("name", j[c]).str.contains(tails).to_numpy()]
    c = c[np.array([_tail_match(a, b, d) for a, b, d in zip(col("name", i[c]).to_list(), col("core", i[c]).to_list(),
                                                            col("name", j[c]).to_list())], dtype=bool)]
    c = c[(house_relation(col("house", i[c]), col("house", j[c]), col("addr", j[c])) == "eq")
          & same_street(col("addr", i[c]), col("addr", j[c]))]
    tail = c[np.unique(j[c], return_index=True)[1]]
    keep[tail] = True
    return keep, {"legal_form_drops": len(drop), "true_variant_adds": len(tv), "tail_adds": len(tail)}
