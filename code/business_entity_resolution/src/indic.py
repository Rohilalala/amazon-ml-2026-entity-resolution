"""Indic-script -> English word dictionary, learned from the training ground truth.

Indian Source 2/3 names written in Devanagari, Bengali, Kannada, Malayalam, ...
are word-by-word transliterations of their Source 1 name ("जय हेल्थकेयर" ->
"Jai Healthcare"). Generic transliteration (anyascii) gets close but not equal
("jy helthkeyr"), so for every training pair whose Source 2/3 name contains
Indic words we align those words, in order, with the Source 1 words the pair
does not already share, and keep the majority English word per Indic word.

Indic runs in addresses are state names; each run votes for the Indian state
code found in its Source 1 address and maps to that state's English name.

Only the provided training data is used.
"""
import re
from collections import Counter, defaultdict

import polars as pl

from . import text

_INDIC_CHAR = re.compile(r"[ऀ-෿]")
_INDIC_RUN = re.compile(r"[ऀ-෿‌‍]+(?:\s+[ऀ-෿‌‍]+)*")
_FIRST_INDIAN_STATE = "andhra pradesh"


def _indian_states() -> dict:
    names = list(text.STATES)
    return {k: text.STATES[k] for k in names[names.index(_FIRST_INDIAN_STATE):]}


def learn_dictionary(raw: pl.DataFrame, truth: pl.DataFrame) -> dict:
    """Map Indic words (and whole Indic address runs) to English words.

    `raw` is the loaded split (row, src, business_name, business_address);
    `truth` holds true pairs as rows (i: Source 1, j: Source 2/3).
    """
    cols = ["row", "business_name", "business_address"]
    pairs = (truth.join(raw.select(pl.col(c).alias(f"{c}_1") for c in cols), left_on="i", right_on="row_1")
             .join(raw.select(pl.col(c).alias(f"{c}_2") for c in cols), left_on="j", right_on="row_2"))
    out = {}

    # Names: align the Indic words of the Source 2/3 name with the Source 1 words it lacks.
    names = pairs.filter(pl.col("business_name_2").str.contains(r"[\x{0900}-\x{0DFF}]"))
    votes = defaultdict(Counter)
    s1_tokens = text.clean(names["business_name_1"]).str.split(" ")
    s2_tokens = names["business_name_2"].fill_null("").str.to_lowercase().str.extract_all(r"\w+")
    for t1, t2 in zip(s1_tokens, s2_tokens):
        indic = [t for t in t2 if _INDIC_CHAR.search(t)]
        latin = Counter(t for t in t2 if not _INDIC_CHAR.search(t))
        rest = []
        for w in t1:
            if latin[w] > 0:
                latin[w] -= 1
            else:
                rest.append(w)
        if len(indic) == len(rest):
            for t, w in zip(indic, rest):
                votes[t][w] += 1
    for t, c in votes.items():
        out[t] = c.most_common(1)[0][0]

    # Addresses: each Indic run votes for the state codes in its Source 1 address.
    states = _indian_states()
    codes = sorted(set(states.values()))
    name_of = {}
    for name, code in states.items():
        name_of.setdefault(code, name)
    addrs = pairs.filter(pl.col("business_address_2").str.contains(r"[\x{0900}-\x{0DFF}]"))
    s1_codes = text.clean_address(addrs["business_address_1"]).str.split(" ").list.eval(
        pl.element().filter(pl.element().is_in(codes)))
    run_votes = defaultdict(Counter)
    for a2, cs in zip(addrs["business_address_2"], s1_codes):
        for run in _INDIC_RUN.findall(a2):
            run_votes[run].update(set(cs))
    for run, c in run_votes.items():
        if c:
            out[run] = name_of[c.most_common(1)[0][0]]
    return out
