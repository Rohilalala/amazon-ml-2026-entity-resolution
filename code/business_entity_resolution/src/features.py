"""Record preparation and pair features for the matching model.

No feature encodes the country itself, so the model transfers to a country
it has not seen in training (France in the test set).
"""
import numpy as np
import pandas as pd
from rapidfuzz import fuzz, process
from rapidfuzz.distance import JaroWinkler

from . import text
from .candidates import pair_cosine, tfidf


def prepare(recs):
    """Add normalised text columns and token/number sets to raw source records."""
    recs = recs.copy()
    recs["name"] = recs["business_name"].map(text.clean_name)
    recs["core"] = recs["name"].map(text.core_name)
    recs["addr"] = recs["business_address"].map(text.clean_address)
    recs["full"] = recs["name"] + " " + recs["addr"]
    recs["country_key"] = recs["country"].map(lambda c: text.to_ascii(c).strip())
    for c in ("name", "core", "addr"):
        recs[f"{c}_tok"] = recs[c].map(lambda s: frozenset(s.split()))
    recs["nums"] = recs["addr"].map(text.numbers)
    recs["posts"] = recs["nums"].map(text.postcodes)
    recs["house"] = recs["addr"].map(text.house_number)
    recs["acr"] = recs["core"].map(text.acronym)
    return recs


def matrices(recs):
    """TF-IDF matrices used for blocking and cosine features, row-aligned with recs."""
    return {
        "name": tfidf(recs["name"]),
        "core": tfidf(recs["core"]),
        "word": tfidf(recs["name"], analyzer="word", ngram_range=(1, 1)),
        "addr": tfidf(recs["addr"]),
        "full": tfidf(recs["full"]),
    }


def _fuzzy(a, b, scorer):
    """Element-wise rapidfuzz score in [0, 1] for two equal-length string lists."""
    return process.cpdist(a, b, scorer=scorer, workers=-1, dtype=np.float32) / 100.0


def _jaccard(A, B):
    """Element-wise Jaccard of two sequences of sets; NaN when both are empty."""
    out = np.full(len(A), np.nan, dtype=np.float32)
    for n, (x, y) in enumerate(zip(A, B)):
        union = len(x | y)
        if union:
            out[n] = len(x & y) / union
    return out


def _overlap_or_nan(A, B):
    """1 if two sets share an element, 0 if both are non-empty but disjoint, else NaN."""
    return np.array([float(bool(x & y)) if x and y else np.nan for x, y in zip(A, B)],
                    dtype=np.float32)


def _dba_best(a, b, base):
    """Best token-set score over 'X dba Y' name parts; `base` where no side has a DBA."""
    out = base.copy()
    for n in np.flatnonzero([(" dba " in x) or (" dba " in y) for x, y in zip(a, b)]):
        pa, pb = a[n].split(" dba "), b[n].split(" dba ")
        out[n] = max(fuzz.token_set_ratio(x, y) for x in pa for y in pb) / 100.0
    return out


def _acronym_match(acr_a, acr_b, core_a, core_b):
    """1 when one name's initials equal the other name (or one of its tokens)."""
    def hit(acr, core):
        return len(acr) >= 2 and (acr == core.replace(" ", "") or acr in core.split())
    return np.array([float(hit(x, cb) or hit(y, ca))
                     for x, y, ca, cb in zip(acr_a, acr_b, core_a, core_b)], dtype=np.float32)


def build(recs, mats, pairs):
    """Feature table for candidate pairs (row positions i -> Source 1, j -> Source 2/3)."""
    i, j = pairs["i"].to_numpy(), pairs["j"].to_numpy()
    col = {c: recs[c].to_numpy() for c in
           ["name", "core", "addr", "name_tok", "core_tok", "addr_tok",
            "nums", "posts", "house", "acr", "src"]}
    g = {c: (v[i], v[j]) for c, v in col.items()}
    f = pd.DataFrame({"s1": recs["entity_id"].to_numpy()[i],
                      "cand": recs["entity_id"].to_numpy()[j]})

    for m, X in mats.items():
        f[f"cos_{m}"] = pair_cosine(X, i, j)

    na, nb = g["name"][0].tolist(), g["name"][1].tolist()
    ca, cb = g["core"][0].tolist(), g["core"][1].tolist()
    aa, ab = g["addr"][0].tolist(), g["addr"][1].tolist()
    f["fz_name_ratio"] = _fuzzy(na, nb, fuzz.ratio)
    f["fz_name_partial"] = _fuzzy(na, nb, fuzz.partial_ratio)
    f["fz_name_sort"] = _fuzzy(na, nb, fuzz.token_sort_ratio)
    f["fz_name_set"] = _fuzzy(na, nb, fuzz.token_set_ratio)
    f["jw_name"] = _fuzzy(na, nb, JaroWinkler.normalized_similarity) * 100.0
    f["fz_core_ratio"] = _fuzzy(ca, cb, fuzz.ratio)
    f["fz_core_partial"] = _fuzzy(ca, cb, fuzz.partial_ratio)
    f["fz_core_set"] = _fuzzy(ca, cb, fuzz.token_set_ratio)
    f["jw_core"] = _fuzzy(ca, cb, JaroWinkler.normalized_similarity) * 100.0
    f["fz_addr_set"] = _fuzzy(aa, ab, fuzz.token_set_ratio)
    f["fz_addr_partial"] = _fuzzy(aa, ab, fuzz.partial_ratio)
    f["fz_addr_sort"] = _fuzzy(aa, ab, fuzz.token_sort_ratio)
    f["fz_dba"] = _dba_best(g["name"][0], g["name"][1], f["fz_name_set"].to_numpy())

    f["jac_name"] = _jaccard(*g["name_tok"])
    f["jac_core"] = _jaccard(*g["core_tok"])
    f["jac_addr"] = _jaccard(*g["addr_tok"])
    f["jac_num"] = _jaccard(*g["nums"])
    f["post_match"] = _overlap_or_nan(*g["posts"])
    ha, hb = g["house"]
    f["house_match"] = np.where((ha == "") | (hb == ""), np.nan, (ha == hb)).astype(np.float32)
    f["acr_match"] = _acronym_match(*g["acr"], ca, cb)

    f["len_name_1"] = np.fromiter(map(len, na), np.float32, len(na))
    f["len_name_2"] = np.fromiter(map(len, nb), np.float32, len(nb))
    f["len_addr_1"] = np.fromiter(map(len, aa), np.float32, len(aa))
    f["len_addr_2"] = np.fromiter(map(len, ab), np.float32, len(ab))
    f["is_s3"] = (g["src"][1] == 3).astype(np.float32)

    # Context: how this pair ranks among the other candidates of the same
    # Source 1 record, and among the Source 1 records competing for the same
    # Source 2/3 record (Source 1 is deduplicated, so there should be one winner).
    for c in ["cos_name", "cos_full", "fz_core_set", "fz_addr_set"]:
        for key, tag in (("s1", "s1"), ("cand", "c")):
            grp = f.groupby(key)[c]
            f[f"rank_{tag}_{c}"] = grp.rank(ascending=False, method="min").astype(np.float32)
            f[f"gap_{tag}_{c}"] = (grp.transform("max") - f[c]).astype(np.float32)
    f["n_cand_s1"] = f.groupby("s1")["cand"].transform("size").astype(np.float32)
    f["n_s1_cand"] = f.groupby("cand")["s1"].transform("size").astype(np.float32)
    return f
