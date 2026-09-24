"""Candidate generation (blocking).

For every Source 1 record we keep the k nearest Source 2 and Source 3 records
under TF-IDF character n-gram cosine, once on the name and once on
name + address. A reverse pass (nearest Source 1 records for each Source 2/3
record) catches matches that rank low from the Source 1 side. Search runs
within a country; countries are treated as open string labels.
"""
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.neighbors import NearestNeighbors


def tfidf(texts, analyzer="char_wb", ngram_range=(2, 4)):
    """L2-normalised TF-IDF matrix, fitted on the texts it transforms."""
    vec = TfidfVectorizer(analyzer=analyzer, ngram_range=ngram_range,
                          sublinear_tf=True, dtype=np.float32)
    return vec.fit_transform(texts)


def nearest(Xq, Xi, k):
    """Indices (into Xi) of the k most cosine-similar rows for each row of Xq."""
    k = min(k, Xi.shape[0])
    nn = NearestNeighbors(n_neighbors=k, metric="cosine", algorithm="brute", n_jobs=-1)
    return nn.fit(Xi).kneighbors(Xq, return_distance=False)


def generate(recs, mats, k, k_rev, by_country=True):
    """Candidate pairs as a DataFrame of row positions: i (Source 1), j (Source 2/3).

    `mats` are the matrices searched (e.g. name and full-text TF-IDF), all
    row-aligned with `recs`. With `by_country`, search stays inside each
    `country_key` value.
    """
    src = recs["src"].to_numpy()
    country = recs["country_key"].to_numpy() if by_country else np.zeros(len(recs), dtype=np.int8)
    parts = []
    for c in pd.unique(country):
        q = np.flatnonzero((src == 1) & (country == c))
        if len(q) == 0:
            continue
        for s in (2, 3):
            p = np.flatnonzero((src == s) & (country == c))
            if len(p) == 0:
                continue
            for X in mats:
                idx = nearest(X[q], X[p], k)
                parts.append(np.column_stack([np.repeat(q, idx.shape[1]), p[idx].ravel()]))
                if k_rev:
                    idx = nearest(X[p], X[q], k_rev)
                    parts.append(np.column_stack([q[idx].ravel(), np.repeat(p, idx.shape[1])]))
    if not parts:
        return pd.DataFrame({"i": [], "j": []}, dtype=np.int64)
    pairs = np.unique(np.vstack(parts), axis=0)
    return pd.DataFrame(pairs, columns=["i", "j"])


def pair_cosine(X, i, j, chunk=200_000):
    """Row-wise cosine between X[i] and X[j] (rows are already L2-normalised)."""
    out = np.empty(len(i), dtype=np.float32)
    for s in range(0, len(i), chunk):
        a, b = X[i[s:s + chunk]], X[j[s:s + chunk]]
        out[s:s + chunk] = np.asarray(a.multiply(b).sum(axis=1)).ravel()
    return out
