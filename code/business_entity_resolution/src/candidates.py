"""Candidate generation (blocking) by weighted multi-key retrieval.

Names in this data are built from a limited vocabulary, so single words are
shared by tens of thousands of businesses. Each record is therefore described
by *keys* that are rare even when their parts are common:

    n:  core-name token skeleton           nn: pair of core-name tokens (order-free)
    d:  number token from the address      dn: number x name token
    a:  address word skeleton              da: number x address word
    c:  whole core name without spaces     na: name token x address word
        (matches 'qualityaceculture.com')

Keys are weighted by inverse document frequency; keys seen once (they cannot
link anything) or more than `max_df` times (too common to block on) are
dropped. Records become L2-normalised sparse vectors, and for every Source 1
record we keep the top-k Source 2 and top-k Source 3 records by cosine (one
sparse matrix product per chunk). A reverse pass keeps, for every Source 2/3
record, its top-k_rev Source 1 records. Search never crosses countries (the
training data has no cross-country match).

Memory: tokens are hashed to 64-bit integers as soon as they are produced and
combination keys are built from those integers, in chunks of records, so the
key table costs 12 bytes per key whatever the text length.
"""
import logging
import time

import numpy as np
import polars as pl
import scipy.sparse as sp

from .text import skeleton

log = logging.getLogger(__name__)

MAX_NAME_TOKENS = 5
MAX_NUMBERS = 3
MAX_ADDR_WORDS = 5
KEY_CHUNK = 500_000
SEARCH_CHUNK = 4000
# Hash seeds keep the key families apart (a name token never equals a number key).
SEED = {"n": 11, "d": 12, "a": 13, "c": 14, "nn": 21, "dn": 22, "da": 23, "na": 24}


def _tokens(df: pl.DataFrame, col: str) -> pl.DataFrame:
    return (df.select("row", pl.col(col).str.split(" ").alias("t")).explode("t")
            .filter(pl.col("t").is_not_null() & (pl.col("t") != "")))


def _first(df: pl.DataFrame, limit: int) -> pl.DataFrame:
    """First `limit` distinct hashed tokens of each row, numbered 0.. in order."""
    return (df.unique(["row", "h"], keep="first", maintain_order=True)
            .with_columns(pl.int_range(pl.len()).over("row").alias("i"))
            .filter(pl.col("i") < limit))


def _keys_chunk(df: pl.DataFrame) -> pl.DataFrame:
    """(row, h) blocking keys for one chunk of records (`row`, `core`, `addr`)."""
    name = _tokens(df, "core").with_columns(skeleton(pl.col("t")).alias("k"))
    name = _first(name.select("row", pl.when(pl.col("k") == "").then(pl.col("t")).otherwise(pl.col("k"))
                              .hash(SEED["n"]).alias("h")), MAX_NAME_TOKENS)
    addr = _tokens(df, "addr")
    nums = _first(addr.filter(pl.col("t").str.contains(r"\d"))
                  .select("row", pl.col("t").hash(SEED["d"]).alias("h")), MAX_NUMBERS)
    words = addr.filter(~pl.col("t").str.contains(r"\d")).with_columns(skeleton(pl.col("t")).alias("k"))
    words = _first(words.filter(pl.col("k").str.len_chars() >= 2)
                   .select("row", pl.col("k").hash(SEED["a"]).alias("h")), MAX_ADDR_WORDS)
    concat = df.select("row", skeleton(pl.col("core").str.replace_all(" ", "")).hash(SEED["c"]).alias("h"))

    def pairs(a, b, seed, ordered=True, a_limit=None):
        left = a if a_limit is None else a.filter(pl.col("i") < a_limit)
        j = left.join(b.select("row", pl.col("h").alias("h2"), pl.col("i").alias("i2")), on="row")
        if not ordered:  # name-token pairs: each unordered pair once, order-free key
            j = j.filter(pl.col("i") < pl.col("i2")).with_columns(
                pl.min_horizontal("h", "h2").alias("h"), pl.max_horizontal("h", "h2").alias("h2"))
        return j.select("row", pl.struct("h", "h2").hash(seed).alias("h"))

    parts = [name.select("row", "h"), nums.select("row", "h"), words.select("row", "h"), concat,
             pairs(name, name, SEED["nn"], ordered=False),
             pairs(nums, name, SEED["dn"]), pairs(nums, words, SEED["da"]),
             pairs(name, words, SEED["na"], a_limit=3)]
    return pl.concat([p.with_columns(pl.col("row").cast(pl.UInt32)) for p in parts]).unique()


def record_keys(recs: pl.DataFrame) -> pl.DataFrame:
    """(row, h) blocking keys of every record; `recs` needs `row`, `core`, `addr`."""
    return pl.concat([_keys_chunk(recs.slice(s, KEY_CHUNK)) for s in range(0, recs.height, KEY_CHUNK)])


def key_matrix(keys: pl.DataFrame, n_rows: int, max_df: int) -> sp.csr_matrix:
    """L2-normalised IDF-weighted key matrix (n_rows x vocabulary)."""
    vocab = (keys.group_by("h").agg(pl.len().alias("df"))
             .filter((pl.col("df") >= 2) & (pl.col("df") <= max_df))
             .with_row_index("col")
             .with_columns((np.log(n_rows) - pl.col("df").log()).cast(pl.Float32).alias("w")))
    kv = keys.join(vocab.select("h", "col", "w"), on="h")
    m = sp.csr_matrix((kv["w"].to_numpy(), (kv["row"].to_numpy(), kv["col"].to_numpy())),
                      shape=(n_rows, vocab.height), dtype=np.float32)
    del kv
    norms = np.sqrt(np.asarray(m.multiply(m).sum(axis=1)).ravel())
    norms[norms == 0] = 1.0
    return sp.diags((1.0 / norms).astype(np.float32)).dot(m).tocsr()


def _top_k(s: sp.csr_matrix, k: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Row, column and score of the k largest entries of every row of `s`."""
    rows = np.repeat(np.arange(s.shape[0]), np.diff(s.indptr))
    order = np.lexsort((-s.data, rows))
    rank = np.arange(len(order)) - s.indptr[rows[order]]
    keep = order[rank < k]
    return rows[keep], s.indices[keep], s.data[keep]


def search(q: sp.csr_matrix, t: sp.csr_matrix, k: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Top-k rows of `t` for every row of `q` by cosine (rows are unit vectors)."""
    tt = t.T.tocsr()
    rows, cols, vals = [], [], []
    for s in range(0, q.shape[0], SEARCH_CHUNK):
        r, c, v = _top_k((q[s:s + SEARCH_CHUNK] @ tt).tocsr(), k)
        rows.append(r + s)
        cols.append(c)
        vals.append(v)
    if not rows:
        return np.zeros(0, np.int64), np.zeros(0, np.int64), np.zeros(0, np.float32)
    return np.concatenate(rows), np.concatenate(cols), np.concatenate(vals)


def generate(recs: pl.DataFrame, k: int, k_rev: int, max_df: int) -> pl.DataFrame:
    """Candidate pairs (i, j, score): i a Source 1 row, j a Source 2/3 row, score the
    key cosine. `recs` needs `row`, `src`, `country_key`, `core`, `addr`."""
    parts = []
    base = recs.select("row", "src", "country_key", "core", "addr")
    for country in base["country_key"].unique(maintain_order=True).to_list():
        in_country = base.filter(pl.col("country_key") == country)
        q_rows = in_country.filter(pl.col("src") == 1)
        if q_rows.height == 0:
            continue
        for s in (2, 3):
            t_rows = in_country.filter(pl.col("src") == s)
            if t_rows.height == 0:
                continue
            t0 = time.time()
            # Local frame: Source 1 rows first, then the target source.
            local = pl.concat([q_rows, t_rows])
            glob = local["row"].to_numpy()
            m = key_matrix(record_keys(local.with_columns(pl.int_range(pl.len(), dtype=pl.UInt32).alias("row"))),
                           local.height, max_df)
            nq = q_rows.height
            q, t = m[:nq], m[nq:]
            del m
            r, c, v = search(q, t, k)
            parts.append(pl.DataFrame({"i": glob[r], "j": glob[nq + c], "score": v},
                                      schema={"i": pl.UInt32, "j": pl.UInt32, "score": pl.Float32}))
            if k_rev:
                r, c, v = search(t, q, k_rev)
                parts.append(pl.DataFrame({"i": glob[c], "j": glob[nq + r], "score": v},
                                          schema={"i": pl.UInt32, "j": pl.UInt32, "score": pl.Float32}))
            log.info("blocking %s / source %d: %d + %d records, %d keys in vocabulary, %.1fs",
                     country, s, nq, t_rows.height, q.shape[1], time.time() - t0)
    if not parts:
        return pl.DataFrame(schema={"i": pl.UInt32, "j": pl.UInt32, "score": pl.Float32})
    return pl.concat(parts).group_by("i", "j").agg(pl.col("score").max()).sort("i", "j")
