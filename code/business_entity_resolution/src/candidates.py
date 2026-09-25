"""Candidate generation (blocking) by weighted multi-key retrieval.

Names in this data are built from a limited vocabulary, so single words are
shared by tens of thousands of businesses. Each record is therefore described
by *keys* that are rare even when their parts are common:

    n:  core-name token skeleton           nn: pair of core-name tokens (order-free)
    d:  number token from the address      dn: number x name token
    a:  address word skeleton              da: number x address word
    c:  whole core name without spaces     na: name token x address word
        (matches 'qualityaceculture.com')
    n3, nn3, c3: n, nn, c under skeleton3, which also merges the sounds that
        Indic-script transliteration confuses ('teknalji' / 'technology')
    cp, cs: first / last 6 letters of the concatenated name skeleton
        ('dentalcare' / 'dentlcaregroup' share a prefix)
    ap: order-free pair of address words

Keys are built on blocking-only normalised strings (`tnorm`: look-alike digits
inside name words, leading zeros and glued number ranges in addresses). The
address words used are the rarest ones of the record within its blocking frame,
so a long address keeps its street and locality words instead of the first
words it happens to start with.

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
import os
import time
from concurrent.futures import ThreadPoolExecutor

import numpy as np
import polars as pl
import scipy.sparse as sp

from .text import skeleton

log = logging.getLogger(__name__)

MAX_NAME_TOKENS = 5
MAX_NUMBERS = 3
MAX_ADDR_WORDS = 5
KEY_CHUNK = 500_000
SEARCH_CHUNK = 2000
# scipy releases the GIL inside sparse products, so chunks run in parallel threads.
SEARCH_THREADS = min(8, os.cpu_count() or 1)
# Hash seeds keep the key families apart (a name token never equals a number key).
SEED = {"n": 11, "d": 12, "a": 13, "c": 14, "nn": 21, "dn": 22, "da": 23, "na": 24,
        "cp": 32, "cs": 33, "ap": 35, "n3": 41, "nn3": 42, "c3": 43}
# Digits used as letters inside a name word ('pi1ates', 'jacks0n'); Source 1 names never mix them.
LEET_FROM, LEET_TO = ["0", "1", "5", "8", "6"], ["o", "l", "s", "b", "g"]


def skeleton3(e: pl.Expr) -> pl.Expr:
    """skeleton() after merging sounds that Indic-script transliteration confuses: soft c -> s,
    soft g -> j, b -> v, m -> n (anusvara), -tion -> sion, tu -> cu, final ng -> n, j -> s, d -> t."""
    e = e.str.replace_all(r"c([eiy])", "s$1").str.replace_all(r"g([eiy])", "j$1")
    e = e.str.replace_all("b", "v").str.replace_all("m", "n")
    e = e.str.replace_all(r"ti([aou])", "si$1").str.replace_all("tu", "cu").str.replace_all(r"ng$", "n")
    return skeleton(e.str.replace_all("j", "s").str.replace_all("d", "t"))


def tnorm(df: pl.DataFrame) -> pl.DataFrame:
    """Blocking-only normalisation of `core` and `addr`: name words mixing letters and digits get
    look-alike letters (pi1ates -> pilates); address numbers lose leading zeros (0355 -> 355) and
    an even-length >= 6 digit run is split in halves (clean() glues '515-517' into '515517')."""
    e = pl.element()
    mixed = e.str.contains(r"[a-z]") & e.str.contains(r"\d")
    core = pl.col("core").str.split(" ").list.eval(
        pl.when(mixed).then(e.str.replace_many(LEET_FROM, LEET_TO)).otherwise(e)).list.join(" ")
    z = pl.when(e.str.contains(r"^0+\d*[1-9a-z]")).then(e.str.strip_chars_start("0")).otherwise(e)
    n = e.str.len_chars()
    split = pl.when(e.str.contains(r"^\d+$") & (n >= 6) & (n % 2 == 0)).then(
        e.str.slice(0, n // 2) + " " + e.str.slice(n // 2)).otherwise(e)
    addr = pl.col("addr").str.split(" ").list.eval(z).list.eval(split).list.join(" ")
    return df.select("row", core.alias("core"), addr.alias("addr"))


def _tokens(df: pl.DataFrame, col: str) -> pl.DataFrame:
    return (df.select("row", pl.col(col).str.split(" ").alias("t")).explode("t")
            .filter(pl.col("t").is_not_null() & (pl.col("t") != "")))


def _first(df: pl.DataFrame, limit: int) -> pl.DataFrame:
    """First `limit` distinct hashed tokens of each row, numbered 0.. in order."""
    return (df.unique(["row", "h"], keep="first", maintain_order=True)
            .with_columns(pl.int_range(pl.len()).over("row").alias("i"))
            .filter(pl.col("i") < limit))


def _address_words(addr: pl.DataFrame, sk=skeleton) -> pl.DataFrame:
    """(row, h) hashed skeletons of the address words (no digits, skeleton >= 2 letters), in order."""
    words = addr.filter(~pl.col("t").str.contains(r"\d")).with_columns(sk(pl.col("t")).alias("k"))
    return words.filter(pl.col("k").str.len_chars() >= 2).select("row", pl.col("k").hash(SEED["a"]).alias("h"))


def _keys_chunk(df: pl.DataFrame, word_df: pl.DataFrame | None = None, uv3: bool = False) -> pl.DataFrame:
    """(row, h) blocking keys for one chunk of records (`row`, `core`, `addr`). `word_df`
    (h, wdf) orders each record's address words rarest first before the first
    MAX_ADDR_WORDS are kept; without it they keep their address order. `uv3`: only the
    eight base families, every skeleton replaced by skeleton3 (the second channel)."""
    df = tnorm(df)
    sk = skeleton3 if uv3 else skeleton
    toks = _tokens(df, "core")

    def name_keys(sk, seed):
        k = sk(pl.col("t"))
        return _first(toks.select("row", pl.when(k == "").then(pl.col("t")).otherwise(k).hash(seed).alias("h")),
                      MAX_NAME_TOKENS)

    name, name3 = name_keys(sk, SEED["n"]), name_keys(skeleton3, SEED["n3"])
    addr = _tokens(df, "addr")
    nums = _first(addr.filter(pl.col("t").str.contains(r"\d"))
                  .select("row", pl.col("t").hash(SEED["d"]).alias("h")), MAX_NUMBERS)
    words = _address_words(addr, sk)
    if word_df is not None:
        # Order-preserving join, then a stable sort: ties keep address order, so the
        # chosen words are the same on every run.
        words = (words.join(word_df, on="h", how="left", maintain_order="left")
                 .with_columns(pl.col("wdf").fill_null(0)).sort("row", "wdf", maintain_order=True).drop("wdf"))
    words = _first(words, MAX_ADDR_WORDS)
    joined = pl.col("core").str.replace_all(" ", "")
    concat = df.select("row", sk(joined).hash(SEED["c"]).alias("h"))
    concat3 = df.select("row", skeleton3(joined).hash(SEED["c3"]).alias("h"))
    cc = df.select("row", skeleton(joined).alias("s"))
    cc = cc.with_columns(pl.when(pl.col("s").str.len_chars() > 6).then(pl.col("s").str.replace(r"km$", ""))
                         .otherwise(pl.col("s")).alias("s")).filter(pl.col("s").str.len_chars() >= 7)

    def pairs(a, b, seed, ordered=True, a_limit=None):
        left = a if a_limit is None else a.filter(pl.col("i") < a_limit)
        j = left.join(b.select("row", pl.col("h").alias("h2"), pl.col("i").alias("i2")), on="row")
        if not ordered:  # token pairs: each unordered pair once, order-free key
            j = j.filter(pl.col("i") < pl.col("i2")).with_columns(
                pl.min_horizontal("h", "h2").alias("h"), pl.max_horizontal("h", "h2").alias("h2"))
        return j.select("row", pl.struct("h", "h2").hash(seed).alias("h"))

    parts = [name.select("row", "h"), nums.select("row", "h"), words.select("row", "h"), concat,
             pairs(name, name, SEED["nn"], ordered=False),
             pairs(nums, name, SEED["dn"]), pairs(nums, words, SEED["da"]),
             pairs(name, words, SEED["na"], a_limit=3)]
    if not uv3:
        parts += [name3.select("row", "h"), concat3, pairs(name3, name3, SEED["nn3"], ordered=False),
                  cc.select("row", pl.col("s").str.slice(0, 6).hash(SEED["cp"]).alias("h")),
                  cc.select("row", pl.col("s").str.slice(-6).hash(SEED["cs"]).alias("h")),
                  pairs(words, words, SEED["ap"], ordered=False)]
    return pl.concat([p.with_columns(pl.col("row").cast(pl.UInt32)) for p in parts]).unique()


def record_keys(recs: pl.DataFrame, uv3: bool = False) -> pl.DataFrame:
    """(row, h) blocking keys of every record of one blocking frame; `recs` needs `row`, `core`, `addr`.
    Main channel: address-word frequencies are counted over the whole frame first (one count per
    record) so each record keeps its rarest words. `uv3` channel: words in address order."""
    word_df = None if uv3 else pl.concat(
        [_address_words(_tokens(tnorm(recs.slice(s, KEY_CHUNK)), "addr")).unique()
         for s in range(0, recs.height, KEY_CHUNK)]).group_by("h").agg(pl.len().alias("wdf"))
    return pl.concat([_keys_chunk(recs.slice(s, KEY_CHUNK), word_df, uv3) for s in range(0, recs.height, KEY_CHUNK)])


def key_matrix(keys: pl.DataFrame, n_rows: int, max_df: int) -> sp.csr_matrix:
    """L2-normalised IDF-weighted key matrix (n_rows x vocabulary)."""
    # Sorted vocabulary and sorted entries: group_by order varies between runs, and
    # column order decides float summation order (hence tie-breaks) in the search.
    vocab = (keys.group_by("h").agg(pl.len().alias("df"))
             .filter((pl.col("df") >= 2) & (pl.col("df") <= max_df))
             .sort("h").with_row_index("col")
             .with_columns((np.log(n_rows) - pl.col("df").log()).cast(pl.Float32).alias("w")))
    kv = keys.join(vocab.select("h", "col", "w"), on="h").sort("row", "col")
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

    def chunk(s):
        r, c, v = _top_k((q[s:s + SEARCH_CHUNK] @ tt).tocsr(), k)
        return r + s, c, v

    with ThreadPoolExecutor(SEARCH_THREADS) as pool:
        res = list(pool.map(chunk, range(0, q.shape[0], SEARCH_CHUNK)))
    if not res:
        return np.zeros(0, np.int64), np.zeros(0, np.int64), np.zeros(0, np.float32)
    return tuple(np.concatenate(x) for x in zip(*res))


def generate(recs: pl.DataFrame, k: int, k_rev: int, max_df: int, uv3: bool = False) -> pl.DataFrame:
    """Candidate pairs (i, j, score): i a Source 1 row, j a Source 2/3 row, score the
    key cosine. `recs` needs `row`, `src`, `country_key`, `core`, `addr`. `uv3` selects the
    second key channel (measured: k=3, k_rev=3, max_df=1000 next to the main k=10, k_rev=3, 3000)."""
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
            keys = record_keys(local.with_columns(pl.int_range(pl.len(), dtype=pl.UInt32).alias("row")), uv3)
            t_keys = time.time()
            m = key_matrix(keys, local.height, max_df)
            del keys
            t_matrix = time.time()
            nq = q_rows.height
            q, t = m[:nq], m[nq:]
            n_keys = m.shape[1]
            del m
            schema = {"i": pl.UInt32, "j": pl.UInt32, "score": pl.Float32}
            r, c, v = search(q, t, k)
            t_fwd = time.time()
            block = [pl.DataFrame({"i": glob[r], "j": glob[nq + c], "score": v}, schema=schema)]
            if k_rev:
                r, c, v = search(t, q, k_rev)
                block.append(pl.DataFrame({"i": glob[c], "j": glob[nq + r], "score": v}, schema=schema))
            del q, t
            # Blocks never share a Source 2/3 record, so de-duplicating per block is
            # exact and keeps the peak memory independent of the number of blocks.
            parts.append(pl.concat(block).group_by("i", "j").agg(pl.col("score").max()))
            log.info("blocking %s / source %d: %d + %d records, %d keys in vocabulary; keys %.0fs, "
                     "matrix %.0fs, forward %.0fs, reverse %.0fs", country, s, nq, t_rows.height, n_keys,
                     t_keys - t0, t_matrix - t_keys, t_fwd - t_matrix, time.time() - t_fwd)
    if not parts:
        return pl.DataFrame(schema={"i": pl.UInt32, "j": pl.UInt32, "score": pl.Float32})
    return pl.concat(parts).sort("i", "j")
