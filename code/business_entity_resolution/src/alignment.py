"""Token-alignment, pool-ambiguity, character and number-alignment pair features.

Decoys are near-copies of a real record with one common word swapped
('Emerald Financial' vs 'Emerald Industries') or a nearby house number, while
true duplicates differ by typos in rare words ('Ftronatber') or shuffled
letters. Whole-string fuzzy ratios cannot tell these apart; comparing the
*leftover* tokens after exact matches, weighted by how common each is, can.

Ambiguity counts say how many records in the pool share a name or address:
name-only evidence is weak when six Source 1 businesses carry that name.

Everything here is label-free and computed from the pool of the split being
processed (training pool at train time, test pool at test time).
"""
import numpy as np
import polars as pl
import scipy.sparse as sp
from rapidfuzz import process
from rapidfuzz.distance import Levenshtein
from sklearn.feature_extraction.text import HashingVectorizer

AMB_KEYS = ("core", "skel", "addr")
BAG_CHARS = "abcdefghijklmnopqrstuvwxyz0123456789"
IDF_CHUNK = 1_000_000
_HASH = HashingVectorizer(analyzer="char_wb", ngram_range=(3, 3), n_features=1 << 20,
                          alternate_sign=False, norm=None)


def pool_tables(recs: pl.DataFrame) -> dict:
    """Per-country ambiguity counts, token log document frequencies and a
    character-trigram IDF, from the prepared records of one split."""
    lf = recs.lazy().select("country_key", "src", *AMB_KEYS)
    out = {"amb": {}}
    for key in AMB_KEYS:
        out["amb"][key] = (lf.filter(pl.col(key) != "").group_by("country_key", key)
                           .agg((pl.col("src") == 1).sum().cast(pl.Float32).alias("s1"),
                                (pl.col("src") != 1).sum().cast(pl.Float32).alias("s23")).collect())
    n = lf.group_by("country_key").agg(pl.len().alias("n")).collect()
    out["floor"] = n.select("country_key", (1.0 / pl.col("n")).log().cast(pl.Float32).alias("floor"))
    for col, words_only in (("core", False), ("addr", True)):
        t = pl.col(col).str.split(" ").list.unique()
        if words_only:
            t = t.list.eval(pl.element().filter(~pl.element().str.contains(r"\d")))
        out[f"df_{col}"] = (lf.select("country_key", t.alias("t")).explode("t").filter(pl.col("t") != "")
                            .group_by("country_key", "t").len("df").filter(pl.col("df") >= 2)
                            .join(n.lazy(), on="country_key")
                            .select("country_key", "t", (pl.col("df") / pl.col("n")).log().cast(pl.Float32).alias("ldf"))
                            .collect())
    # Trigram IDF with document frequencies accumulated chunk by chunk (flat memory);
    # same formula as sklearn's TfidfTransformer(smooth_idf=True).
    core = recs["core"]
    df = np.zeros(_HASH.n_features, dtype=np.int64)
    for s in range(0, len(core), IDF_CHUNK):
        x = _HASH.transform(core.slice(s, IDF_CHUNK).to_list())
        df += np.bincount(x.indices, minlength=_HASH.n_features)
    out["idf"] = (np.log((1 + len(core)) / (1 + df)) + 1).astype(np.float32)
    return out


def _amb(R: pl.DataFrame, tables: dict, side: str) -> dict:
    f = {}
    for key, t in tables["amb"].items():
        got = R.select("country_key", key).join(t, on=["country_key", key], how="left", maintain_order="left")
        for pre in ("s1", "s23"):
            v = got[pre].fill_null(0).to_numpy().astype(np.float32)
            if key == "addr":
                v[(R["addr"] == "").to_numpy()] = np.nan
            f[f"amb_{pre}_{key}_{side}"] = v
    return f


def _align(ta: pl.Series, tb: pl.Series, country: pl.Series, df: pl.DataFrame, floor: pl.DataFrame,
           prefix: str) -> dict:
    """Leftover tokens after exact matching: how many, their best edit similarity to the
    other side's leftovers (typo vs substituted word) and how common they are."""
    n = len(ta)
    fr = pl.DataFrame({"idx": np.arange(n, dtype=np.uint32), "ta": ta.list.unique(), "tb": tb.list.unique(),
                       "country_key": country}).join(floor, on="country_key", how="left", maintain_order="left")
    ok = ((fr["ta"].list.len() > 0) & (fr["tb"].list.len() > 0)).to_numpy()
    fr = fr.with_columns(la=pl.col("ta").list.set_difference("tb"), lb=pl.col("tb").list.set_difference("ta"))
    nan = np.float32(np.nan)
    res = {"n_left_1": np.where(ok, fr["la"].list.len().to_numpy(), nan).astype(np.float32),
           "n_left_2": np.where(ok, fr["lb"].list.len().to_numpy(), nan).astype(np.float32)}
    okf = fr.filter(pl.Series(ok))

    def side(col):
        e = okf.select("idx", "country_key", "floor", pl.col(col).alias("t")).explode("t").drop_nulls("t")
        return e.join(df, on=["country_key", "t"], how="left").with_columns(pl.col("ldf").fill_null(pl.col("floor")))

    eb, ea = side("lb"), side("la")
    agg_b = eb.group_by("idx").agg(pl.col("ldf").max().alias("ldf_max_2"), pl.col("ldf").min().alias("ldf_min_2"))
    agg_a = ea.group_by("idx").agg(pl.col("ldf").max().alias("ldf_max_1"))
    cross = eb.join(ea.select("idx", pl.col("t").alias("u")), on="idx")
    cross = cross.with_columns(pl.Series("sim", process.cpdist(
        cross["t"].to_list(), cross["u"].to_list(), scorer=Levenshtein.normalized_similarity,
        workers=-1, dtype=np.float32)))
    best = (cross.group_by("idx", "t").agg(pl.col("sim").max(), pl.col("ldf").first(), pl.col("floor").first())
            .group_by("idx").agg(pl.col("sim").min().alias("best_sim_min"), pl.col("sim").max().alias("best_sim_max"),
                                 ((pl.col("sim") < 0.5) & (pl.col("ldf") > pl.col("floor") + np.log(20))).any()
                                 .cast(pl.Float32).alias("sub")))
    allf = pl.DataFrame({"idx": np.arange(n, dtype=np.uint32)})
    for agg in (agg_b, agg_a, best):
        allf = allf.join(agg, on="idx", how="left", maintain_order="left")
    for c in ("best_sim_min", "best_sim_max", "ldf_max_2", "ldf_min_2", "ldf_max_1", "sub"):
        res[c] = allf[c].cast(pl.Float32).fill_null(np.nan).to_numpy()
    return {f"{prefix}_{k}": v for k, v in res.items()}


def _bag(s: pl.Series) -> np.ndarray:
    """L2-normalised letter/digit counts: a similarity that survives shuffled letters."""
    m = np.column_stack([s.str.count_matches(ch, literal=True).fill_null(0).to_numpy()
                         for ch in BAG_CHARS]).astype(np.float32)
    return m / np.maximum(np.linalg.norm(m, axis=1, keepdims=True), 1e-9)


def _trigram_cosine(a: pl.Series, b: pl.Series, idf: np.ndarray) -> np.ndarray:
    """TF-IDF cosine of character trigrams (IDF from the split's pool)."""
    def vec(s):
        x = _HASH.transform(s.to_list()).astype(np.float32).tocsr()
        x = x @ sp.diags(idf)
        norms = np.sqrt(np.asarray(x.multiply(x).sum(axis=1)).ravel())
        norms[norms == 0] = 1.0
        return sp.diags(1.0 / norms) @ x
    return np.asarray(vec(a).multiply(vec(b)).sum(axis=1)).ravel().astype(np.float32)


def _numbers(A: pl.DataFrame, B: pl.DataFrame) -> dict:
    """Alignment of all address numbers (not only the first): coverage, closest
    numeric gap, closest edit distance and prefix relations."""
    def get(R):
        return R["addr"].str.extract_all(r"\d+").list.eval(
            pl.element().str.strip_chars_start("0").str.slice(0, 9)).list.eval(pl.element().filter(pl.element() != ""))
    n = A.height
    fr = pl.DataFrame({"idx": np.arange(n, dtype=np.uint32), "a": get(A), "b": get(B)})
    ok = pl.Series((fr["a"].list.len() > 0) & (fr["b"].list.len() > 0))
    x = (fr.filter(ok).select("idx", "a").explode("a").with_row_index("k")
         .join(fr.filter(ok).select("idx", "b").explode("b"), on="idx"))
    x = x.with_columns(
        pl.Series("lev", process.cpdist(x["a"].to_list(), x["b"].to_list(), scorer=Levenshtein.distance,
                                        workers=-1, dtype=np.int32)),
        (pl.col("a").cast(pl.Int64) - pl.col("b").cast(pl.Int64)).abs().alias("d"),
        ((pl.col("a") != pl.col("b")) & (pl.col("b").str.starts_with(pl.col("a"))
                                         | pl.col("a").str.starts_with(pl.col("b")))).alias("pre"))
    per_a = x.group_by("idx", "k").agg((pl.col("a") == pl.col("b")).any().alias("hit"))
    agg = (x.group_by("idx").agg(pl.col("d").min().log1p().alias("num_min_logdiff"),
                                 pl.col("lev").min().alias("num_min_lev"),
                                 pl.col("pre").any().cast(pl.Float32).alias("num_prefix"))
           .join(per_a.group_by("idx").agg(pl.col("hit").mean().alias("num_frac_1_in_2")), on="idx"))
    allf = pl.DataFrame({"idx": np.arange(n, dtype=np.uint32)}).join(agg, on="idx", how="left", maintain_order="left")
    return {c: allf[c].cast(pl.Float32).fill_null(np.nan).to_numpy()
            for c in ("num_frac_1_in_2", "num_min_logdiff", "num_min_lev", "num_prefix")}


def pair_features(a: pl.DataFrame, b: pl.DataFrame, tables: dict) -> dict:
    """a, b: record rows (country_key, name, core, skel, addr) of the pairs' two sides."""
    f = _amb(a, tables, "1") | _amb(b, tables, "2")
    f |= _align(a["core"].str.split(" "), b["core"].str.split(" "), a["country_key"],
                tables["df_core"], tables["floor"], "tok")

    def words(R):
        return R["addr"].str.split(" ").list.eval(
            pl.element().filter((pl.element() != "") & ~pl.element().str.contains(r"\d")))
    f |= _align(words(a), words(b), a["country_key"], tables["df_addr"], tables["floor"], "atok")
    f["chr_bag_core"] = np.einsum("ij,ij->i", _bag(a["core"]), _bag(b["core"]))
    f["chr_bag_name"] = np.einsum("ij,ij->i", _bag(a["name"]), _bag(b["name"]))
    f["chr_tri_core"] = _trigram_cosine(a["core"], b["core"], tables["idf"])
    f |= _numbers(a, b)
    return f
