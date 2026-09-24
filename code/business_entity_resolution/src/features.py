"""Record preparation and pair features for the matching model.

No feature encodes the country itself, so the model applies unchanged to a
country it has not seen in training (France in the test set).
"""
import numpy as np
import polars as pl
from rapidfuzz import fuzz, process
from rapidfuzz.distance import JaroWinkler

from . import text

RECORD_COLUMNS = ["name", "core", "addr", "skel", "concat", "acr", "house", "src",
                   "name_non_latin", "addr_non_latin"]
PREPARE_CHUNK = 1_000_000


def _prepare_chunk(recs: pl.DataFrame) -> pl.DataFrame:
    name = text.clean_name(recs["business_name"])
    addr = text.clean_address(recs["business_address"])
    core = text.core_name(name)
    return recs.select("row", "entity_id", "src").with_columns(
        country_key=recs["country"].str.strip_chars().str.to_lowercase(),
        name=name, core=core, addr=addr,
        skel=text.skeleton_string(core), concat=text.concat_skeleton(core), acr=text.acronym(core),
        house=addr.str.extract(r"(\S*\d\S*)", 1).fill_null(""),
        name_non_latin=text.is_non_latin(recs["business_name"]),
        addr_non_latin=text.is_non_latin(recs["business_address"]),
    )


def prepare(recs: pl.DataFrame) -> pl.DataFrame:
    """Normalised strings and flags for every record (row-aligned with `recs`).
    Token lists are built per feature chunk instead of stored, to save memory."""
    return pl.concat([_prepare_chunk(recs.slice(s, PREPARE_CHUNK))
                      for s in range(0, recs.height, PREPARE_CHUNK)], rechunk=True)


def _token_lists(a: pl.DataFrame) -> pl.DataFrame:
    """Distinct core-name tokens, address words and address numbers per row."""
    toks = pl.col("addr").str.split(" ").list.eval(pl.element().filter(pl.element() != ""))
    return a.select(
        pl.col("core").str.split(" ").list.unique().alias("core_tok"),
        toks.list.eval(pl.element().filter(~pl.element().str.contains(r"\d"))).list.unique().alias("addr_tok"),
        toks.list.eval(pl.element().filter(pl.element().str.contains(r"\d"))).list.unique().alias("nums"),
    )


def competition(pairs: pl.DataFrame, col: str, tag: str) -> pl.DataFrame:
    """How each pair ranks on `col` among the candidates of its Source 1 record
    and among the Source 1 records competing for its Source 2/3 record.
    Source 1 is deduplicated, so a Source 2/3 record should have one owner."""
    return pairs.with_columns(
        pl.col(col).rank("min", descending=True).over("i").cast(pl.Float32).alias(f"rank_s1_{tag}"),
        (pl.col(col).max().over("i") - pl.col(col)).cast(pl.Float32).alias(f"gap_s1_{tag}"),
        pl.col(col).rank("min", descending=True).over("j").cast(pl.Float32).alias(f"rank_c_{tag}"),
        (pl.col(col).max().over("j") - pl.col(col)).cast(pl.Float32).alias(f"gap_c_{tag}"),
    )


def add_group_features(pairs: pl.DataFrame, src_of: np.ndarray) -> pl.DataFrame:
    """Retrieval-score context features (must see the whole candidate table)."""
    pairs = pairs.with_columns(pl.Series("jsrc", src_of[pairs["j"].to_numpy()]))
    pairs = competition(pairs, "score", "key")
    return pairs.with_columns(
        pl.len().over("i").cast(pl.Float32).alias("n_cand_s1"),
        pl.len().over("j").cast(pl.Float32).alias("n_s1_cand"),
        pl.col("score").rank("min", descending=True).over("i", "jsrc").cast(pl.Float32).alias("rank_s1_src_key"),
    )


def _fuzzy(a: list, b: list, scorer, scale: float = 100.0) -> np.ndarray:
    return process.cpdist(a, b, scorer=scorer, workers=-1, dtype=np.float32) / scale


def _jaccard(a: pl.Series, b: pl.Series) -> np.ndarray:
    """Element-wise Jaccard of two list columns; NaN when both lists are empty."""
    inter = a.list.set_intersection(b).list.len().to_numpy().astype(np.float32)
    union = a.list.set_union(b).list.len().to_numpy().astype(np.float32)
    with np.errstate(divide="ignore", invalid="ignore"):
        return np.where(union > 0, inter / union, np.nan).astype(np.float32)


def pair_features(recs: pl.DataFrame, pairs: pl.DataFrame) -> pl.DataFrame:
    """String-similarity features for one chunk of pairs, plus the context
    columns already on `pairs`. `recs` is the prepared frame (row-aligned)."""
    i, j = pairs["i"].to_numpy(), pairs["j"].to_numpy()
    a = recs.select(RECORD_COLUMNS)[i]
    b = recs.select(RECORD_COLUMNS)[j]
    f = {}
    na, nb = a["name"].to_list(), b["name"].to_list()
    ca, cb = a["core"].to_list(), b["core"].to_list()
    aa, ab = a["addr"].to_list(), b["addr"].to_list()
    sa, sb = a["skel"].to_list(), b["skel"].to_list()
    f["fz_name_ratio"] = _fuzzy(na, nb, fuzz.ratio)
    f["fz_name_partial"] = _fuzzy(na, nb, fuzz.partial_ratio)
    f["fz_name_sort"] = _fuzzy(na, nb, fuzz.token_sort_ratio)
    f["fz_name_set"] = _fuzzy(na, nb, fuzz.token_set_ratio)
    f["jw_name"] = _fuzzy(na, nb, JaroWinkler.normalized_similarity, 1.0)
    f["fz_core_ratio"] = _fuzzy(ca, cb, fuzz.ratio)
    f["fz_core_partial"] = _fuzzy(ca, cb, fuzz.partial_ratio)
    f["fz_core_set"] = _fuzzy(ca, cb, fuzz.token_set_ratio)
    f["jw_core"] = _fuzzy(ca, cb, JaroWinkler.normalized_similarity, 1.0)
    f["fz_skel_ratio"] = _fuzzy(sa, sb, fuzz.ratio)
    f["fz_skel_set"] = _fuzzy(sa, sb, fuzz.token_set_ratio)
    f["fz_concat"] = _fuzzy(a["concat"].to_list(), b["concat"].to_list(), fuzz.ratio)
    f["fz_addr_ratio"] = _fuzzy(aa, ab, fuzz.ratio)
    f["fz_addr_set"] = _fuzzy(aa, ab, fuzz.token_set_ratio)
    f["fz_addr_sort"] = _fuzzy(aa, ab, fuzz.token_sort_ratio)
    f["fz_addr_partial"] = _fuzzy(aa, ab, fuzz.partial_ratio)

    ta, tb = _token_lists(a), _token_lists(b)
    f["jac_core"] = _jaccard(ta["core_tok"], tb["core_tok"])
    f["jac_addr"] = _jaccard(ta["addr_tok"], tb["addr_tok"])
    f["jac_num"] = _jaccard(ta["nums"], tb["nums"])
    f["shared_num"] = ta["nums"].list.set_intersection(tb["nums"]).list.len().to_numpy().astype(np.float32)
    ha, hb = a["house"].to_numpy(), b["house"].to_numpy()
    f["house_match"] = np.where((ha == "") | (hb == ""), np.nan, ha == hb).astype(np.float32)
    joined_a, joined_b = a["core"].str.replace_all(" ", ""), b["core"].str.replace_all(" ", "")
    f["acr_match"] = (((a["acr"] != "") & (a["acr"] == joined_b))
                      | ((b["acr"] != "") & (b["acr"] == joined_a))).to_numpy().astype(np.float32)
    f["len_core_1"] = a["core"].str.len_chars().to_numpy().astype(np.float32)
    f["len_core_2"] = b["core"].str.len_chars().to_numpy().astype(np.float32)
    f["len_addr_1"] = a["addr"].str.len_chars().to_numpy().astype(np.float32)
    f["len_addr_2"] = b["addr"].str.len_chars().to_numpy().astype(np.float32)
    f["is_s3"] = (b["src"].to_numpy() == 3).astype(np.float32)
    f["name_non_latin"] = b["name_non_latin"].to_numpy().astype(np.float32)
    f["addr_non_latin"] = b["addr_non_latin"].to_numpy().astype(np.float32)
    return pairs.drop("jsrc", strict=False).with_columns(
        [pl.Series(k, v) for k, v in f.items()])
