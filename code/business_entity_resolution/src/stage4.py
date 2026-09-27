"""Stage 4: a LightGBM stacker over the decided probability and the pair encoder's logit.

Only "active" pairs (decided probability >= Config.stage4_active) are rescored; every other
pair keeps its probability. Two feature sets (Config.stage4_features):

  A       the pair's own two scores, the candidate's empty-address flag and, for both scores,
          rank and gap to the best other active candidate of the same Source 1 record (8).
  ACNRST  A plus (57 in all):
          C  Source 1 context: best-vs-second gaps of both scores, candidates with encoder
             logit >= 8, summed probability of the record's active candidates;
          N  neighbours: other active candidates of the record at the same address / with the
             same core name as this one (count, best encoder logit);
          R  house-number and core-name relation of the pair, as integer codes;
          S  the stage-1 string similarities (features.pair_features without pool tables);
          T  letter-bag cosines and address-number alignment (alignment._bag, _numbers).

Every feature is a function of the Source 1 record's own active candidates and the two
records' strings, so it reads the same whatever the size of the pool, and the features can
be computed over any split of the Source 1 records.
"""
import numpy as np
import polars as pl
from rapidfuzz import process
from rapidfuzz.distance import Levenshtein

from . import alignment, unseen_rules
from . import features as string_features
from .data import DataError

FEATURES = ["lp", "ce", "b_empty", "ce_rank", "ce_gap", "lp_rank", "lp_gap", "n_i"]
GROUPS = {
    "C": ["ce_top2", "lp_top2", "n_ce_hi", "sum_p"],
    "N": ["sa_n", "sa_ce", "sc_n", "sc_ce"],
    "R": ["hrel", "nrel"],
    "S": ["fz_name_ratio", "fz_name_partial", "fz_name_sort", "fz_name_set", "jw_name", "fz_core_ratio",
          "fz_core_partial", "fz_core_set", "jw_core", "fz_skel_ratio", "fz_skel_set", "fz_concat", "fz_addr_ratio",
          "fz_addr_set", "fz_addr_sort", "fz_addr_partial", "jac_core", "jac_addr", "jac_num", "shared_num",
          "house_match", "house_num_eq", "house_lev", "house_logdiff", "house_inside", "n_nums_1", "n_nums_2",
          "acr_match", "len_core_1", "len_core_2", "len_addr_1", "len_addr_2", "is_s3"],
    "T": ["chr_bag_core", "chr_bag_name", "num_frac_1_in_2", "num_min_logdiff", "num_min_lev", "num_prefix"],
}
FEATURE_SETS = {"A": FEATURES, "ACNRST": FEATURES + [f for g in "CNRST" for f in GROUPS[g]]}
# Prepared record columns the ACNRST set reads (features.RECORD_COLUMNS covers name, core, addr, house).
RECORD_COLUMNS = string_features.RECORD_COLUMNS
HREL = ["no_addr", "no_house", "eq", "substr", "edit1", "off20", "far"]
NREL = ["same", "subset", "typo1", "swap1", "disjoint", "multi"]
BUCKETS = 20  # ACNRST features are computed per Source 1 bucket (row % BUCKETS) to bound memory
STRING_CHUNK = 500_000
FOLDS = 5  # stacker cross-validation folds (grouped by Source 1 record)
SCOPES = ("none", "seen", "all")


def best_other(col: str, by: str) -> pl.Expr:
    """Max of `col` over the other rows of the same `by` group (null when the group has one row)."""
    c, top = pl.col(col), pl.col(col).max().over(by)
    n_top = (c == top).sum().over(by)
    second = pl.when(pl.len().over(by) > 1).then(c.top_k(2).min().over(by))
    return pl.when(c < top).then(top).when(n_top > 1).then(top).otherwise(second)


def features(df: pl.DataFrame, empty: np.ndarray) -> pl.DataFrame:
    """df: active pairs (i, j, prob, ce_logit); `empty` flags records with an empty address.
    lp = logit of the probability; rank and gap to the best other active candidate of the same
    Source 1 record for both scores; n_i = number of active candidates of the record."""
    q = pl.col("prob").cast(pl.Float64).clip(1e-6, 1 - 1e-6)
    df = df.with_columns(lp=(q / (1 - q)).log(), ce=pl.col("ce_logit").cast(pl.Float64),
                         b_empty=pl.Series(empty[df["j"].to_numpy()]).cast(pl.Int8))
    return df.with_columns(
        ce_rank=pl.col("ce").rank("min", descending=True).over("i"), ce_gap=pl.col("ce") - best_other("ce", "i"),
        lp_rank=pl.col("lp").rank("min", descending=True).over("i"), lp_gap=pl.col("lp") - best_other("lp", "i"),
        n_i=pl.len().over("i"))


def best_other_keyed(df: pl.DataFrame, col: str, keys: list) -> pl.Series:
    """best_other(col, keys) by one sort and group_by (a top_k window stalls on large groups);
    rows with a null key get null."""
    s = df.select(*keys, col).sort([*keys, col], descending=[False] * len(keys) + [True])
    g = s.group_by(keys).agg(_m1=pl.col(col).first(), _m2=pl.col(col).slice(1, 1).first())
    v = df.select(*keys, col).join(g, on=keys, how="left", maintain_order="left", nulls_equal=False)
    return v.select(pl.when(pl.col(col) < pl.col("_m1")).then(pl.col("_m1")).otherwise(pl.col("_m2")))[:, 0]


def relation(recs: pl.DataFrame, i: np.ndarray, j: np.ndarray) -> pl.DataFrame:
    """House relation (hrel, as unseen_rules.house_relation) and core-name relation (nrel) of
    pairs (i, j): same core words, one side a subset, one word each differing by a typo
    (similarity >= 0.5) or by a swap, no shared word, or several differences."""
    ci, cj = recs["core"].gather(i), recs["core"].gather(j)
    hrel = unseen_rules.house_relation(recs["house"].gather(i), recs["house"].gather(j), recs["addr"].gather(j))
    p = pl.DataFrame({"ci": ci, "cj": cj})
    ta, tb = pl.col("ci").str.split(" ").list.unique(), pl.col("cj").str.split(" ").list.unique()
    p = p.select(ta.list.set_difference(tb).alias("la"), tb.list.set_difference(ta).alias("lb"),
                 ta.list.set_intersection(tb).list.len().alias("ninter"))
    one = ((p["la"].list.len() == 1) & (p["lb"].list.len() == 1)).to_numpy()
    sim = process.cpdist(p["la"].list.first().fill_null("").to_list(), p["lb"].list.first().fill_null("").to_list(),
                         scorer=Levenshtein.normalized_similarity, workers=-1, dtype=np.float32)
    p = p.with_columns(pl.Series("sim1", np.where(one, sim, np.nan)))
    nrel = (pl.when((pl.col("la").list.len() == 0) & (pl.col("lb").list.len() == 0)).then(pl.lit("same"))
            .when((pl.col("la").list.len() == 0) | (pl.col("lb").list.len() == 0)).then(pl.lit("subset"))
            .when(pl.col("sim1") >= 0.5).then(pl.lit("typo1"))
            .when(pl.col("sim1") < 0.5).then(pl.lit("swap1"))
            .when(pl.col("ninter") == 0).then(pl.lit("disjoint"))
            .otherwise(pl.lit("multi")))
    return pl.DataFrame({"hrel": pl.Series(hrel, dtype=pl.String).replace_strict(HREL, range(len(HREL)), return_dtype=pl.Int8),
                         "nrel": p.select(nrel)[:, 0].replace_strict(NREL, range(len(NREL)), return_dtype=pl.Int8)})


def string_group(recs: pl.DataFrame, i: np.ndarray, j: np.ndarray) -> pl.DataFrame:
    """Groups S and T of pairs (i, j), in chunks."""
    out = []
    for s in range(0, len(i), STRING_CHUNK):
        ci, cj = i[s:s + STRING_CHUNK], j[s:s + STRING_CHUNK]
        S = string_features.pair_features(recs, pl.DataFrame({"i": ci, "j": cj})).select(GROUPS["S"])
        a, b = recs[ci], recs[cj]
        t = {"chr_bag_core": np.einsum("ij,ij->i", alignment._bag(a["core"]), alignment._bag(b["core"])),
             "chr_bag_name": np.einsum("ij,ij->i", alignment._bag(a["name"]), alignment._bag(b["name"])),
             **alignment._numbers(a, b)}
        out.append(S.hstack(pl.DataFrame({k: t[k] for k in GROUPS["T"]})))
    return pl.concat(out)


def features_acnrst(df: pl.DataFrame, empty: np.ndarray, recs: pl.DataFrame) -> pl.DataFrame:
    """df: active pairs (i, j, prob, ce_logit) of whole Source 1 records; `recs`: prepared
    records (RECORD_COLUMNS, row-aligned). Adds every ACNRST column."""
    df = features(df, empty)
    df = df.with_columns(
        ce_top2=pl.col("ce_gap").max().over("i"), lp_top2=pl.col("lp_gap").max().over("i"),
        n_ce_hi=(pl.col("ce") >= 8).sum().over("i"), sum_p=pl.col("prob").cast(pl.Float64).sum().over("i"))
    i, j = df["i"].to_numpy(), df["j"].to_numpy()
    df = df.with_columns(recs["addr"].gather(j).alias("_a"), recs["core"].gather(j).alias("_c"))
    df = df.with_columns(ah=pl.when(pl.col("_a") != "").then(pl.col("_a").hash()),
                         ch=pl.when(pl.col("_c") != "").then(pl.col("_c").hash())).drop("_a", "_c")
    df = df.with_columns(
        sa_n=pl.when(pl.col("ah").is_not_null()).then(pl.len().over("i", "ah") - 1).otherwise(0),
        sa_ce=best_other_keyed(df, "ce", ["i", "ah"]),
        sc_n=pl.when(pl.col("ch").is_not_null()).then(pl.len().over("i", "ch") - 1).otherwise(0),
        sc_ce=best_other_keyed(df, "ce", ["i", "ch"]))
    return df.hstack(relation(recs, i, j)).hstack(string_group(recs, i, j))


def matrix(df: pl.DataFrame, names: list = FEATURES) -> np.ndarray:
    return df.select(names).cast(pl.Float32).to_numpy()


def feature_set(names: list) -> str:
    """Name of the feature set whose column list is `names`."""
    for k, v in FEATURE_SETS.items():
        if v == names:
            return k
    raise DataError(f"stage-4 features are none of the known sets {sorted(FEATURE_SETS)}")


def countries(scope: str, seen: set, present) -> list:
    """Countries decided on stage 4: none, those with training labels ("seen"), or all."""
    if scope not in SCOPES:
        raise DataError(f"stage4_scope must be one of {SCOPES}, got {scope!r}")
    return [] if scope == "none" else [c for c in present if scope == "all" or c in seen]


def probability(model, i: np.ndarray, j: np.ndarray, prob: np.ndarray, ce: np.ndarray, empty: np.ndarray,
                active: np.ndarray, recs: pl.DataFrame | None = None) -> np.ndarray:
    """`prob` as float64 with the `active` pairs rescored by the stage-4 `model` (a pipeline.Stage,
    whose boosters' probabilities are averaged). The ACNRST set also needs `recs`."""
    out = prob.astype(np.float64)
    df = pl.DataFrame({"i": i[active], "j": j[active], "prob": prob[active], "ce_logit": ce[active]})
    names = getattr(model, "features", FEATURES)
    if feature_set(names) == "A":
        out[active] = model.predict(matrix(features(df, empty)))
        return out
    idx, p = np.flatnonzero(active), np.empty(int(active.sum()))
    bucket = df["i"].to_numpy() % BUCKETS
    for b in range(BUCKETS):
        m = np.flatnonzero(bucket == b)
        if not len(m):
            continue
        p[m] = model.predict(matrix(features_acnrst(df[m], empty, recs), names))
    out[idx] = p
    return out


def encoder_pairs(pairs: pl.DataFrame, truth: pl.DataFrame, pool: pl.DataFrame, params: dict,
                  exclude_j: pl.Series) -> tuple[pl.DataFrame, pl.DataFrame]:
    """Pairs the encoder is fine-tuned on (fit) and checked on (val), columns i, j, y.

    `pool`: Source 1 records (row) outside the matcher's sample and holdout. params["train_s1"]
    of them are drawn at random, params["val_s1"] of those are kept for validation; each record
    brings its top params["top_k"] candidates by retrieval score plus every true pair among its
    candidates. Fit pairs whose candidate is in `exclude_j` are dropped, so the encoder never
    saw the candidates of the pairs stage 4 is trained on."""
    s1 = pool.sample(params["train_s1"], seed=7).sort("row")
    val_rows = s1.sample(params["val_s1"], seed=8)["row"]
    p = (pairs.select("i", "j", "score").filter(pl.col("i").is_in(s1["row"].implode()))
         .join(truth.with_columns(pl.lit(1, dtype=pl.Int8).alias("y")), on=["i", "j"], how="left")
         .with_columns(pl.col("y").fill_null(0),
                       pl.col("score").rank("ordinal", descending=True).over("i").alias("rank"))
         .filter((pl.col("rank") <= params["top_k"]) | (pl.col("y") == 1)).sort("i", "rank"))
    is_val = pl.col("i").is_in(val_rows.implode())
    fit = p.filter(~is_val & ~pl.col("j").is_in(exclude_j.implode())).select("i", "j", "y")
    return fit, p.filter(is_val).select("i", "j", "y")
