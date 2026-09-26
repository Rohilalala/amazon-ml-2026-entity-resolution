"""Stage 4: a LightGBM stacker over the decided probability and the pair encoder's logit.

Only "active" pairs (decided probability >= Config.stage4_active) are rescored; every other
pair keeps its probability. The features use nothing but the pair's own two scores, the
candidate's empty-address flag and the other active candidates of the same Source 1 record,
so they read the same whatever the size of the pool.
"""
import numpy as np
import polars as pl

from .data import DataError

FEATURES = ["lp", "ce", "b_empty", "ce_rank", "ce_gap", "lp_rank", "lp_gap", "n_i"]
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


def matrix(df: pl.DataFrame) -> np.ndarray:
    return df.select(FEATURES).cast(pl.Float32).to_numpy()


def countries(scope: str, seen: set, present) -> list:
    """Countries decided on stage 4: none, those with training labels ("seen"), or all."""
    if scope not in SCOPES:
        raise DataError(f"stage4_scope must be one of {SCOPES}, got {scope!r}")
    return [] if scope == "none" else [c for c in present if scope == "all" or c in seen]


def probability(model, i: np.ndarray, j: np.ndarray, prob: np.ndarray, ce: np.ndarray, empty: np.ndarray,
                active: np.ndarray) -> np.ndarray:
    """`prob` as float64 with the `active` pairs rescored by the stage-4 `model` (a pipeline.Stage)."""
    df = features(pl.DataFrame({"i": i[active], "j": j[active], "prob": prob[active], "ce_logit": ce[active]}), empty)
    out = prob.astype(np.float64)
    out[active] = model.predict(matrix(df))
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
