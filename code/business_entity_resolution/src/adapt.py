"""Test-time adaptation of the pair encoder: pseudo-labelled test pairs plus labelled training anchors.

One round (r) starts from round r-1's encoder and learns from round r-1's own test decision:

  pool    test pairs active for stage 4 (decided probability >= stage4_active), with the previous
          round's stage-4 probability p4 and encoder logit ce;
  classes pos    = kept in the previous round's final decision and p4 >= 0.98;
          hard   = per Source 1 record, the 2 unkept candidates with the highest ce among p4 <= 0.05;
          easy   = other pairs with p4 <= 0.02;
          sister = pairs an earlier post-decision filter removed (`negatives`), label 0, all used;
  sample  n_pseudo pairs, `france_share` of them from France (the unseen country), US and India by
          their active-pair counts; per country 55% pos, 22.5% hard, 22.5% easy; 10% of the sampled
          Source 1 records held out (is_ho) for a check, never trained on;
  anchors n_anchor random labelled training pairs from the encoder's own fit set (see
          ROUNDS / README), so the encoder does not drift from the labelled countries.

The fine-tune is pair_encoder.train over anchors (training text) and pseudo pairs (test text),
lr 1e-5, one pass.
"""
import numpy as np
import polars as pl

from . import pair_encoder, postprocess

POS_SHARE, POS_P4, HARD_P4, EASY_P4, HARD_PER_S1 = 0.55, 0.98, 0.05, 0.02, 2
# The three rounds of the leaderboard run (init = the encoder the round starts from).
ROUNDS = {
    1: {"n_pseudo": 555_000, "n_anchor": 500_000, "france_share": 0.35, "seed": 0, "init": "v3"},
    2: {"n_pseudo": 555_000, "n_anchor": 500_000, "france_share": 0.40, "seed": 1, "init": "round 1"},
    3: {"n_pseudo": 350_000, "n_anchor": 350_000, "france_share": 0.45, "seed": 2, "init": "round 2"},
}
LR = 1e-5


def classes(T: pl.DataFrame, kept: pl.DataFrame, negatives: pl.DataFrame | None) -> pl.DataFrame:
    """T: active pairs (i, j, country, p4, ce); kept / negatives: (i, j). Adds `cls`."""
    T = T.join(kept.select("i", "j", kept=pl.lit(True)), on=["i", "j"], how="left").with_columns(pl.col("kept").fill_null(False))
    if negatives is not None:
        T = (T.join(negatives.select("i", "j", sister=pl.lit(True)), on=["i", "j"], how="left")
             .with_columns(pl.col("sister").fill_null(False)))
    else:
        T = T.with_columns(sister=pl.lit(False))
    hard = (T.filter(~pl.col("kept") & (pl.col("p4") <= HARD_P4))
            .with_columns(r=pl.col("ce").rank("ordinal", descending=True).over("i"))
            .filter(pl.col("r") <= HARD_PER_S1).select("i", "j", hard=pl.lit(True)))
    return (T.join(hard, on=["i", "j"], how="left").with_columns(pl.col("hard").fill_null(False))
            .with_columns(cls=pl.when(pl.col("sister")).then(pl.lit("sister"))
                          .when(pl.col("kept") & (pl.col("p4") >= POS_P4)).then(pl.lit("pos"))
                          .when(pl.col("hard")).then(pl.lit("hard")).when(pl.col("p4") <= EASY_P4).then(pl.lit("easy"))
                          .otherwise(None))
            .drop("hard"))


def sample(T: pl.DataFrame, n_pseudo: int, france_share: float, rng: np.random.Generator,
           with_negatives: bool) -> pl.DataFrame:
    """Pseudo-labelled sample (i, j, country, p4, ce, kept, cls, y, is_ho), sorted by (i, j)."""
    act = dict(T.group_by("country").len().iter_rows())
    quota = {"france": france_share * n_pseudo}
    for c in ("us", "india"):
        quota[c] = (1 - france_share) * n_pseudo * act[c] / (act["us"] + act["india"])
    S = []
    for c, q in quota.items():
        for k, f in (("pos", POS_SHARE), ("hard", (1 - POS_SHARE) / 2), ("easy", (1 - POS_SHARE) / 2)):
            d = T.filter((pl.col("country") == c) & (pl.col("cls") == k))
            n = int(round(q * f))
            if d.height < n:
                raise ValueError(f"pseudo-labels: {c}/{k} has {d.height} pairs, needs {n}")
            S.append(d.gather(rng.choice(d.height, n, replace=False)))
    if with_negatives:
        S.append(T.filter(pl.col("cls") == "sister"))
    S = pl.concat(S).drop("sister", strict=False).with_columns(y=(pl.col("cls") == "pos").cast(pl.Int8))
    s1 = S["i"].unique().sort()
    ho = s1.gather(rng.choice(len(s1), len(s1) // 10, replace=False))
    return S.with_columns(is_ho=pl.col("i").is_in(ho.implode())).sort("i", "j")


def to_rows(p: pl.DataFrame, ids: pl.DataFrame) -> pl.DataFrame:
    """(s1, m) id pairs -> (i, j) record rows; `ids`: row, entity_id."""
    return (p.join(ids.rename({"entity_id": "s1", "row": "i"}), on="s1")
            .join(ids.rename({"entity_id": "m", "row": "j"}), on="m").select("i", "j"))


def negatives_of(pre: list[pl.DataFrame], decision: pl.DataFrame, recs: postprocess.TestRecords,
                 drop_scrambles: bool) -> pl.DataFrame:
    """(s1, m) pairs kept by an earlier stage of a chain (`pre`: pairs frames) but not by the final
    `decision` (pairs frame), without generator typo scrambles when `drop_scrambles`."""
    neg = pl.concat(pre).unique().join(decision, on=["s1", "m"], how="anti")
    if drop_scrambles:
        neg = neg.filter(~pl.Series([postprocess._scramble(recs, s, m) for s, m in neg.iter_rows()], dtype=pl.Boolean))
    return neg.join(decision, on=["s1", "m"], how="anti")


def anchors(fit: pl.DataFrame, n: int, rng: np.random.Generator) -> pl.DataFrame:
    """n random labelled pairs (i, j, y, country) of an encoder fit set, in fit-set order."""
    return fit.gather(np.sort(rng.choice(fit.height, n, replace=False))).select("i", "j", "y", "country")


def build(T: pl.DataFrame, kept: pl.DataFrame, negatives: pl.DataFrame | None, fit: pl.DataFrame,
          round_: dict) -> tuple[pl.DataFrame, pl.DataFrame]:
    """(pseudo, anchor) tables of one round; one random stream (seed) for both, pseudo first."""
    rng = np.random.default_rng(round_["seed"])
    S = sample(classes(T, kept, negatives), round_["n_pseudo"], round_["france_share"], rng, negatives is not None)
    return S, anchors(fit, round_["n_anchor"], rng)


def train(init, out_dir, text_train: pl.Series, text_test: pl.Series, anchor: pl.DataFrame, pseudo: pl.DataFrame,
          params: dict) -> dict:
    """One pass from `init` over anchors (training text) then pseudo pairs (test text, is_ho excluded),
    shuffled together (pair_encoder.train with lr LR); the held-out pseudo pairs are the check."""
    n = len(text_train)
    text = pl.concat([text_train, text_test])
    fit = pl.concat([anchor.select("i", "j", "y"),
                     pseudo.filter(~pl.col("is_ho")).select(pl.col("i") + n, pl.col("j") + n, "y")])
    ho = pseudo.filter("is_ho").select(pl.col("i") + n, pl.col("j") + n, "y")
    ho = ho.gather(np.random.default_rng(1).permutation(ho.height)[:8192])
    return pair_encoder.train(init, out_dir, text, fit.with_columns(pl.col("y").cast(pl.Int8)),
                              ho.with_columns(pl.col("y").cast(pl.Int8)), {**params, "lr": LR})
