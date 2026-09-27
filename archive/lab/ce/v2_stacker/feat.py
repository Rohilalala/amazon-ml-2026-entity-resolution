"""Stage-4 v2 features. One code path for STAGE4, HOLDOUT and TEST active pairs (prob >= 0.002).

Groups (all computed from the active pairs of the dataset plus the split's cache):
  A  s01_stage4 set A (exact expressions, as build/b01_stage4_test.features_a).
  C  S1 context: CE and v7-logit gap between the S1's best and second-best candidate, number of candidates with
     CE >= 8, sum of v7 prob over the S1's active candidates.
  J  j context of s01 set B (n_j, j_ce_other), vectorised; its values depend on how many S1 of the pool the dataset
     holds (STAGE4 500k of 2.2M train S1, HOLDOUT 100k, TEST all), so it is reported for information only.
  P  pool-wide retrieval columns from the split's cache pairs (same pool construction for train and test).
  N  within-S1 neighbours: other active candidates with the same address / same core name as j (count, max CE).
  R  house and core-name relation of (i, j) (v8_lab/audit_a/common.relation) as integer codes.
  S  v7 stage-1 string features of (i, j): src.features.pair_features without the pool alignment tables.
  T  v7 alignment features that need no pool table: letter-bag cosines, number alignment (src.alignment).
Pool-free groups (A, C, N, R, S, T) read the same whatever the size of the pool; P and J do not (P's n_s1_cand and
rank_c_key halve on the US TEST pool, whose Source 1 density is half the train pool's), so the kept model avoids them."""
import sys
from pathlib import Path
import numpy as np, polars as pl

ROOT = Path("/Users/adityarohilla/Documents/amazon-ml-2026")
sys.path.insert(0, str(ROOT / "v8_lab/ce/stack"))
import s01_stage4 as s4  # noqa: E402  (puts the main repo on sys.path)
sys.path.insert(0, str(ROOT / "v8_lab/audit_a"))
from common import relation  # noqa: E402
from src.features import pair_features, RECORD_COLUMNS  # noqa: E402  (v7 stage-1 string features)
from src.alignment import _bag, _numbers  # noqa: E402  (v7 alignment features that need no pool table)

import os
V2CE = ROOT / "v8_lab/ce/v2_encoder/out" if os.environ.get("CE_V2") else None  # v2 encoder logits (HOLDOUT/STAGE4/TEST_ce)
CACHE = {"train": ROOT / "cache/58f0a60c5c264fdd", "test": ROOT / "cache/3ee42f05d06efbcb"}
PAIR_COLS = ["score", "n_s1_cand", "rank_c_key", "gap_c_key"]
HREL = ["no_addr", "no_house", "eq", "substr", "edit1", "off20", "far"]
NREL = ["same", "subset", "typo1", "swap1", "disjoint", "multi"]
GROUPS = {"A": s4.FEAT_A,
          "C": ["ce_top2", "lp_top2", "n_ce_hi", "sum_p"],
          "J": ["n_j", "j_ce_other"],
          "P": PAIR_COLS,
          "N": ["sa_n", "sa_ce", "sc_n", "sc_ce"],
          "R": ["hrel", "nrel"],
         "S": ["fz_name_ratio", "fz_name_partial", "fz_name_sort", "fz_name_set", "jw_name", "fz_core_ratio",
               "fz_core_partial", "fz_core_set", "jw_core", "fz_skel_ratio", "fz_skel_set", "fz_concat", "fz_addr_ratio",
               "fz_addr_set", "fz_addr_sort", "fz_addr_partial", "jac_core", "jac_addr", "jac_num", "shared_num",
               "house_match", "house_num_eq", "house_lev", "house_logdiff", "house_inside", "n_nums_1", "n_nums_2",
               "acr_match", "len_core_1", "len_core_2", "len_addr_1", "len_addr_2", "is_s3"],
         "T": ["chr_bag_core", "chr_bag_name", "num_frac_1_in_2", "num_min_logdiff", "num_min_lev", "num_prefix"]}


def best_other_fast(df: pl.DataFrame, col: str, keys: list) -> pl.Series:
    """s01_stage4.best_other(col, keys) with one sort + group_by instead of a top_k window (which stalls on dense
    j groups). Rows with a null key get null."""
    s = df.select(*keys, col).sort([*keys, col], descending=[False] * len(keys) + [True])
    g = s.group_by(keys).agg(_m1=pl.col(col).first(), _m2=pl.col(col).slice(1, 1).first())
    v = df.select(*keys, col).join(g, on=keys, how="left", maintain_order="left", nulls_equal=False)
    return v.select(pl.when(pl.col(col) < pl.col("_m1")).then(pl.col("_m1")).otherwise(pl.col("_m2")))[:, 0]


def _sliced(path: Path, cols: list, keep: pl.DataFrame, on: list, step: int = 1_000_000) -> pl.DataFrame:
    """Rows of a cache parquet whose `on` columns are in `keep`, read in slices to bound peak memory."""
    n = pl.scan_parquet(path).select(pl.len()).collect().item()
    return pl.concat([pl.scan_parquet(path).slice(s, step).select(cols).join(keep.lazy(), on=on, how="semi").collect()
                      for s in range(0, n, step)])


def pair_inputs(p: pl.DataFrame, split: str) -> pl.DataFrame:
    """p(i, j, ...) unique active pairs -> + PAIR_COLS, hrel, nrel (codes), ah / ch (hash of j's address / core name,
    null when empty). Rows keep p's order."""
    c = CACHE[split]
    key = p.select("i", "j")
    rows = pl.concat([key["i"], key["j"]]).unique().to_frame("row")
    cp = _sliced(c / "pairs.parquet", ["i", "j", *PAIR_COLS], key, ["i", "j"])
    r = _sliced(c / "recs.parquet", ["row", "core", "addr", "house"], rows, ["row"])
    rel = relation(key, r).select("i", "j", hrel=pl.col("hrel").replace_strict(HREL, range(len(HREL)), return_dtype=pl.Int8),
                                  nrel=pl.col("nrel").replace_strict(NREL, range(len(NREL)), return_dtype=pl.Int8))
    h = r.select(pl.col("row").alias("j"),
                 ah=pl.when(pl.col("addr") != "").then(pl.col("addr").hash()),
                 ch=pl.when(pl.col("core") != "").then(pl.col("core").hash()))
    out = (p.join(cp, on=["i", "j"], how="left", maintain_order="left")
           .join(rel, on=["i", "j"], how="left", maintain_order="left")
           .join(h, on="j", how="left", maintain_order="left"))
    assert out.height == p.height and out.select(*PAIR_COLS, "hrel", "nrel").null_count().sum_horizontal().item() == 0
    return out


def _table_free(r: pl.DataFrame, i: np.ndarray, j: np.ndarray) -> pl.DataFrame:
    """GROUPS["T"]: src.alignment's letter-bag cosines and number alignment (as alignment.pair_features)."""
    a, b = r[i], r[j]
    f = {"chr_bag_core": np.einsum("ij,ij->i", _bag(a["core"]), _bag(b["core"])),
         "chr_bag_name": np.einsum("ij,ij->i", _bag(a["name"]), _bag(b["name"])), **_numbers(a, b)}
    return pl.DataFrame({k: f[k] for k in GROUPS["T"]})


def string_inputs(p: pl.DataFrame, split: str, groups: str = "S", chunk: int = 500_000) -> pl.DataFrame:
    """GROUPS["S"]: src.features.pair_features (v7 stage 1, without the pool alignment tables) and/or GROUPS["T"]
    (_table_free) of p's (i, j) pairs, from the split's prepared cache records; rows keep p's order."""
    rows = pl.concat([p["i"], p["j"]]).unique().sort().to_frame("row")
    r = _sliced(CACHE[split] / "recs.parquet", ["row", *RECORD_COLUMNS], rows, ["row"]).sort("row")
    assert r.height == rows.height
    rr = r["row"].to_numpy()
    li, lj = np.searchsorted(rr, p["i"].to_numpy()), np.searchsorted(rr, p["j"].to_numpy())
    out = []
    for s in range(0, p.height, chunk):
        ci, cj = li[s:s + chunk], lj[s:s + chunk]
        parts = []
        if "S" in groups:
            parts.append(pair_features(r, pl.DataFrame({"i": ci, "j": cj})).select(GROUPS["S"]))
        if "T" in groups:
            parts.append(_table_free(r, ci, cj))
        out.append(pl.concat(parts, how="horizontal"))
    return p.hstack(pl.concat(out))


def features(df: pl.DataFrame, emp: np.ndarray) -> pl.DataFrame:
    """df: i, j, prob, ce_logit + pair_inputs columns -> + every column of GROUPS."""
    q = pl.col("prob").cast(pl.Float64).clip(1e-6, 1 - 1e-6)
    df = df.with_columns(lp=(q / (1 - q)).log(), ce=pl.col("ce_logit").cast(pl.Float64),
                         b_empty=pl.Series(emp[df["j"].to_numpy()]).cast(pl.Int8))
    df = df.with_columns(  # set A, s01 expressions
        ce_rank=pl.col("ce").rank("min", descending=True).over("i"), ce_gap=pl.col("ce") - s4.best_other("ce", "i"),
        lp_rank=pl.col("lp").rank("min", descending=True).over("i"), lp_gap=pl.col("lp") - s4.best_other("lp", "i"),
        n_i=pl.len().over("i"))
    df = df.with_columns(  # C: top-2 gaps are the best row's gap, spread over the S1
        ce_top2=pl.col("ce_gap").max().over("i"), lp_top2=pl.col("lp_gap").max().over("i"),
        n_ce_hi=(pl.col("ce") >= 8).sum().over("i"), sum_p=pl.col("prob").cast(pl.Float64).sum().over("i"))
    df = df.with_columns(  # J
        n_j=pl.len().over("j"), j_ce_other=best_other_fast(df, "ce", ["j"]))
    return df.with_columns(  # N
        sa_n=pl.when(pl.col("ah").is_not_null()).then(pl.len().over("i", "ah") - 1).otherwise(0),
        sa_ce=best_other_fast(df, "ce", ["i", "ah"]),
        sc_n=pl.when(pl.col("ch").is_not_null()).then(pl.len().over("i", "ch") - 1).otherwise(0),
        sc_ce=best_other_fast(df, "ce", ["i", "ch"]))


def names(groups: str) -> list:
    return [f for g in groups for f in GROUPS[g]]
