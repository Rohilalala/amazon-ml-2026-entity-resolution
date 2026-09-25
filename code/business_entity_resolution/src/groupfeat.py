"""Group-structure features for candidate pairs (i, c) given a probability p for every pair.

For S1 record i with candidates G_i, the *anchors* are the candidates with p >= ANCHOR_P
(the likely members of i's duplicate group). For each pair (i, c) we compare c with every
anchor a != c and aggregate, weighting by p_a. Only probabilities of i's own candidates
are used, so with out-of-fold p there is no label leakage.
"""
import numpy as np
import polars as pl
from rapidfuzz import fuzz, process

ANCHOR_P = 0.02
MAX_ANCHORS = 10
WORKERS = -1
SOURCE_CTX = ("g_rank_src", "g_gap_src", "g_n_half_src", "g_sum_src_others", "g_pmax_othersrc", "g_psum_othersrc")


def _cp(a, b, scorer):
    return process.cpdist(a, b, scorer=scorer, workers=WORKERS, dtype=np.float32) / 100.0


def _toks(col):
    return pl.col(col).str.split(" ").list.eval(pl.element().filter(pl.element() != "")).list.unique()


def group_features(pairs: pl.DataFrame, recs: pl.DataFrame, pcol: str = "p") -> pl.DataFrame:
    """`pairs` has i, j, `pcol`; `recs` has row, src, name, core, addr, house.
    Returns pairs (same order) with g_* columns added."""
    need = pl.concat([pairs["i"], pairs["j"]]).unique().implode()
    r = recs.filter(pl.col("row").is_in(need)).select("row", "src", "core", "addr", "house").with_columns(
        _toks("core").alias("ct"), _toks("addr").alias("at"))
    base = pairs.select("i", "j", pl.col(pcol).alias("p")).with_row_index("pid")
    base = base.join(r.select(pl.col("row").alias("j"), pl.col("src").alias("sc")), on="j", how="left")
    # within-i context on p (i-side only; the j side is handled by stage 2 already)
    base = base.with_columns(
        pl.col("p").rank("min", descending=True).over("i", "sc").cast(pl.Float32).alias("g_rank_src"),
        (pl.col("p").max().over("i", "sc") - pl.col("p")).cast(pl.Float32).alias("g_gap_src"),
        (pl.col("p") >= 0.5).sum().over("i", "sc").cast(pl.Float32).alias("g_n_half_src"),
        (pl.col("p").sum().over("i", "sc") - pl.col("p")).cast(pl.Float32).alias("g_sum_src_others"),
    )
    other = base.group_by("i", "sc").agg(pl.col("p").max().alias("pmax"), pl.col("p").sum().alias("psum"))
    base = base.join(other.select("i", (5 - pl.col("sc")).alias("sc"), pl.col("pmax").alias("g_pmax_othersrc"),
                                  pl.col("psum").alias("g_psum_othersrc")), on=["i", "sc"], how="left") \
        .with_columns(pl.col("g_pmax_othersrc").fill_null(0.0), pl.col("g_psum_othersrc").fill_null(0.0))

    rj = r.select(pl.col("row").alias("j"), pl.col("core").alias("_core"), (pl.col("addr") == "").alias("_empty"))
    base = base.join(rj, on="j", how="left").with_columns(
        (pl.len().over("i", "_core") - 1).cast(pl.Float32).alias("g_n_same_core"),
        (pl.col("_empty").sum().over("i", "_core") - pl.col("_empty").cast(pl.UInt32)).cast(pl.Float32).alias("g_n_same_core_empty"),
        (pl.col("p").sum().over("i", "_core") - pl.col("p")).cast(pl.Float32).alias("g_p_same_core"),
        pl.col("_empty").sum().over("i").cast(pl.Float32).alias("g_n_empty_addr"),
    ).drop("_core", "_empty")

    anchors = (base.filter(pl.col("p") >= ANCHOR_P).sort(["i", "p"], descending=[False, True])
               .with_columns(pl.int_range(pl.len()).over("i").alias("ar")).filter(pl.col("ar") < MAX_ANCHORS)
               .select("i", pl.col("j").alias("a"), pl.col("p").alias("pa"), pl.col("sc").alias("sa")))
    x = base.select("pid", "i", "j", "sc").join(anchors, on="i").filter(pl.col("j") != pl.col("a"))
    rc = r.select(pl.col("row").alias("j"), pl.col("core").alias("core_c"), pl.col("addr").alias("addr_c"),
                  pl.col("house").alias("h_c"), pl.col("ct").alias("ct_c"), pl.col("at").alias("at_c"))
    ra = r.select(pl.col("row").alias("a"), pl.col("core").alias("core_a"), pl.col("addr").alias("addr_a"),
                  pl.col("house").alias("h_a"), pl.col("ct").alias("ct_a"), pl.col("at").alias("at_a"))
    ri = r.select(pl.col("row").alias("i"), pl.col("ct").alias("ct_i"), pl.col("at").alias("at_i"),
                  pl.col("house").alias("h_i"))
    x = x.join(rc, on="j", how="left").join(ra, on="a", how="left").join(ri, on="i", how="left")

    s_name = _cp(x["core_c"].to_list(), x["core_a"].to_list(), fuzz.token_set_ratio)
    s_addr = _cp(x["addr_c"].to_list(), x["addr_a"].to_list(), fuzz.token_sort_ratio)
    both_empty = ((x["addr_c"] == "") | (x["addr_a"] == "")).to_numpy()
    s_addr = np.where(both_empty, np.nan, s_addr).astype(np.float32)
    h_known = ((x["h_c"] != "") & (x["h_a"] != "")).to_numpy()
    h_eq = np.where(h_known, (x["h_c"] == x["h_a"]).to_numpy(), np.nan).astype(np.float32)
    # Deviations of c from S1 (tokens of c absent from S1) that the anchor shares: shared corruption.
    isnum = pl.element().str.contains(r"\d")
    nov = x.select(pl.concat_list([pl.col("ct_c").list.set_difference(pl.col("ct_i")),
                                   pl.col("at_c").list.set_difference(pl.col("at_i"))]).alias("nov"),
                   pl.col("ct_c").list.set_difference(pl.col("ct_i")).alias("nov_n"),
                   pl.col("at_c").list.set_difference(pl.col("at_i")).list.eval(pl.element().filter(isnum)).alias("nov_d"),
                   pl.concat_list(["ct_a", "at_a"]).alias("tok_a"), pl.col("ct_a"), pl.col("at_a"))

    def share(col, other):
        n = nov[col].list.len().to_numpy().astype(np.float32)
        sh = nov[col].list.set_intersection(nov[other]).list.len().to_numpy().astype(np.float32)
        with np.errstate(invalid="ignore", divide="ignore"):
            return np.where(n > 0, sh / n, np.nan).astype(np.float32)
    nov_share, nov_name_share, nov_num_share = share("nov", "tok_a"), share("nov_n", "ct_a"), share("nov_d", "at_a")
    # union coverage of c's deviations by the strong anchors (p >= 0.5), computed once per pair
    strong = pl.Series((x["pa"] >= 0.5).to_numpy())
    cov = (pl.DataFrame({"pid": x["pid"], "nov": nov["nov"], "nov_d": nov["nov_d"], "tok_a": nov["tok_a"],
                         "at_a": nov["at_a"]}).filter(strong)
           .group_by("pid").agg(pl.col("nov").first(), pl.col("nov_d").first(),
                                pl.col("tok_a").flatten().unique().alias("U"), pl.col("at_a").flatten().unique().alias("Ua"))
           .select("pid",
                   pl.col("nov").list.len().cast(pl.Float32).alias("g2_n_nov"),
                   pl.col("nov").list.set_difference(pl.col("U")).list.len().cast(pl.Float32).alias("g2_nov_uncovered"),
                   pl.col("nov_d").list.len().cast(pl.Float32).alias("g2_n_nov_num"),
                   pl.col("nov_d").list.set_difference(pl.col("Ua")).list.len().cast(pl.Float32).alias("g2_nov_num_uncovered")))
    del nov
    eq_addr = ((x["addr_c"] == x["addr_a"]) & (x["addr_c"] != "")).to_numpy().astype(np.float32)
    eq_core = (x["core_c"] == x["core_a"]).to_numpy().astype(np.float32)
    # the anchor's own similarity to S1: is c closer to the anchor than the anchor is to S1?
    anchor_pairs = x.select("i", "a").unique().join(r.select(pl.col("row").alias("a"), pl.col("core").alias("ca"), pl.col("addr").alias("aa")), on="a") \
        .join(r.select(pl.col("row").alias("i"), pl.col("core").alias("ci"), pl.col("addr").alias("ad")), on="i")
    sn = _cp(anchor_pairs["ca"].to_list(), anchor_pairs["ci"].to_list(), fuzz.token_set_ratio)
    sa = _cp(anchor_pairs["aa"].to_list(), anchor_pairs["ad"].to_list(), fuzz.token_sort_ratio)
    emp = ((anchor_pairs["aa"] == "") | (anchor_pairs["ad"] == "")).to_numpy()
    anchor_pairs = anchor_pairs.select("i", "a", pl.Series("s_ai", np.where(emp, sn, (sn + sa) / 2).astype(np.float32)))
    s_ai = x.select("i", "a").join(anchor_pairs, on=["i", "a"], how="left", maintain_order="left")["s_ai"].to_numpy()
    # house of c equals house of anchor, where c's house differs from S1's
    h_dev = ((x["h_c"] != x["h_i"]) & (x["h_c"] != "")).to_numpy()
    h_dev_sup = np.where(h_dev & h_known, (x["h_c"] == x["h_a"]).to_numpy(), np.nan).astype(np.float32)

    x = x.select("pid", "j", "a", "sc", "sa", "pa").with_columns(
        pl.Series("s_name", s_name), pl.Series("s_addr", s_addr), pl.Series("h_eq", h_eq),
        pl.Series("nov_share", nov_share), pl.Series("h_dev_sup", h_dev_sup),
        pl.Series("nov_name_share", nov_name_share), pl.Series("nov_num_share", nov_num_share),
        pl.Series("s_both", np.where(np.isnan(s_addr), s_name, (s_name + s_addr) / 2).astype(np.float32)),
        pl.Series("eq_addr", eq_addr), pl.Series("eq_core", eq_core), pl.Series("s_ai", s_ai))
    w = pl.col("pa")
    same = pl.col("sc") == pl.col("sa")

    def wmean(c):
        ok = pl.col(c).is_not_nan()
        return ((pl.when(ok).then(pl.col(c) * w).otherwise(0).sum())
                / pl.when(ok).then(w).otherwise(0).sum()).alias(f"g_wmean_{c}")

    agg = x.group_by("pid").agg(
        (w * pl.col("s_name")).max().alias("g_wmax_name"),
        (w * pl.col("s_addr").fill_nan(None)).max().alias("g_wmax_addr"),
        (w * pl.col("s_both")).max().alias("g_wmax_both"),
        wmean("s_name"), wmean("s_addr"), wmean("s_both"), wmean("h_eq"), wmean("nov_share"), wmean("h_dev_sup"),
        wmean("nov_name_share"), wmean("nov_num_share"),
        (w * pl.col("nov_num_share").fill_nan(None)).max().alias("g_wmax_nov_num"),
        (w * pl.col("nov_name_share").fill_nan(None)).max().alias("g_wmax_nov_name"),
        pl.col("s_both").sort_by(["pa", "a"], descending=[True, False]).first().alias("g_top_sim_both"),
        pl.col("pa").max().alias("g_top_p"),
        (w * pl.col("s_both")).filter(same).max().alias("g_wmax_both_same"),
        (w * pl.col("s_both")).filter(~same).max().alias("g_wmax_both_other"),
        (w * (pl.col("s_both") >= 0.9)).sum().alias("g_support90"),
        (w * (pl.col("h_eq") == 1.0)).sum().alias("g_house_agree"),
        (w * (pl.col("h_eq") == 0.0)).sum().alias("g_house_disagree"),
        pl.len().alias("g_n_anchor"),
        (w * pl.col("eq_addr")).max().alias("g2_wmax_eq_addr"),
        (w * pl.col("eq_core")).max().alias("g2_wmax_eq_core"),
        (w * (pl.col("s_both") - pl.col("s_ai"))).max().alias("g2_wmax_closer"),
        (w * (pl.col("s_both") - pl.col("s_ai"))).min().alias("g2_wmin_closer"),
    )
    # Mutual nearest neighbour across sources: c's most similar other-source anchor a*,
    # and whether c is the most similar same-source candidate of a*.
    xo = x.filter(pl.col("sc") != pl.col("sa")).join(base.select("pid", "i"), on="pid")
    xo = xo.with_columns(pl.col("s_both").rank("min", descending=True).over("i", "a", "sc").alias("rank_for_a"))
    best = (xo.sort(["pid", "s_both", "pa", "a"], descending=[False, True, True, False]).group_by("pid", maintain_order=True)
            .agg(pl.col("s_both").first().alias("g_mnn_sim"), pl.col("pa").first().alias("g_mnn_p"),
                 (pl.col("rank_for_a").first() == 1).cast(pl.Float32).alias("g_mnn")))
    out = (base.join(agg, on="pid", how="left").join(best, on="pid", how="left").join(cov, on="pid", how="left").sort("pid")
           .with_columns(pl.col("g_n_anchor").fill_null(0)))
    gcols = [c for c in out.columns if c.startswith("g_") or c.startswith("g2_")]
    return pairs.with_columns([out[c].cast(pl.Float32) for c in gcols])


def stage3_features(pairs: pl.DataFrame, recs: pl.DataFrame, pcol: str = "p", chunk_s1: int = 100_000) -> pl.DataFrame:
    """i-side probability context + group features, computed in chunks of Source 1 records
    (all features are per Source 1 record, so chunking is exact). Row order of `pairs` kept."""
    pairs = pairs.with_row_index("_ord")
    s1 = pairs["i"].unique().sort()
    out = []
    for s in range(0, len(s1), chunk_s1):
        ch = pairs.filter(pl.col("i").is_in(s1.slice(s, chunk_s1).implode()))
        ctx = ch.select("i", pl.col(pcol).alias("p")).select(
            pl.col("p").rank("min", descending=True).over("i").cast(pl.Float32).alias("rank_s1_oof"),
            (pl.col("p").max().over("i") - pl.col("p")).cast(pl.Float32).alias("gap_s1_oof"),
            pl.col("p").sum().over("i").cast(pl.Float32).alias("sum_s1_oof"),
            (pl.col("p") > 0.5).sum().over("i").cast(pl.Float32).alias("n_s1_oof_half"))
        g = group_features(ch.select("i", "j", pcol), recs, pcol)
        out.append(pl.concat([ch.select("_ord"), ctx, g.drop("i", "j", pcol)], how="horizontal"))
    return pl.concat(out).sort("_ord").drop("_ord")
