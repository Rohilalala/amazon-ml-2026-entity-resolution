"""The leaderboard-run pieces on small hand-made inputs: post-decision steps, rules, recall,
test-time adaptation labels and the ACNRST stacker features."""
import numpy as np
import polars as pl
import pytest

from src import adapt, final, postprocess as pp, recall, rules, stage4
from src.features import prepare

COMMON = ["club", "garage", "maison", "atelier", "cabinet", "boulangerie"]


def raw_records(s1: list, s23: list, filler: int = 25) -> pl.DataFrame:
    """Test records: s1 / s23 are (id, name, address); plus `filler` France pool records per
    common word so every word of COMMON is a common France name word."""
    rows = [(i, n, a, "France", 1) for i, n, a in s1] + [(i, n, a, "France", int(i[1])) for i, n, a in s23]
    rows += [(f"S3-9{k}{w}", f"{w} filler{k}", "", "France", 3) for w in COMMON for k in range(filler)]
    return pl.DataFrame(rows, schema=["entity_id", "business_name", "business_address", "country", "src"], orient="row")


def sub_of(d: dict) -> pl.DataFrame:
    return pp.frame(pl.Series(list(d)), pl.Series([",".join(v) for v in d.values()]))


def lists_of(sub: pl.DataFrame) -> dict:
    return {s: m.split(",") if m else [] for s, m in sub.select(pp.S1, pp.M).iter_rows()}


# ---------------------------------------------------------------- post-decision steps

def test_sister_filter_drops_common_word_swaps_only():
    recs = pp.TestRecords(raw_records(
        [("S1-1", "Club Soleil", "3 rue x"), ("S1-2", "Garage Lune", "4 rue y"), ("S1-3", "Maison Bleue", "5 rue z")],
        [("S2-1", "Atelier Soleil", ""), ("S2-2", "Club Soleil SARL", ""), ("S2-3", "Garaeg Lune", ""),
         ("S2-4", "Cabinet Bleue", "")]))
    sub = sub_of({"S1-1": ["S2-1", "S2-2"], "S1-2": ["S2-3"], "S1-3": ["S2-4"]})
    out, rep = pp.sister_filter(sub, recs)
    # club -> atelier is a swap of common words (dropped); a legal form and a typo are not;
    # S1-3 kept only a swap, so it is emptied
    assert lists_of(out) == {"S1-1": ["S2-2"], "S1-2": ["S2-3"], "S1-3": []}
    assert rep == {"drop_pairs": 2, "drop_s1": 2, "emptied_s1": 1}


def test_restore_scrambles_readds_rare_same_letter_words_if_unclaimed():
    recs = pp.TestRecords(raw_records([("S1-1", "Club Soleil", ""), ("S1-2", "Club Etoile", "")],
                                      [("S2-1", "Cbul Soleil", ""), ("S2-2", "Atelier Etoile", "")]))
    pre = sub_of({"S1-1": ["S2-1"], "S1-2": ["S2-2"]})
    post = sub_of({"S1-1": [], "S1-2": []})
    out, rep = pp.restore_scrambles(pre, post, recs)
    assert lists_of(out) == {"S1-1": ["S2-1"], "S1-2": []} and rep == {"sister_drops": 2, "restored": 1}


def test_apply_lists_semantics():
    sub = sub_of({"a": ["x", "y"], "b": ["z"], "c": []})
    drops = pl.DataFrame({"s1": ["a", "c"], "m": ["y", "q"]})
    adds = pl.DataFrame({"s1": ["a", "b", "c", "zz"], "m": ["w", "x", "v", "u"]})
    out, rep = pp.apply_lists(sub, drops, adds)
    # x is still kept by a (no add), zz is no Source 1 row; adds are appended in list order
    assert lists_of(out) == {"a": ["x", "w"], "b": ["z"], "c": ["v"]} and rep == {"dropped": 1, "added": 2}
    src = pl.DataFrame({"s1": ["a", "c"], "m": ["w", "v"], "src": [["x"], ["nope"]]})
    out, _ = pp.apply_lists(sub, drops, adds, sources=src)
    assert lists_of(out) == {"a": ["x", "w"], "b": ["z"], "c": []}  # c no longer keeps v's source
    out, _ = pp.apply_lists(sub, drops.clear(), pl.DataFrame({"s1": ["c", "c"], "m": ["n", "n"]}), unique_adds=True)
    assert lists_of(out)["c"] == ["n"]
    late, rep = pp.apply_late(sub, drops, adds)
    assert lists_of(late) == {"a": ["x", "w"], "b": ["z"], "c": ["v"]} and rep == {"dropped": 1, "added": 2}


def test_candidates_hold_every_match():
    # list adds (India copies) can match ids no candidate list has; the written candidates must hold them
    cands = sub_of({"a": ["x"], "b": [], "c": ["z"]})
    sub = sub_of({"a": ["x", "w"], "b": ["v"], "c": []})
    assert lists_of(final.with_matches(cands, sub)) == {"a": ["x", "w"], "b": ["v"], "c": ["z"]}
    assert final.with_matches(cands, sub_of({"a": ["x"], "b": [], "c": []})).equals(cands)


# ---------------------------------------------------------------- rules

def test_different_street_needs_same_number_and_no_shared_street_word():
    assert rules.different_street("12 rue des lilas, paris", "12 avenue victor hugo, paris")
    assert not rules.different_street("12 rue des lilas, paris", "12 rue des lillas, paris")  # typo
    assert not rules.different_street("12 rue des lilas, paris", "14 avenue victor hugo, paris")
    assert not rules.different_street("appt 12, paris", "12 avenue victor hugo")  # no street type


def test_name_rules():
    freq = __import__("collections").Counter({w: 50 for w in COMMON})
    assert rules.sister2_swap("soleil club maison", "soleil atelier maison", freq)
    assert not rules.sister2_swap("soleil club maison", "soleil clubs maison", freq)
    assert rules._swap_last(["cercle", "gestion"], ["cercle", "groupe"]) == "groupe"
    assert rules._swap_last(["cercle", "gestion"], ["cerlce", "gestion"]) is None
    assert rules.cie_tailmid_rule("martin et cie sarl", "martin et compagnie sarl") == "fr_cie"
    assert rules.cie_tailmid_rule("martin bois services", "martin services groupe") == "fr_tailmid"
    assert rules.cie_tailmid_rule("martin bois services", "martin bois groupe") is None  # last word: E's rule
    assert rules.region("12 rue x, Bordeaux, Nouvelle-Aquitaine") == "NA" and rules.region("3 place y, 59000 Lille") == "HF"
    assert rules.region("5 rue a, Gironde, Nord") is None and rules.region("") is None


def _prep(raw: pl.DataFrame) -> pl.DataFrame:
    return prepare(raw.with_row_index("row"))


def test_groupe_swap_and_street_drops_on_a_frame():
    raw = raw_records([("S1-1", "Cercle Gestion SARL", "3 rue des lilas, lyon"),
                       ("S1-2", "Garage Martin", "9 rue haute, lyon")],
                      [("S2-1", "Cercle SARL Groupe", "3 rue des lilas, lyon"),
                       ("S2-2", "Garage Martin", "9 avenue victor hugo, lyon")], filler=0)
    prep = _prep(raw)
    row = dict(zip(prep["entity_id"].to_list(), prep["row"].to_list()))
    cand = pl.DataFrame({"i": [row["S1-1"], row["S1-2"]], "j": [row["S2-1"], row["S2-2"]]}, schema={"i": pl.UInt32, "j": pl.UInt32})
    sub = sub_of({"S1-1": [], "S1-2": ["S2-2"]})
    assert rules.groupe_swap_adds(sub, prep, cand).rows() == [("S1-1", "S2-1")]
    assert rules.street_drops(sub, pp.TestRecords(raw)).rows() == [("S1-2", "S2-2")]
    assert rules.groupe_swap_adds(sub_of({"S1-1": [], "S1-2": ["S2-1"]}), prep, cand).height == 0  # j claimed


def test_us_state_conflict():
    raw = pl.DataFrame([("S1-1", "Acme", "5 main st, austin, TX 73301", "US", 1), ("S2-1", "Acme", "5 main st, austin, CA", "US", 2),
                        ("S2-2", "Acme", "5 main st, austin, TX", "US", 2), ("S3-1", "Acme", "5 main st", "US", 3)],
                       schema=["entity_id", "business_name", "business_address", "country", "src"], orient="row")
    sub = sub_of({"S1-1": ["S2-1", "S2-2", "S3-1"]})
    assert rules.us_state_conflict_drops(sub, _prep(raw)).rows() == [("S1-1", "S2-1")]


def test_india_copy_adds_need_an_unscored_unclaimed_copy():
    raw = pl.DataFrame([("S1-1", "Ram Traders", "5 mg road", "India", 1), ("S2-1", "Ram Traders", "5 mg road", "India", 2),
                        ("S3-1", "Ram Traders", "5 mg road", "India", 3), ("S3-2", "Ram Traders", "", "India", 3)],
                       schema=["entity_id", "business_name", "business_address", "country", "src"], orient="row")
    prep = _prep(raw)
    cand = pl.DataFrame({"i": [0], "j": [1]}, schema={"i": pl.UInt32, "j": pl.UInt32})
    none = pl.DataFrame(schema={"s1_row": pl.UInt32, "j_row": pl.UInt32})
    adds = rules.india_copy_adds(sub_of({"S1-1": ["S2-1"]}), prep, cand, none)
    assert adds.rows() == [("S1-1", "S3-1", ["S2-1"])]  # the empty-address record is no copy
    scored = pl.DataFrame({"s1_row": [0], "j_row": [2]}, schema={"s1_row": pl.UInt32, "j_row": pl.UInt32})
    assert rules.india_copy_adds(sub_of({"S1-1": ["S2-1"]}), prep, cand, scored).height == 0


# ---------------------------------------------------------------- recall

def test_add_new_thresholds_owner_and_ties():
    code, j = np.array([0, 0, 1, 1, 2]), np.array([10, 11, 10, 12, 13])
    p = np.array([0.9, 0.7, 0.9, 0.95, 0.99])
    taken = np.array([False, False, False, False, True])
    # S1 0 keeps a pair (threshold 0.8), S1 1 keeps none (0.75); j 10 tied: first row wins; j 13 taken
    assert recall.add_new(code, j, p, np.array([0.8, 0.75, 0.5]), taken).tolist() == [True, False, False, True, False]


def test_new_pairs_rule():
    u = np.uint32
    nb = {(2, "fwd"): (np.array([0], u), np.array([[5, 6, 7]], u), np.array([[0.9, 0.8, 0.7]], np.float16)),
          (2, "rev"): (np.array([6, 7, 8], u), np.array([[0], [0], [1]], u), np.array([[0.8], [0.7], [0.6]], np.float16))}
    cand = pl.DataFrame({"i": [0], "j": [5]}, schema={"i": pl.UInt32, "j": pl.UInt32})
    P = recall.sort_pairs(recall.new_pairs(nb, cand, "us"))
    # 5 is a candidate, so 6 is S1 0's forward new rank 1; 7 (new rank 2) comes in as S1 0's
    # reverse rank 1, 8 as S1 1's (reverse only)
    assert P.select("s1_row", "j_row", "new_rank", "rev_rank").rows() == [(0, 6, 1, 1), (0, 7, 2, 1), (1, 8, None, 1)]
    assert P["dense_rank"].to_list() == [2, 3, None]


def test_recall_apply_appends_sorted_adds_and_candidates():
    records = pl.DataFrame({"row": [0, 1, 2, 3, 4], "entity_id": ["S1-a", "S1-b", "S2-x", "S2-y", "S3-z"],
                            "country_key": ["us"] * 5, "src": [1, 1, 2, 2, 3]})
    sub = sub_of({"S1-a": ["S2-x"], "S1-b": []})
    cands = sub_of({"S1-a": ["S2-x"], "S1-b": []})
    P = pl.DataFrame({"s1_row": [0, 0, 1], "j_row": [4, 3, 3], "country": ["us"] * 3, "prob": [0.85, 0.9, 0.8]})
    out, c, rep = recall.apply(sub, cands, P, records, 0.8, 0.75)
    assert lists_of(out) == {"S1-a": ["S2-x", "S2-y", "S3-z"], "S1-b": []}  # S2-y goes to its best S1 only
    assert lists_of(c) == {"S1-a": ["S2-x", "S2-y", "S3-z"], "S1-b": ["S2-y"]} and rep["added"] == 2


def test_knn_is_exact():
    rng = np.random.default_rng(0)
    q, t = (rng.normal(size=(n, 384)) for n in (50, 300))
    q, t = ((x / np.linalg.norm(x, axis=1, keepdims=True)).astype(np.float16) for x in (q, t))
    idx, cos = recall.knn(q, t, 5)
    ref = np.argsort(-(q.astype(np.float32) @ t.astype(np.float32).T), axis=1)[:, :5]
    assert (idx[:, 0] == ref[:, 0]).mean() > 0.95 and (np.diff(cos.astype(np.float32), axis=1) <= 0).all()


# ---------------------------------------------------------------- adaptation

def test_pseudo_label_classes_and_sample():
    rng = np.random.default_rng(0)
    n = 4000
    T = pl.DataFrame({"i": np.arange(n) // 4, "j": np.arange(n), "country": np.repeat(["us", "india", "france", "us"], n // 4),
                      "p4": np.where(rng.random(n) < 0.5, 0.99, 0.0) + 0.01 * rng.random(n), "ce": rng.normal(size=n)})
    kept = T.filter(pl.col("p4") > 0.5).select("i", "j")
    neg = T.filter(pl.col("p4") < 0.001).select("i", "j")
    C = adapt.classes(T, kept, neg)
    row = C.filter(pl.col("cls") == "pos").row(0, named=True)
    assert row["kept"] and row["p4"] >= adapt.POS_P4
    assert C.filter(pl.col("cls") == "hard").group_by("i").len()["len"].max() <= adapt.HARD_PER_S1
    S = adapt.sample(C, 400, 0.4, np.random.default_rng(1), True)
    assert S["y"].sum() == S.filter(pl.col("cls") == "pos").height and S.filter(pl.col("cls") == "sister")["y"].sum() == 0
    fr = S.filter((pl.col("country") == "france") & (pl.col("cls") != "sister")).height
    assert fr == pytest.approx(160, abs=2) and S["is_ho"].any()
    fit = pl.DataFrame({"i": np.arange(100), "j": np.arange(100), "y": np.zeros(100, np.int8), "country": ["us"] * 100})
    a1 = adapt.build(T, kept, neg, fit, {**adapt.ROUNDS[3], "n_pseudo": 400, "n_anchor": 30})
    a2 = adapt.build(T, kept, neg, fit, {**adapt.ROUNDS[3], "n_pseudo": 400, "n_anchor": 30})
    assert a1[0].equals(a2[0]) and a1[1].equals(a2[1]) and a1[1].height == 30


# ---------------------------------------------------------------- stage 4 ACNRST

def test_acnrst_features_by_hand_and_bucket_invariance():
    raw = raw_records([("S1-1", "Cercle Gestion", "3 rue des lilas"), ("S1-2", "Garage Martin", "9 rue haute")],
                      [("S2-1", "Cercle Gestion", "3 rue des lilas"), ("S2-2", "Cercle Gestoin", "3 rue des lilas"),
                       ("S2-3", "Garage Martin", "")], filler=0)
    prep = _prep(raw)
    df = pl.DataFrame({"i": [0, 0, 1], "j": [2, 3, 4], "prob": [0.9, 0.3, 0.5], "ce_logit": [9.0, 2.0, 1.0]},
                      schema_overrides={"prob": pl.Float32, "ce_logit": pl.Float32})
    empty = np.array([False, False, False, False, True])
    F = stage4.features_acnrst(df, empty, prep)
    assert set(stage4.FEATURE_SETS["ACNRST"]) <= set(F.columns) and len(stage4.FEATURE_SETS["ACNRST"]) == 57
    assert F["n_ce_hi"].to_list() == [1, 1, 0] and F["sum_p"].to_list() == pytest.approx([1.2, 1.2, 0.5])
    assert F["sa_n"].to_list() == [1, 1, 0] and F["sa_ce"].to_list() == [2.0, 9.0, None]  # same address, other's logit
    assert [stage4.NREL[k] for k in F["nrel"]] == ["same", "typo1", "same"]
    assert [stage4.HREL[k] for k in F["hrel"]] == ["eq", "eq", "no_addr"]

    class Mean:
        features = stage4.FEATURE_SETS["ACNRST"]

        def predict(self, x):
            return np.nan_to_num(x).sum(axis=1) / 1e3
    i, j = df["i"].to_numpy(), df["j"].to_numpy()
    prob, ce, act = df["prob"].to_numpy(), df["ce_logit"].to_numpy(), np.ones(3, bool)
    one = stage4.probability(Mean(), i, j, prob, ce, empty, act, prep)
    stage4.BUCKETS, keep = 1, stage4.BUCKETS
    try:
        assert np.array_equal(stage4.probability(Mean(), i, j, prob, ce, empty, act, prep), one)
    finally:
        stage4.BUCKETS = keep
