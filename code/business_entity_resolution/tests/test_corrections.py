"""Training-pool vocabulary tables and the unseen-country post-decision rules, on tiny examples."""
import json

import numpy as np
import polars as pl

from src import alignment, unseen_rules
from src.config import Config


def _pool(country, cores, addrs):
    return pl.DataFrame({"country_key": [country] * len(cores), "src": [1] * len(cores), "core": cores,
                         "skel": cores, "addr": addrs})


def _sorted(t):
    return t.sort(t.columns)


def test_vocab_tables_are_the_pool_tables_and_identity_on_train():
    train = _pool("us", ["acme group", "acme", "blue group", "red"], ["1 main st", "2 main st", "3 oak rd", "4 oak rd"])
    pool, vocab = alignment.pool_tables(train), alignment.vocab_tables(train)
    assert set(vocab) == set(alignment.VOCAB)
    for k in alignment.VOCAB:
        assert _sorted(pool[k]).equals(_sorted(vocab[k]))
        assert _sorted(alignment.with_vocab(pool, vocab)[k]).equals(_sorted(pool[k]))
    # address words exclude tokens with digits; words seen once are left to the floor
    assert set(vocab["df_addr"]["t"]) == {"main", "st", "oak", "rd"}
    assert set(vocab["df_core"]["t"]) == {"acme", "group"}


def test_with_vocab_swaps_only_countries_with_a_training_pool():
    train = _pool("us", ["acme group", "acme", "blue group"], ["1 main st", "2 main st", "3 oak rd"])
    test = pl.concat([_pool("us", ["acme group", "group x", "group y", "z"], ["a", "b", "c", "d"]),
                      _pool("france", ["alpha sarl", "alpha", "beta"], ["rue a", "rue b", "rue c"])])
    tables = alignment.with_vocab(alignment.pool_tables(test), alignment.vocab_tables(train))
    own, trained = alignment.pool_tables(test), alignment.vocab_tables(train)
    for k in alignment.VOCAB:
        got = tables[k]
        assert _sorted(got.filter(pl.col("country_key") == "us")).equals(_sorted(trained[k]))
        assert _sorted(got.filter(pl.col("country_key") == "france")).equals(
            _sorted(own[k].filter(pl.col("country_key") == "france")))
    # 'group' is in 3 of 4 test records but reads at its training frequency (2 of 3)
    ldf = tables["df_core"].filter((pl.col("country_key") == "us") & (pl.col("t") == "group"))["ldf"].item()
    assert np.isclose(ldf, np.log(2 / 3))


def _recs(rows):
    """rows: (src, name, core, addr) -> record frame with the house number as in features._prepare_chunk."""
    d = pl.DataFrame(rows, schema=["src", "name", "core", "addr"], orient="row")
    return d.with_columns(pl.col("addr").str.extract(r"(\S*\d\S*)", 1).fill_null("").alias("house"))


def test_legal_form_conflict_drop_needs_an_anchor_at_the_own_house():
    recs = _recs([(1, "alpha conseil sarl", "alpha conseil", "12 rue lilas"),
                  (2, "alpha conseil sarl", "alpha conseil", "12 rue lilas"),      # same house: anchor
                  (2, "alpha conseil sas", "alpha conseil", "13 rue lilas"),       # nearby house, conflicting form
                  (3, "alpha conseil", "alpha conseil", "13 rue lilas"),           # nearby house, no form
                  (1, "omega sarl", "omega", "40 rue b"),
                  (2, "omega sas", "omega", "41 rue b")])                          # conflict, but no anchor
    i, j = np.array([0, 0, 0, 4]), np.array([1, 2, 3, 5])
    keep, counts = unseen_rules.correct(recs, i, j, np.array([0.9, 0.8, 0.7, 0.9]), np.ones(4, dtype=bool))
    assert keep.tolist() == [True, False, True, True]
    assert counts["legal_form_drops"] == 1


def test_true_variant_word_add():
    recs = _recs([(1, "beta", "beta", "5 rue x"),
                  (2, "beta", "beta", "5 rue x"),
                  (2, "beta associes", "beta associes", "5 rue x"),   # true-variant word, same house: added
                  (2, "beta groupe", "beta groupe", "5 rue x"),       # word not in the list
                  (3, "beta cie", "beta cie", "9 rue x"),             # other house
                  (3, "beta fils", "beta fils", "5 rue x")])          # below the probability floor
    i, j = np.zeros(5, dtype=int), np.array([1, 2, 3, 4, 5])
    keep, counts = unseen_rules.correct(recs, i, j, np.array([0.9, 0.3, 0.3, 0.3, 0.1]),
                                        np.array([True, False, False, False, False]))
    assert keep.tolist() == [True, True, False, False, False]
    assert counts["true_variant_adds"] == 1


def test_generator_tail_add_and_owner_rule():
    recs = _recs([(1, "gamma delta", "gamma delta", "7 rue des lilas"),
                  (2, "gamma delta", "gamma delta", "7 rue des lilas"),
                  (2, "gamma france", "gamma france", "7 rue lilas"),              # swap tail, same house and street
                  (3, "gamma delta groupe", "gamma delta groupe", "7 lilas"),     # append tail
                  (3, "gamma delta groupe", "gamma delta groupe", "9 rue des lilas"),  # other house
                  (1, "gamma delta sud", "gamma delta sud", "7 rue des lilas"),
                  (2, "gamma delta france", "gamma delta france", "7 rue des lilas")])  # more likely owner is S1 5
    i, j = np.array([0, 0, 0, 0, 0, 5]), np.array([1, 2, 3, 4, 6, 6])
    prob = np.array([0.9, 0.05, 0.05, 0.05, 0.2, 0.6])
    keep, counts = unseen_rules.correct(recs, i, j, prob, np.array([True, False, False, False, False, True]))
    assert keep.tolist() == [True, True, True, False, False, True]
    assert counts["tail_adds"] == 2


def test_new_settings_round_trip_through_json():
    cfg = Config(feature_caps={"amb_s23_addr_2": 1.0}, stage3_use="none", stage3_by_country={"india": "full"},
                 unseen_empty_target=0.05597295153414383, unseen_rules=True)
    assert Config.from_dict(json.loads(json.dumps(cfg.to_dict()))) == cfg
