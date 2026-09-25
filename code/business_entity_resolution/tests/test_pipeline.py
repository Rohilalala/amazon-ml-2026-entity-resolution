"""End-to-end and data-contract tests on a synthetic dataset (no network)."""
import subprocess
import sys
from pathlib import Path

import polars as pl
import pytest

from src import cli, data, pipeline
from src.config import Config
from tests.synth import make_dataset

VALIDATOR = Path(__file__).resolve().parents[3] / "student_resource" / "utils" / "validate_submission.py"


@pytest.fixture(scope="module")
def dataset(tmp_path_factory):
    root = tmp_path_factory.mktemp("dataset")
    make_dataset(root)
    return root


@pytest.fixture(scope="module")
def trained(dataset, tmp_path_factory):
    model_dir = tmp_path_factory.mktemp("model")
    out_dir = tmp_path_factory.mktemp("output")
    code = cli.main(["run", "--data", str(dataset), "--model", str(model_dir),
                     "--out", str(out_dir), "--folds", "3", "--sample", "300"])
    assert code == 0
    return model_dir, out_dir


def _read(path):
    return pl.read_csv(path, separator="\t", quote_char=None, infer_schema=False,
                       missing_utf8_is_empty_string=True)


def _ids(cell):
    return [x for x in cell.split(",") if x]


def test_outputs_follow_submission_rules(dataset, trained):
    _, out = trained
    test_s1 = _read(dataset / "test" / "test_source1.tsv")["entity_id"].to_list()
    valid = set(_read(dataset / "test" / "test_source2.tsv")["entity_id"]) | \
        set(_read(dataset / "test" / "test_source3.tsv")["entity_id"])
    match = _read(out / "matching_results.tsv")
    cand = _read(out / "candidate_pairs.tsv")
    assert match.columns == ["source1_entity_id", "matched_entity_ids"]
    assert cand.columns == ["source1_entity_id", "candidate_entity_ids"]
    assert match["source1_entity_id"].to_list() == test_s1
    assert cand["source1_entity_id"].to_list() == test_s1
    for m, c in zip(match["matched_entity_ids"], cand["candidate_entity_ids"]):
        m, c = _ids(m), _ids(c)
        assert len(m) == len(set(m)) and len(c) == len(set(c))
        assert set(m) <= set(c) <= valid


@pytest.mark.skipif(not VALIDATOR.is_file(), reason="challenge validator not present")
def test_official_validator_passes(dataset, trained):
    _, out = trained
    res = subprocess.run([sys.executable, str(VALIDATOR), "--matching", str(out / "matching_results.tsv"),
                          "--candidate", str(out / "candidate_pairs.tsv"),
                          "--test-dir", str(dataset / "test"), "--check-ids"],
                         capture_output=True, text=True)
    assert res.returncode == 0, res.stdout + res.stderr


def test_model_quality_and_unseen_country(trained):
    model_dir, _ = trained
    report = pipeline.Bundle.load(model_dir).report
    assert report["blocking"]["pair_recall"] > 0.95
    assert report["oof_f05"] > 0.9
    assert len(report["oof_f05_by_stage"]) == 3
    assert report["oof_f05"] >= report["oof_f05_by_stage"][0] - 0.01
    assert set(report["holdout"]) == {"none", "full", "robust"}
    assert all(v["f05"] > 0.85 for v in report["holdout"].values())
    assert "oof_f05_stage3_robust" in report
    assert set(report["unseen_country_stage1"]) == {"us", "india"}
    for v in report["unseen_country_stage1"].values():
        assert v["f05_with_rule_from_other_country"] > 0.85
    assert (model_dir / "predict_report.json").is_file()


def test_every_test_country_gets_predictions(trained):
    model_dir, _ = trained
    import json
    report = json.loads((model_dir / "predict_report.json").read_text())
    assert set(report["by_country"]) == {"us", "india", "france"}
    assert all(v["with_matches"] > 0 for v in report["by_country"].values())


def test_predict_is_deterministic(dataset, trained, tmp_path):
    model_dir, out = trained
    assert cli.main(["predict", "--data", str(dataset), "--model", str(model_dir),
                     "--out", str(tmp_path)]) == 0
    for name in ("matching_results.tsv", "candidate_pairs.tsv"):
        assert (tmp_path / name).read_bytes() == (out / name).read_bytes()


def test_bundle_round_trip_keeps_config(trained):
    model_dir, _ = trained
    bundle = pipeline.Bundle.load(model_dir)
    assert bundle.config.folds == 3
    assert bundle.rule["kind"] in {"threshold", "expected_f05"}
    assert [len(st.boosters) for st in bundle.stages] == [3, 3, 3]
    assert bundle.stages[1].features[:len(bundle.stages[0].features)] == bundle.stages[0].features
    assert any(f.startswith("tok_") for f in bundle.stages[0].features)  # alignment features in use
    assert bundle.stages[2].features[0] == "oof"


def test_bad_prefix_is_rejected(dataset, tmp_path):
    bad = tmp_path / "train"
    bad.mkdir()
    for f in (dataset / "train").iterdir():
        (bad / f.name).write_text(f.read_text())
    s2 = bad / "train_source2.tsv"
    s2.write_text(s2.read_text().replace("S2-00000", "S3-00000", 1))
    with pytest.raises(data.DataError, match="prefix S2-"):
        data.load_split(tmp_path, "train")


def test_missing_column_is_rejected(tmp_path):
    (tmp_path / "x.tsv").write_text("entity_id\tbusiness_name\nS1-1\tA\n")
    with pytest.raises(data.DataError, match="missing columns"):
        data.read_tsv(tmp_path / "x.tsv", data.SOURCE_COLUMNS)


def test_quote_characters_do_not_break_rows(tmp_path):
    (tmp_path / "x.tsv").write_text(
        'entity_id\tbusiness_name\tbusiness_address\tcountry\n'
        'S1-1\t"Joe\'s "Best\tMain St\tUS\nS1-2\tB\tC\tUS\n')
    assert data.read_tsv(tmp_path / "x.tsv", data.SOURCE_COLUMNS).height == 2


def test_submission_check_catches_violations():
    ids = pl.Series(["S1-1", "S1-2"])
    good = pl.DataFrame({"s1": ["S1-1", "S1-2"], "id": ["S2-1", ""]})
    data.check_submission(ids, good, good)
    with pytest.raises(data.DataError, match="Source 1 ids"):
        bad = pl.DataFrame({"s1": ["S1-1", "S1-2"], "id": ["S1-9", ""]})
        data.check_submission(ids, bad, bad)
    with pytest.raises(data.DataError, match="missing from the candidate set"):
        data.check_submission(ids, good, pl.DataFrame({"s1": ["S1-1", "S1-2"], "id": ["", ""]}))
    with pytest.raises(data.DataError, match="exactly the test Source 1 ids"):
        data.check_submission(ids, good.head(1), good)
    with pytest.raises(data.DataError, match="duplicate"):
        dup = pl.DataFrame({"s1": ["S1-1", "S1-2"], "id": ["S2-1,S2-1", ""]})
        data.check_submission(ids, dup, dup)


def test_missing_bundle_gives_exit_code_2(dataset, tmp_path):
    assert cli.main(["predict", "--data", str(dataset), "--model", str(tmp_path / "none"),
                     "--out", str(tmp_path)]) == 2


def test_default_config_round_trips():
    cfg = Config()
    assert Config.from_dict(cfg.to_dict()) == cfg


def test_expected_f05_rule_keeps_the_best_prefix_or_nothing():
    import numpy as np
    keep = pipeline.expected_f05_mask(np.array([0, 0, 0, 1]), np.array([0.9, 0.8, 0.1, 0.05]), 0.0)
    assert keep.tolist() == [True, True, False, False]  # group 1 is probably a singleton
    keep = pipeline.expected_f05_mask(np.array([5, 5, 2]), np.array([0.3, 0.95, 0.6]), 0.2)
    assert keep.tolist() == [False, True, True]


def test_cached_run_matches_uncached(dataset, trained, tmp_path):
    model_dir, out = trained
    cache = tmp_path / "cache"
    for n in (1, 2):  # first run fills the cache, second reads it
        dest = tmp_path / f"out{n}"
        assert cli.main(["predict", "--data", str(dataset), "--model", str(model_dir),
                         "--out", str(dest), "--cache", str(cache)]) == 0
        for name in ("matching_results.tsv", "candidate_pairs.tsv"):
            assert (dest / name).read_bytes() == (out / name).read_bytes()
    assert len(list(cache.iterdir())) == 1


def test_decide_from_saved_scores_matches_predict(dataset, trained, tmp_path):
    model_dir, out = trained
    assert (model_dir / "test_scores" / "pairs.parquet").is_file()
    assert cli.main(["decide", "--model", str(model_dir), "--out", str(tmp_path)]) == 0
    for name in ("matching_results.tsv", "candidate_pairs.tsv"):
        assert (tmp_path / name).read_bytes() == (out / name).read_bytes()


def test_unseen_country_calibration_only_touches_unseen_countries(trained, tmp_path):
    import json
    model_dir, out = trained
    report = json.loads((model_dir / "predict_report.json").read_text())
    assert set(report["unseen_country_calibration"]) <= {"france"}
    off = tmp_path / "off"
    assert cli.main(["decide", "--model", str(model_dir), "--out", str(off), "--unseen-none-cap", "1"]) == 0
    a = _read(out / "matching_results.tsv")
    b = _read(off / "matching_results.tsv")
    # Rows for seen countries are identical with and without the calibration.
    recs = pl.read_parquet(model_dir / "test_scores" / "records.parquet").filter(pl.col("src") == 1)
    seen = recs.filter(pl.col("country_key") != "france")["entity_id"]
    ja = a.filter(pl.col("source1_entity_id").is_in(seen.implode()))
    jb = b.filter(pl.col("source1_entity_id").is_in(seen.implode()))
    assert ja.equals(jb)
    # With the calibration France never gains matches.
    fr = recs.filter(pl.col("country_key") == "france")["entity_id"]
    na = a.filter(pl.col("source1_entity_id").is_in(fr.implode()))["matched_entity_ids"].str.len_chars().sum()
    nb = b.filter(pl.col("source1_entity_id").is_in(fr.implode()))["matched_entity_ids"].str.len_chars().sum()
    assert na <= nb


def test_decide_can_switch_stage3_variant(trained, tmp_path):
    model_dir, _ = trained
    bundle = pipeline.Bundle.load(model_dir)
    assert set(bundle.variants) == {"robust"}
    assert not any("nov" in f or "h_dev" in f for f in bundle.variants["robust"].features)
    _, _, probs = pipeline.load_scores(model_dir)
    assert set(probs) == {"stage2", "full", "robust"}
    for choice in ("robust", "none"):
        assert cli.main(["decide", "--model", str(model_dir), "--out", str(tmp_path / choice),
                         "--stage3", choice]) == 0
        assert (tmp_path / choice / "matching_results.tsv").is_file()


def test_per_country_stage3_choice(trained, tmp_path):
    model_dir, _ = trained
    out = tmp_path / "mixed"
    assert cli.main(["decide", "--model", str(model_dir), "--out", str(out), "--stage3", "none",
                     "--stage3-country", "india=full"]) == 0
    records, pairs, probs = pipeline.load_scores(model_dir)
    bundle = pipeline.Bundle.load(model_dir)
    import dataclasses
    bundle.config = dataclasses.replace(bundle.config, stage3_use="none", stage3_by_country={"india": "full"})
    country = records["country_key"].to_numpy()[pairs["i"].to_numpy()]
    prob = pipeline.select_probability(probs, bundle, country)
    india = country == "india"
    assert (prob[india] == probs["full"][india]).all() and (prob[~india] == probs["stage2"][~india]).all()


def test_feature_caps_touch_only_named_columns_and_masked_rows():
    import numpy as np
    x = np.array([[3.0, 5.0], [2.0, 0.5], [9.0, 9.0]], dtype=np.float32)
    pipeline.apply_feature_caps(x, ["amb_s23_addr_2", "other"], {"amb_s23_addr_2": 1.0},
                                np.array([True, False, True]))
    assert x.tolist() == [[1.0, 5.0], [2.0, 0.5], [1.0, 9.0]]
    pipeline.apply_feature_caps(x, ["a", "b"], {"missing": 0.0}, None)  # unknown name: no-op
    assert x.tolist() == [[1.0, 5.0], [2.0, 0.5], [1.0, 9.0]]


def test_predict_with_cap_changes_only_capped_country(dataset, trained, tmp_path):
    model_dir, out = trained
    bundle = pipeline.Bundle.load(model_dir)
    import dataclasses
    bundle.config = dataclasses.replace(bundle.config, feature_caps={"amb_s23_addr_2": 0.0},
                                        feature_cap_countries=["us"])
    ids, matches, cands, _ = pipeline.predict(dataset, bundle, bundle_dir=tmp_path)
    base = _read(out / "matching_results.tsv")
    recs = pl.read_parquet(tmp_path / "test_scores" / "records.parquet").filter(pl.col("src") == 1)
    got = matches.rename({"s1": "source1_entity_id", "id": "matched_entity_ids"})
    for country, may_change in (("india", False), ("france", False)):
        sub = recs.filter(pl.col("country_key") == country)["entity_id"]
        a = base.filter(pl.col("source1_entity_id").is_in(sub.implode()))
        b = got.filter(pl.col("source1_entity_id").is_in(sub.implode()))
        assert a.equals(b) or may_change
