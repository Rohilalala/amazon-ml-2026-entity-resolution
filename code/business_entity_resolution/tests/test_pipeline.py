"""End-to-end and data-contract tests on a synthetic dataset (no network)."""
import pandas as pd
import pytest

from src import cli, data, pipeline
from src.config import Config
from tests.synth import make_dataset


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
                     "--out", str(out_dir), "--folds", "3"])
    assert code == 0
    return model_dir, out_dir


def _read(path):
    return pd.read_csv(path, sep="\t", dtype=str, keep_default_na=False)


def test_outputs_follow_submission_rules(dataset, trained):
    _, out = trained
    test_s1 = _read(dataset / "test" / "test_source1.tsv")["entity_id"].tolist()
    valid = set(_read(dataset / "test" / "test_source2.tsv")["entity_id"]) | \
        set(_read(dataset / "test" / "test_source3.tsv")["entity_id"])
    match = _read(out / "matching_results.tsv")
    cand = _read(out / "candidate_pairs.tsv")
    assert list(match.columns) == ["source1_entity_id", "matched_entity_ids"]
    assert list(cand.columns) == ["source1_entity_id", "candidate_entity_ids"]
    assert match["source1_entity_id"].tolist() == test_s1
    assert cand["source1_entity_id"].tolist() == test_s1
    for m, c in zip(match["matched_entity_ids"], cand["candidate_entity_ids"]):
        m, c = data.parse_ids(m), data.parse_ids(c)
        assert len(m) == len(set(m)) and len(c) == len(set(c))
        assert set(m) <= set(c) <= valid


def test_model_quality_and_unseen_country(trained):
    model_dir, _ = trained
    bundle = pipeline.Bundle.load(model_dir)
    report = bundle.report
    assert report["pair_recall"] > 0.95
    assert report["oof_f05"] > 0.85
    assert set(report["unseen_country"]) == {"us", "india"}
    for v in report["unseen_country"].values():
        assert v["f05_at_tuned_threshold"] > 0.8
    # The unseen country gets predictions too.
    assert (model_dir / "predict_report.json").is_file()


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
    assert 0 < bundle.threshold < 1
    assert len(bundle.boosters) == 3


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


def test_submission_check_catches_violations():
    with pytest.raises(data.DataError, match="not in test Source 2/3"):
        data.check_submission(["S1-1"], {"S1-1": {"S1-9"}}, {"S1-1": {"S1-9"}}, {"S2-1"})
    with pytest.raises(data.DataError, match="missing from the candidate set"):
        data.check_submission(["S1-1"], {"S1-1": {"S2-1"}}, {}, {"S2-1"})
    data.check_submission(["S1-1"], {"S1-1": {"S2-1"}}, {"S1-1": {"S2-1"}}, {"S2-1"})


def test_missing_bundle_gives_exit_code_2(dataset, tmp_path):
    assert cli.main(["predict", "--data", str(dataset), "--model", str(tmp_path / "none"),
                     "--out", str(tmp_path)]) == 2


def test_default_config_round_trips():
    cfg = Config()
    assert Config.from_dict(cfg.to_dict()) == cfg
