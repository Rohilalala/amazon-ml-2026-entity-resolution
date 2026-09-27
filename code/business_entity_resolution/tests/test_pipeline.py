"""End-to-end and data-contract tests on a synthetic dataset (no network)."""
import json
import shutil
import subprocess
import sys
from pathlib import Path

import numpy as np
import polars as pl
import pytest

from src import alignment, cli, data, pair_encoder, pipeline, stage4
from src.config import Config
from tests.synth import make_dataset, make_tiny_encoder

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


def test_vocab_command_rebuilds_the_training_tables(dataset, trained, tmp_path):
    model_dir, _ = trained
    saved = pipeline.Bundle.load(model_dir).vocab
    assert set(saved) == set(alignment.VOCAB) and set(saved["floor"]["country_key"]) == {"us", "india"}
    copy = tmp_path / "model"
    shutil.copytree(model_dir, copy)
    for k in alignment.VOCAB:
        (copy / pipeline.VOCAB_FILE.format(k)).unlink()
    assert pipeline.Bundle.load(copy).vocab is None
    assert cli.main(["vocab", "--data", str(dataset), "--model", str(copy)]) == 0
    rebuilt = pipeline.Bundle.load(copy).vocab
    for k in alignment.VOCAB:
        assert rebuilt[k].sort(rebuilt[k].columns).equals(saved[k].sort(saved[k].columns))


def test_configure_pins_the_unseen_target_and_enables_unseen_rules(trained, tmp_path):
    model_dir, _ = trained
    copy = tmp_path / "model"
    shutil.copytree(model_dir, copy)
    assert cli.main(["configure", "--model", str(copy), "--set", "unseen_empty_target=0.5",
                     "--set", "unseen_rules=true"]) == 0
    bundle = pipeline.Bundle.load(copy)
    assert bundle.config.unseen_empty_target == 0.5 and bundle.config.unseen_rules
    assert cli.main(["decide", "--model", str(copy), "--out", str(tmp_path / "out")]) == 0
    france = json.loads((copy / "predict_report.json").read_text())["unseen_country_calibration"]["france"]
    assert {"legal_form_drops", "true_variant_adds", "tail_adds"} <= set(france)
    if bundle.rule["kind"] == "expected_f05":
        assert france["target"] == 0.5
    assert cli.main(["configure", "--model", str(copy), "--set", "no_such_setting=1"]) == 2
    assert cli.main(["configure", "--model", str(copy), "--set", "unseen_rules"]) == 2
    records = copy / "test_scores" / "records.parquet"
    pl.read_parquet(records).drop("name").write_parquet(records)  # scores saved before unseen_rules existed
    assert cli.main(["decide", "--model", str(copy), "--out", str(tmp_path / "out")]) == 2


@pytest.fixture(scope="module")
def stage4_model(dataset, trained, tmp_path_factory):
    """The trained bundle plus stage 4 (a tiny encoder fine-tuned on CPU), predicted with
    stage4_scope "all", then re-decided with "seen" and "none"."""
    model_dir, _ = trained
    copy, base, out = (tmp_path_factory.mktemp(n) for n in ("model4", "base", "out4"))
    shutil.copytree(model_dir, copy, dirs_exist_ok=True)
    raw = pl.concat([data.load_split(dataset, s) for s in ("train", "test")])
    make_tiny_encoder(base, pair_encoder.texts(raw).to_list())
    small = {**Config().ce_params, "train_s1": 60, "val_s1": 10, "batch_size": 16, "infer_batch": 64, "device": "cpu"}
    assert cli.main(["configure", "--model", str(copy), "--set", "holdout_s1=20", "--set",
                     f"ce_params={json.dumps(small)}", "--set", 'stage4_scope="all"']) == 0
    assert cli.main(["stage4-train", "--data", str(dataset), "--model", str(copy), "--base", str(base)]) == 0
    assert cli.main(["predict", "--data", str(dataset), "--model", str(copy), "--out", str(out / "all")]) == 0
    for scope in ("seen", "none"):
        assert cli.main(["decide", "--model", str(copy), "--out", str(out / scope), "--stage4-scope", scope]) == 0
    return copy, out


def test_stage4_train_saves_encoder_and_stacker(stage4_model):
    model_dir, _ = stage4_model
    bundle = pipeline.Bundle.load(model_dir)
    assert bundle.stage4.features == stage4.FEATURES and bundle.stage4_rule["kind"] in {"threshold", "expected_f05"}
    report = bundle.report["stage4"]
    assert report["active_pairs"] > 0 and report["encoder"]["fit_pairs"] > 0 and 0 < report["oof_f05"] <= 1
    assert (model_dir / pair_encoder.ENCODER_DIR / "model.safetensors").is_file()
    records, pairs, probs = pipeline.load_scores(model_dir)
    country = records["country_key"].to_numpy()[pairs["i"].to_numpy()]
    active = pipeline.select_probability(probs, bundle, country) >= bundle.config.stage4_active
    assert np.isfinite(pairs["ce_logit"].to_numpy()[active]).all() and np.isnan(pairs["ce_logit"].to_numpy()[~active]).all()


def test_stage4_scope_decides_only_its_countries(dataset, stage4_model):
    _, out = stage4_model
    s1 = _read(dataset / "test" / "test_source1.tsv").select("entity_id", pl.col("country").str.to_lowercase())
    rows = {k: _read(out / k / "matching_results.tsv").join(s1, left_on="source1_entity_id", right_on="entity_id")
            for k in ("all", "seen", "none")}

    def part(k, countries):
        return rows[k].filter(pl.col("country").is_in(countries))
    assert part("seen", ["us", "india"]).equals(part("all", ["us", "india"]))
    assert part("seen", ["france"]).equals(part("none", ["france"]))


@pytest.mark.skipif(not VALIDATOR.is_file(), reason="challenge validator not present")
def test_stage4_output_passes_the_official_validator(dataset, stage4_model):
    _, out = stage4_model
    res = subprocess.run([sys.executable, str(VALIDATOR), "--matching", str(out / "all" / "matching_results.tsv"),
                          "--candidate", str(out / "all" / "candidate_pairs.tsv"),
                          "--test-dir", str(dataset / "test"), "--check-ids"], capture_output=True, text=True)
    assert res.returncode == 0, res.stdout + res.stderr


def test_stage4_import_copies_encoder_and_stacker(stage4_model, trained, tmp_path):
    src, _ = stage4_model
    model_dir, _ = trained
    copy = tmp_path / "model"
    shutil.copytree(model_dir, copy)
    meta = tmp_path / "stage4.json"
    rule = pipeline.Bundle.load(src).stage4_rule
    meta.write_text(json.dumps({"features": stage4.FEATURES, "rule": rule}))
    args = ["stage4-import", "--model", str(copy), "--encoder", str(src / pair_encoder.ENCODER_DIR),
            "--stacker", str(src / pipeline.STAGE4_FILE), "--stacker-meta", str(meta)]
    assert cli.main(args) == 0
    bundle = pipeline.Bundle.load(copy)
    assert bundle.stage4_rule == rule and (copy / pipeline.STAGE4_FILE).read_bytes() == (src / pipeline.STAGE4_FILE).read_bytes()
    assert str(src) not in (copy / pipeline.BUNDLE_FILE).read_text()  # file names only, no local paths
    meta.write_text(json.dumps({"features": stage4.FEATURES[:-1], "rule": rule}))
    assert cli.main(args) == 2


@pytest.fixture(scope="module")
def final_model(dataset, trained, tmp_path_factory):
    """The trained bundle with an ACNRST stage 4 (two seed members, tiny encoder), the recall
    pieces (tiny encoders, a small head) and one list row, run through `final`."""
    import lightgbm as lgb
    from src import final, recall
    model_dir, _ = trained
    copy, base, out = (tmp_path_factory.mktemp(n) for n in ("model_final", "base_final", "out_final"))
    shutil.copytree(model_dir, copy, dirs_exist_ok=True)
    raw = pl.concat([data.load_split(dataset, s) for s in ("train", "test")])
    make_tiny_encoder(base, pair_encoder.texts(raw).to_list())
    small = {**Config().ce_params, "train_s1": 60, "val_s1": 10, "batch_size": 16, "infer_batch": 64, "device": "cpu"}
    assert cli.main(["configure", "--model", str(copy), "--set", "holdout_s1=20", "--set", f"ce_params={json.dumps(small)}",
                     "--set", 'stage4_scope="all"', "--set", 'stage4_features="ACNRST"', "--set", "stage4_seeds=[1, 2]"]) == 0
    assert cli.main(["stage4-train", "--data", str(dataset), "--model", str(copy), "--base", str(base)]) == 0
    rng = np.random.default_rng(0)
    x = rng.random((400, len(recall.HEAD_FEATURES)))
    head = tmp_path_factory.mktemp("head")
    lgb.train({"objective": "binary", "verbose": -1}, lgb.Dataset(x, (x[:, 0] > 0.3).astype(int)), 5).save_model(head / "head.txt")
    (head / "head.json").write_text(json.dumps({"features": recall.HEAD_FEATURES, "t_keep": 0.8, "t_none": 0.75}))
    s1 = data.load_split(dataset, "test").filter(pl.col("src") == 1)["entity_id"][0]
    pl.DataFrame({"step": ["H"], "action": ["drop"], "rule": ["example"], "s1": [s1], "m": ["S2-99999"]},
                 schema_overrides={"src": pl.List(pl.String)}).with_columns(pl.lit(None, pl.List(pl.String)).alias("src")) \
        .write_parquet(head / "lists.parquet")
    assert cli.main(["post-import", "--model", str(copy), "--retriever", str(base), "--recall-encoder", str(base),
                     "--head", str(head / "head.txt"), "--head-meta", str(head / "head.json"),
                     "--lists", str(head / "lists.parquet")]) == 0
    assert cli.main(["final", "--data", str(dataset), "--model", str(copy), "--out", str(out / "a")]) == 0
    return copy, out, final


def test_final_stage4_members_and_outputs(dataset, final_model):
    model_dir, out, _ = final_model
    bundle = pipeline.Bundle.load(model_dir)
    assert bundle.stage4.features == stage4.FEATURE_SETS["ACNRST"] and len(bundle.stage4.boosters) == 2
    assert bundle.report["stage4"]["seeds"] == [1, 2] and len(bundle.report["stage4"]["fold_rounds"]) == 2
    report = json.loads((model_dir / "predict_report.json").read_text())
    assert set(report["post"]) == {"B_recall", "C_sister", "D_restore", "E_lists", "F_lists", "G_lists", "H_lists"}
    assert report["post"]["B_recall"]["new_pairs"] > 0
    if VALIDATOR.is_file():
        res = subprocess.run([sys.executable, str(VALIDATOR), "--matching", str(out / "a" / "matching_results.tsv"),
                              "--candidate", str(out / "a" / "candidate_pairs.tsv"), "--test-dir", str(dataset / "test"),
                              "--check-ids"], capture_output=True, text=True)
        assert res.returncode == 0, res.stdout + res.stderr


def test_final_with_a_saved_recall_list_matches(dataset, final_model):
    model_dir, out, final = final_model
    cached = model_dir / "test_scores" / "recall_pairs.parquet"
    fixed = out / "recall_pairs.parquet"
    pl.read_parquet(cached).drop("prob").write_parquet(fixed)
    cached.unlink()
    assert cli.main(["final", "--data", str(dataset), "--model", str(model_dir), "--out", str(out / "b"),
                     "--recall-pairs", str(fixed)]) == 0
    for f in ("matching_results.tsv", "candidate_pairs.tsv"):
        assert (out / "a" / f).read_bytes() == (out / "b" / f).read_bytes()


def test_encoder_and_recall_head_training(dataset, final_model, tmp_path):
    model_dir, _, final = final_model
    copy = tmp_path / "model"
    shutil.copytree(model_dir, copy, ignore=shutil.ignore_patterns("test_scores"))
    enc1, enc2 = tmp_path / "enc1", tmp_path / "enc2"
    args = ["encoder-train", "--data", str(dataset), "--model", str(copy), "--init", str(copy / pair_encoder.ENCODER_DIR)]
    assert cli.main([*args, "--out", str(enc1), "--draw", "30"]) == 0
    assert cli.main([*args, "--out", str(enc2), "--exclude", str(enc1)]) == 0
    d1, d2 = (pl.read_parquet(d / "s1_draw.parquet")["i"] for d in (enc1, enc2))
    assert d1.len() == 30 and not d2.is_in(d1.implode()).any() and (enc2 / "model.safetensors").is_file()
    assert cli.main(["recall-train", "--data", str(dataset), "--model", str(copy)]) == 0
    meta = json.loads((copy / final.RECALL_DIR / "head.json").read_text())
    assert meta["t_keep"] in [round(x, 2) for x in np.arange(0.20, 0.96, 0.05)] + [0.97, 0.98, 0.99, 1.01]
