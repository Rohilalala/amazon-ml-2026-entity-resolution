"""Stage 4 pieces on small hand-made inputs: pair text, features, scope routing, decision, config."""
import math
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import polars as pl
import pytest

from src import cli, pair_encoder, pipeline, stage4
from src.config import Config
from src.data import DataError


def test_pair_text_is_the_lowercased_raw_name_and_address():
    raw = pl.DataFrame({"business_name": ["  Café Lumière SARL ", "ACME Inc", None],
                        "business_address": ["12 Rue X, Lyon", None, " 5 Main St "]})
    assert pair_encoder.texts(raw).to_list() == ["café lumière sarl | 12 rue x, lyon", "acme inc | ",
                                                 " | 5 main st"]


def test_importing_the_pair_encoder_does_not_load_torch():
    # torch's OpenMP runtime loaded before LightGBM's makes LightGBM segfault (macOS), so only
    # the spawned encoder process may import torch.
    code = "import sys; from src import pair_encoder, pipeline; sys.exit('torch' in sys.modules)"
    assert subprocess.run([sys.executable, "-c", code], cwd=Path(__file__).resolve().parents[1]).returncode == 0


def test_empty_address_reads_the_text_after_the_first_bar():
    text = pl.Series(["acme | ", "acme |  ", "acme | 5 main st", "acme | www.acme.com | "])
    # The last name contains ' | ': read as non-empty, exactly as stage 4 was trained.
    assert pair_encoder.empty_address(text).tolist() == [True, True, False, False]


def test_stage4_features_by_hand():
    df = pl.DataFrame({"i": [0, 0, 0, 1], "j": [0, 1, 2, 3], "prob": [0.9, 0.5, 0.9, 0.2],
                       "ce_logit": [2.0, 3.0, 1.0, -1.0]}, schema_overrides={"prob": pl.Float32})
    f = stage4.features(df, np.array([False, True, False, True]))
    lp = [math.log(float(np.float32(p)) / (1 - float(np.float32(p)))) for p in (0.9, 0.5, 0.9, 0.2)]
    assert f["lp"].to_list() == pytest.approx(lp)
    assert f["b_empty"].to_list() == [0, 1, 0, 1]
    assert f["ce_rank"].to_list() == [2, 1, 3, 1]
    assert f["ce_gap"].to_list() == [2 - 3, 3 - 2, 1 - 3, None]  # gap to the best other candidate
    assert f["lp_rank"].to_list() == [1, 3, 1, 1]  # tie at the top: both rank 1, gap 0
    assert f["lp_gap"].to_list()[:3] == pytest.approx([0.0, lp[1] - lp[0], 0.0]) and f["lp_gap"][3] is None
    assert f["n_i"].to_list() == [3, 3, 3, 1]
    assert stage4.matrix(f).shape == (4, len(stage4.FEATURES)) and stage4.matrix(f).dtype == np.float32


def test_scope_routing():
    seen, present = {"us", "india"}, ["france", "india", "us"]
    assert stage4.countries("none", seen, present) == []
    assert stage4.countries("seen", seen, present) == ["india", "us"]
    assert stage4.countries("all", seen, present) == present
    with pytest.raises(DataError):
        stage4.countries("some", seen, present)


def test_stage4_probability_rescores_only_active_pairs():
    class Model:
        def predict(self, x):
            return x[:, stage4.FEATURES.index("ce")].astype(np.float64) / 10
    prob = np.array([0.5, 0.001, 0.3], dtype=np.float32)
    out = stage4.probability(Model(), np.array([0, 0, 1]), np.array([0, 1, 2]), prob,
                             np.array([2.0, np.nan, 5.0], dtype=np.float32), np.zeros(3, bool), prob >= 0.002)
    assert out.dtype == np.float64 and out.tolist() == [0.2, float(np.float32(0.001)), 0.5]


def test_decide_uses_the_stage4_rule_only_for_stage4_countries():
    bundle = SimpleNamespace(config=Config(exclusive=False), rule={"kind": "threshold", "threshold": 0.5},
                             stage4_rule={"kind": "threshold", "threshold": 0.8}, report={})
    i, j = np.array([0, 1, 2, 3]), np.array([4, 5, 6, 7])
    prob = np.array([0.6, 0.9, 0.6, 0.9])
    country = np.array(["us", "us", "fr", "fr"])
    keep, _ = pipeline.decide(i, j, prob, country, country, bundle, ["us"])
    assert keep.tolist() == [False, True, True, True]
    keep, _ = pipeline.decide(i, j, prob, country, country, bundle)
    assert keep.tolist() == [True, True, True, True]


def test_config_round_trip_and_bundles_saved_before_stage4():
    cfg = Config(stage4_scope="all", ce_params={**Config().ce_params, "top_k": 4})
    assert Config.from_dict(cfg.to_dict()) == cfg
    old = Config().to_dict()
    for name in ("stage4_scope", "stage4_active", "ce_params"):
        del old[name]
    assert Config.from_dict(old).stage4_scope == "none" and Config.from_dict(old).stage4_active == 0.002


def test_configure_rejects_an_unknown_scope(tmp_path):
    (tmp_path / pipeline.BUNDLE_FILE).write_text(__import__("json").dumps({"config": Config().to_dict()}))
    assert cli.main(["configure", "--model", str(tmp_path), "--set", 'stage4_scope="seen"']) == 0
    assert cli.main(["configure", "--model", str(tmp_path), "--set", 'stage4_scope="some"']) == 2


def test_isolated_child_does_not_import_the_callers_main(tmp_path):
    # The child must not re-import the caller (it would load LightGBM next to torch); run as a script and as -m.
    root = Path(__file__).resolve().parents[1]
    (tmp_path / "caller.py").write_text(f"import sys\nsys.path.insert(0, {str(root)!r})\nprint('imported', flush=True)\n"
                                        "from src import pair_encoder\nif __name__ == '__main__':\n"
                                        "    print(pair_encoder._isolated(sorted, [2, 1]), flush=True)\n")
    for args in ([str(tmp_path / "caller.py")], ["-m", "caller"]):
        res = subprocess.run([sys.executable, *args], capture_output=True, text=True, cwd=tmp_path, timeout=120)
        assert res.stdout.split("\n")[:2] == ["imported", "[1, 2]"], res.stdout + res.stderr
