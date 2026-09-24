"""Training and prediction: blocking -> pair features -> LightGBM -> decision rule.

Training produces a Bundle (fold models, feature list, tuned threshold, config,
metrics, data fingerprints). Prediction only reads the Bundle, so the test run
uses exactly the settings that were validated.
"""
import hashlib
import json
import logging
import platform
import subprocess
import time
from dataclasses import dataclass
from importlib import metadata
from pathlib import Path

import lightgbm as lgb
import numpy as np
import pandas as pd
from sklearn.model_selection import GroupKFold

from . import candidates, data, features
from .config import Config
from .metric import f05_macro

log = logging.getLogger(__name__)

ID_COLS = ("s1", "cand")
THRESHOLDS = np.round(np.arange(0.05, 0.96, 0.01), 2)
BUNDLE_FILE = "bundle.json"
PACKAGES = ("numpy", "pandas", "scikit-learn", "scipy", "lightgbm", "rapidfuzz")


# ---------------------------------------------------------------- stages

def pair_table(recs: pd.DataFrame, cfg: Config) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Prepared records and the feature table of their candidate pairs."""
    t0 = time.time()
    recs = features.prepare(recs)
    mats = features.matrices(recs)
    pairs = candidates.generate(recs, [mats["name"], mats["full"]], cfg.k, cfg.k_rev,
                                cfg.block_by_country)
    table = features.build(recs, mats, pairs)
    log.info("%d candidate pairs, %d features in %.1fs",
             len(table), table.shape[1] - len(ID_COLS), time.time() - t0)
    return recs, table


def source1_ids(recs: pd.DataFrame) -> list[str]:
    return recs.loc[recs["src"] == 1, "entity_id"].tolist()


def labels(pairs: pd.DataFrame, truth: dict) -> np.ndarray:
    """1 where the candidate is a true match of the Source 1 record."""
    empty = frozenset()
    return np.fromiter((c in truth.get(s, empty) for s, c in zip(pairs["s1"], pairs["cand"])),
                       dtype=np.int8, count=len(pairs))


def best_for_cand(pairs: pd.DataFrame, prob: np.ndarray) -> np.ndarray:
    """True where a pair has the top probability among all Source 1 records
    competing for the same Source 2/3 record."""
    top = pd.Series(prob, index=pairs.index).groupby(pairs["cand"].to_numpy()).transform("max")
    return prob >= top.to_numpy()


def decide(pairs: pd.DataFrame, prob: np.ndarray, threshold: float, exclusive: bool) -> dict:
    """Final matches {s1: frozenset(cands)}: probability >= threshold and, when
    `exclusive`, each Source 2/3 record assigned to its best Source 1 record only."""
    keep = prob >= threshold
    if exclusive:
        keep &= best_for_cand(pairs, prob)
    return pairs.loc[keep].groupby("s1")["cand"].agg(frozenset).to_dict()


def tune(pairs, prob, y, truth, ids) -> tuple[float, float, bool]:
    """(macro F0.5, threshold, exclusive) maximising the metric on `ids`.

    Vectorised form of metric.f05_macro: per Source 1 entity, predicted and
    correct counts come from bincount instead of set operations. Train()
    cross-checks the winner against the reference implementation.
    """
    codes = pd.Index(ids).get_indexer(pairs["s1"])
    inside = codes >= 0
    codes, prob, y = codes[inside], prob[inside], y[inside]
    n_true = np.array([len(truth.get(i, ())) for i in ids], dtype=float)
    best = best_for_cand(pairs[inside], prob)
    results = []
    for exclusive in (True, False):
        for t in THRESHOLDS:
            sel = (prob >= t) & best if exclusive else prob >= t
            n_pred = np.bincount(codes, weights=sel, minlength=len(ids))
            tp = np.bincount(codes, weights=sel & (y == 1), minlength=len(ids))
            with np.errstate(divide="ignore", invalid="ignore"):
                p, r = tp / n_pred, tp / n_true
                f = np.where(tp > 0, 1.25 * p * r / (0.25 * p + r), 0.0)
            f = np.where(n_true == 0, (n_pred == 0).astype(float), f)
            results.append((float(f.mean()), float(t), exclusive))
    return max(results)


def fit(X, y, cfg: Config, valid=None, rounds=None) -> lgb.Booster:
    """One LightGBM model: early-stopped on `valid`, or a fixed number of rounds."""
    train_set = lgb.Dataset(X, y, free_raw_data=False)
    if valid is None:
        return lgb.train(cfg.lgb(), train_set, rounds)
    return lgb.train(cfg.lgb(), train_set, cfg.num_boost_round,
                     valid_sets=[lgb.Dataset(*valid, reference=train_set)],
                     callbacks=[lgb.early_stopping(cfg.early_stopping_rounds, verbose=False)])


def cross_validate(X, y, groups, cfg: Config) -> tuple[np.ndarray, list[lgb.Booster]]:
    """Out-of-fold probabilities, folds grouped by Source 1 entity so no entity's
    pairs are split between training and validation."""
    oof, boosters = np.zeros(len(y)), []
    splitter = GroupKFold(n_splits=cfg.folds, shuffle=True, random_state=cfg.seed)
    for n, (tr, va) in enumerate(splitter.split(X, y, groups)):
        b = fit(X.iloc[tr], y[tr], cfg, valid=(X.iloc[va], y[va]))
        oof[va] = b.predict(X.iloc[va], num_iteration=b.best_iteration)
        boosters.append(b)
        log.info("fold %d/%d: %d rounds", n + 1, cfg.folds, b.best_iteration)
    return oof, boosters


# ---------------------------------------------------------------- bundle

@dataclass
class Bundle:
    config: Config
    features: list[str]
    threshold: float
    exclusive: bool
    boosters: list[lgb.Booster]
    report: dict

    def predict(self, table: pd.DataFrame) -> np.ndarray:
        """Mean probability of the fold models."""
        X = table[self.features]
        return np.mean([b.predict(X, num_iteration=b.best_iteration) for b in self.boosters], axis=0)

    def save(self, directory: Path) -> None:
        directory.mkdir(parents=True, exist_ok=True)
        files = []
        for n, b in enumerate(self.boosters):
            name = f"fold_{n}.txt"
            b.save_model(directory / name, num_iteration=b.best_iteration)
            files.append(name)
        meta = {"config": self.config.to_dict(), "features": self.features,
                "threshold": self.threshold, "exclusive": self.exclusive,
                "models": files, "report": self.report}
        (directory / BUNDLE_FILE).write_text(json.dumps(meta, indent=2))
        log.info("saved bundle to %s", directory)

    @classmethod
    def load(cls, directory: Path) -> "Bundle":
        path = directory / BUNDLE_FILE
        if not path.is_file():
            raise data.DataError(f"no model bundle at {directory}; run `train` first")
        meta = json.loads(path.read_text())
        boosters = [lgb.Booster(model_file=str(directory / f)) for f in meta["models"]]
        for b in boosters:
            if b.feature_name() != meta["features"]:
                raise data.DataError(f"{directory}: model features do not match the bundle")
        return cls(Config.from_dict(meta["config"]), meta["features"], meta["threshold"],
                   meta["exclusive"], boosters, meta["report"])


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for block in iter(lambda: fh.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def provenance(data_dir: Path, split: str) -> dict:
    """Input file hashes, package versions and git commit, for reproducibility."""
    files = sorted((data_dir / split).glob("*.tsv"))
    try:
        commit = subprocess.run(["git", "rev-parse", "--short", "HEAD"], capture_output=True,
                                text=True, cwd=Path(__file__).parent, timeout=5).stdout.strip()
    except (OSError, subprocess.SubprocessError):
        commit = ""
    return {"inputs": {f.name: _sha256(f) for f in files},
            "packages": {p: metadata.version(p) for p in PACKAGES},
            "python": platform.python_version(), "git_commit": commit or None}


# ---------------------------------------------------------------- entry points

def _per_country(pairs, prob, t, exclusive, truth, ids, country_of) -> dict:
    pred = decide(pairs, prob, t, exclusive)
    out = {}
    for c in sorted({country_of[i] for i in ids}):
        sub = [i for i in ids if country_of[i] == c]
        out[c] = {"s1": len(sub), "f05": round(f05_macro(pred, truth, sub), 5)}
    return out


def _country_transfer(pairs, X, y, truth, ids, country_of, cfg, rounds, t, exclusive) -> dict:
    """Leave-one-country-out: train on the other countries, score the held-out
    one at the tuned threshold. Estimates quality on a country never seen in
    training (the test set adds France)."""
    s1_country = pairs["s1"].map(country_of).to_numpy()
    out = {}
    for c in sorted(set(s1_country)):
        held = s1_country == c
        if held.all():
            continue
        b = fit(X[~held], y[~held], cfg, rounds=rounds)
        prob = b.predict(X[held])
        sub = [i for i in ids if country_of[i] == c]
        pred = decide(pairs[held], prob, t, exclusive)
        oracle = tune(pairs[held], prob, y[held], truth, sub)
        out[c] = {"f05_at_tuned_threshold": round(f05_macro(pred, truth, sub), 5),
                  "f05_best_threshold": round(oracle[0], 5), "best_threshold": oracle[1]}
        log.info("unseen-country check, held out %s: F0.5 %.4f (best possible %.4f at %.2f)",
                 c, out[c]["f05_at_tuned_threshold"], oracle[0], oracle[1])
    return out


def train(data_dir: Path, cfg: Config, country_check: bool = True) -> Bundle:
    """Build training pairs, cross-validate, tune the decision rule, return a Bundle."""
    t0 = time.time()
    recs = data.load_split(data_dir, "train")
    truth = data.load_truth(data_dir / "train" / "train_ground_truth.tsv", recs)
    recs, pairs = pair_table(recs, cfg)
    ids = source1_ids(recs)
    country_of = dict(zip(recs["entity_id"], recs["country_key"]))
    y = labels(pairs, truth)

    n_true = sum(map(len, truth.values()))
    cross = sum(1 for s, ms in truth.items() for m in ms if country_of[m] != country_of[s])
    ceiling = f05_macro(pairs[y == 1].groupby("s1")["cand"].agg(frozenset).to_dict(), truth, ids)
    recall = float(y.sum() / n_true) if n_true else 1.0
    log.info("blocking: pair recall %.4f (%d of %d true pairs), F0.5 ceiling %.4f, "
             "%d true pairs cross countries", recall, int(y.sum()), n_true, ceiling, cross)

    feats = [c for c in pairs.columns if c not in ID_COLS]
    X = pairs[feats]
    oof, boosters = cross_validate(X, y, pairs["s1"].to_numpy(), cfg)
    score, threshold, exclusive = tune(pairs, oof, y, truth, ids)
    reference = f05_macro(decide(pairs, oof, threshold, exclusive), truth, ids)
    if abs(score - reference) > 1e-9:
        raise RuntimeError(f"tuned score {score} != reference metric {reference}")
    log.info("out-of-fold macro F0.5 %.4f at threshold %.2f (exclusive=%s)",
             score, threshold, exclusive)

    rounds = max(1, int(np.mean([b.best_iteration for b in boosters])))
    report = {
        "trained_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        "train_s1": len(ids), "train_pairs": len(pairs), "positives": int(y.sum()),
        "pair_recall": round(recall, 5), "f05_ceiling": round(ceiling, 5),
        "cross_country_true_pairs": cross,
        "oof_f05": round(score, 5),
        "oof_by_country": _per_country(pairs, oof, threshold, exclusive, truth, ids, country_of),
        "fold_rounds": [b.best_iteration for b in boosters],
        "feature_gain": dict(sorted(
            zip(feats, np.mean([b.feature_importance("gain") for b in boosters], axis=0).round(1).tolist()),
            key=lambda kv: -kv[1])),
        "provenance": provenance(data_dir, "train"),
    }
    if country_check:
        report["unseen_country"] = _country_transfer(pairs, X, y, truth, ids, country_of, cfg,
                                                     rounds, threshold, exclusive)
    report["train_seconds"] = round(time.time() - t0, 1)
    return Bundle(cfg, feats, threshold, exclusive, boosters, report)


def predict(data_dir: Path, bundle: Bundle) -> tuple[list[str], dict, dict, dict]:
    """Test-set matches and candidates, checked against the submission rules."""
    recs = data.load_split(data_dir, "test")
    recs, pairs = pair_table(recs, bundle.config)
    ids = source1_ids(recs)
    prob = bundle.predict(pairs) if len(pairs) else np.zeros(0)
    matches = decide(pairs, prob, bundle.threshold, bundle.exclusive)
    cands = pairs.groupby("s1")["cand"].agg(frozenset).to_dict()
    data.check_submission(ids, matches, cands, set(recs.loc[recs["src"] != 1, "entity_id"]))

    country_of = dict(zip(recs["entity_id"], recs["country_key"]))
    by_country = {}
    for c in sorted({country_of[i] for i in ids}):
        sub = [i for i in ids if country_of[i] == c]
        n = [len(matches.get(i, ())) for i in sub]
        by_country[c] = {"s1": len(sub), "with_matches": int(np.count_nonzero(n)),
                         "matched_ids": int(sum(n)),
                         "with_candidates": sum(1 for i in sub if cands.get(i))}
    report = {"predicted_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"), "test_s1": len(ids),
              "test_pairs": len(pairs), "by_country": by_country,
              "provenance": provenance(data_dir, "test")}
    for c, v in by_country.items():
        log.info("test %s: %d S1 records, %d with matches, %d matched ids",
                 c, v["s1"], v["with_matches"], v["matched_ids"])
    return ids, matches, cands, report
