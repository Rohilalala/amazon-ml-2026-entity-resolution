"""Training and prediction: blocking -> pair features -> LightGBM -> decision rule.

Training produces a Bundle (fold models, feature list, decision rule, config,
metrics, data fingerprints). Prediction only reads the Bundle, so the test run
uses exactly the settings that were validated.
"""
import hashlib
import json
import logging
import platform
import resource
import subprocess
import sys
import time
from dataclasses import dataclass
from importlib import metadata
from pathlib import Path

import lightgbm as lgb
import numpy as np
import polars as pl
from sklearn.model_selection import GroupKFold

from . import candidates, data, features
from .config import Config
from .metric import f05_counts

log = logging.getLogger(__name__)

THRESHOLDS = np.round(np.arange(0.05, 0.96, 0.01), 2)
MISSED_GRID = (0.0, 0.05, 0.1, 0.2, 0.3, 0.5, 0.8)
BUNDLE_FILE = "bundle.json"
PACKAGES = ("numpy", "polars", "scipy", "scikit-learn", "lightgbm", "rapidfuzz", "anyascii")
NON_FEATURES = {"i", "j", "y", "jsrc", "prob"}


# ---------------------------------------------------------------- stages

def peak_gb() -> float:
    """Peak resident memory of this process so far, in GB."""
    peak = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    return peak / (1 << 30) if sys.platform == "darwin" else peak / (1 << 20)


def candidate_table(raw: pl.DataFrame, cfg: Config) -> tuple[pl.DataFrame, pl.DataFrame]:
    """Prepared records and their candidate pairs with retrieval-context features."""
    t0 = time.time()
    recs = features.prepare(raw)
    log.info("normalised %d records in %.1fs (peak %.1f GB)", recs.height, time.time() - t0, peak_gb())
    t0 = time.time()
    pairs = candidates.generate(recs, cfg.k, cfg.k_rev, cfg.max_df)
    pairs = features.add_group_features(pairs, recs["src"].to_numpy())
    log.info("%d candidate pairs (%.1f per Source 1 record) in %.1fs (peak %.1f GB)", pairs.height,
             pairs.height / max(1, (recs["src"] == 1).sum()), time.time() - t0, peak_gb())
    return recs, pairs


def feature_chunks(recs: pl.DataFrame, pairs: pl.DataFrame, size: int):
    """Yield feature frames for consecutive chunks of `pairs`."""
    for s in range(0, pairs.height, size):
        yield features.pair_features(recs, pairs.slice(s, size))


def build_features(recs, pairs, size) -> pl.DataFrame:
    t0 = time.time()
    out = pl.concat(list(feature_chunks(recs, pairs, size))) if pairs.height else pairs
    log.info("features for %d pairs in %.1fs (peak %.1f GB)", pairs.height, time.time() - t0, peak_gb())
    return out


def matrix(frame: pl.DataFrame, names: list[str]) -> np.ndarray:
    return frame.select(names).to_numpy().astype(np.float32, copy=False)


def fit(X, y, cfg: Config, valid=None, rounds=None) -> lgb.Booster:
    """One LightGBM model: early-stopped on `valid`, or a fixed number of rounds."""
    train_set = lgb.Dataset(X, y, free_raw_data=True)
    if valid is None:
        return lgb.train(cfg.lgb(), train_set, rounds)
    return lgb.train(cfg.lgb(), train_set, cfg.num_boost_round,
                     valid_sets=[lgb.Dataset(valid[0], valid[1], reference=train_set)],
                     callbacks=[lgb.early_stopping(cfg.early_stopping_rounds, verbose=False)])


def cross_validate(X, y, groups, cfg: Config) -> tuple[np.ndarray, list[lgb.Booster]]:
    """Out-of-fold probabilities, folds grouped by Source 1 record so no record's
    pairs are split between training and validation."""
    oof, boosters = np.zeros(len(y), dtype=np.float64), []
    splitter = GroupKFold(n_splits=cfg.folds, shuffle=True, random_state=cfg.seed)
    for n, (tr, va) in enumerate(splitter.split(X, y, groups)):
        t0 = time.time()
        b = fit(X[tr], y[tr], cfg, valid=(X[va], y[va]))
        oof[va] = b.predict(X[va], num_iteration=b.best_iteration)
        boosters.append(b)
        log.info("fold %d/%d: %d rounds in %.0fs", n + 1, cfg.folds, b.best_iteration, time.time() - t0)
    return oof, boosters


def macro_f05(codes, n_true, keep, y) -> float:
    """Macro F0.5 over entities 0..len(n_true)-1 for the pairs selected by `keep`.
    `codes` maps each pair to its entity; `y` marks true pairs."""
    n = len(n_true)
    n_pred = np.bincount(codes, weights=keep, minlength=n)
    tp = np.bincount(codes, weights=keep & (y == 1), minlength=n)
    return float(f05_counts(n_pred, n_true, tp).mean())


def owner_mask(pairs: pl.DataFrame, prob: np.ndarray) -> np.ndarray:
    """True where the pair is the most likely owner of its Source 2/3 record."""
    top = pl.DataFrame({"j": pairs["j"], "p": prob}).select(pl.col("p").max().over("j"))["p"].to_numpy()
    return prob >= top


def expected_f05_mask(group: np.ndarray, prob: np.ndarray, missed: float) -> np.ndarray:
    """Per Source 1 record, keep the top-k candidates that maximise expected F0.5.

    With candidates sorted by probability p1 >= p2 >= ..., keeping the top k scores
    E[F] ~ 1.25 * (p1 + .. + pk) / (0.25 * (sum(p) + missed) + k), where `missed`
    is the expected number of true matches blocking did not retrieve. Keeping
    nothing scores 1 only for a singleton: E[F] ~ prod(1 - p) when `missed` is 0,
    shrunk by exp(-missed). The best k (possibly 0) is chosen per record.
    """
    df = pl.DataFrame({"g": group, "p": prob, "idx": np.arange(len(prob))}).sort(["g", "p"], descending=[False, True])
    df = df.with_columns(
        k=pl.int_range(1, pl.len() + 1).over("g"),
        cum=pl.col("p").cum_sum().over("g"),
        total=pl.col("p").sum().over("g"),
        none=((1 - pl.col("p")).clip(1e-12).log().sum().over("g").exp() * np.exp(-missed)),
    ).with_columns(ef=1.25 * pl.col("cum") / (0.25 * (pl.col("total") + missed) + pl.col("k")))
    df = df.with_columns(best=pl.col("ef").max().over("g"))
    best_k = pl.when(pl.col("best") > pl.col("none")).then(
        pl.col("k").filter(pl.col("ef") == pl.col("best")).first().over("g")).otherwise(0)
    keep_sorted = df.select(pl.col("k") <= best_k)["k"].to_numpy()
    keep = np.zeros(len(prob), dtype=bool)
    keep[df["idx"].to_numpy()] = keep_sorted
    return keep


def apply_rule(rule: dict, group: np.ndarray, prob: np.ndarray) -> np.ndarray:
    """Pairs kept by a decision rule: a global probability threshold, or the
    per-record expected-F0.5 selection."""
    if rule["kind"] == "threshold":
        return prob >= rule["threshold"]
    return expected_f05_mask(group, prob, rule["missed"])


def choose_rule(codes, n_true, prob, y) -> tuple[float, dict, list]:
    """Best decision rule on out-of-fold probabilities: every global threshold
    and several expected-F0.5 settings are scored with the exact metric."""
    trials = [(macro_f05(codes, n_true, prob >= t, y), {"kind": "threshold", "threshold": float(t)})
              for t in THRESHOLDS]
    trials += [(macro_f05(codes, n_true, expected_f05_mask(codes, prob, m), y),
                {"kind": "expected_f05", "missed": m}) for m in MISSED_GRID]
    best = max(trials, key=lambda t: t[0])
    top = sorted(trials, key=lambda t: -t[0])[:5]
    return best[0], best[1], [{"f05": round(f, 5), **r} for f, r in top]


# ---------------------------------------------------------------- bundle

@dataclass
class Bundle:
    config: Config
    features: list[str]
    rule: dict
    boosters: list[lgb.Booster]
    report: dict

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Mean probability of the fold models."""
        return np.mean([b.predict(X, num_iteration=b.best_iteration) for b in self.boosters], axis=0)

    def save(self, directory: Path) -> None:
        directory.mkdir(parents=True, exist_ok=True)
        files = []
        for n, b in enumerate(self.boosters):
            name = f"fold_{n}.txt"
            b.save_model(directory / name, num_iteration=b.best_iteration)
            files.append(name)
        meta = {"config": self.config.to_dict(), "features": self.features,
                "rule": self.rule, "models": files, "report": self.report}
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
            if b.num_feature() != len(meta["features"]):
                raise data.DataError(f"{directory}: model features do not match the bundle")
        return cls(Config.from_dict(meta["config"]), meta["features"], meta["rule"],
                   boosters, meta["report"])


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for block in iter(lambda: fh.read(1 << 22), b""):
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

def train(data_dir: Path, cfg: Config, country_check: bool = True) -> Bundle:
    """Candidates for the whole training split, a matcher cross-validated on a
    sample of Source 1 records, and the decision rule that maximises macro F0.5."""
    t_start = time.time()
    raw = data.load_split(data_dir, "train")
    truth = data.load_truth(data_dir / "train" / "train_ground_truth.tsv", raw)
    recs, pairs = candidate_table(raw, cfg)
    del raw
    pairs = pairs.join(truth.with_columns(pl.lit(1, dtype=pl.Int8).alias("y")), on=["i", "j"], how="left") \
        .with_columns(pl.col("y").fill_null(0))

    s1 = recs.filter(pl.col("src") == 1).select("row", "country_key")
    n_true_all = s1.join(truth.group_by("i").len("n_true"), left_on="row", right_on="i", how="left") \
        .with_columns(pl.col("n_true").fill_null(0))
    blocking = _blocking_report(pairs, n_true_all)

    # Sample Source 1 records for the matcher; their candidates come from the full pool.
    sample = n_true_all.sample(min(cfg.train_s1_sample, n_true_all.height), seed=cfg.seed).sort("row")
    code_of = np.full(recs.height, -1, dtype=np.int64)
    code_of[sample["row"].to_numpy()] = np.arange(sample.height)
    sp = pairs.filter(pl.col("i").is_in(sample["row"].implode()))
    table = build_features(recs, sp, cfg.feature_chunk)
    names = [c for c in table.columns if c not in NON_FEATURES]
    X, y = matrix(table, names), table["y"].to_numpy()
    codes = code_of[table["i"].to_numpy()]
    n_true = sample["n_true"].to_numpy()
    log.info("training sample: %d Source 1 records, %d pairs, %d features, %d positives",
             sample.height, len(y), len(names), int(y.sum()))

    oof, boosters = cross_validate(X, y, codes, cfg)
    score, rule, trials = choose_rule(codes, n_true, oof, y)
    log.info("out-of-fold macro F0.5 %.4f with rule %s; best trials %s", score, rule, trials)

    country = sample["country_key"].to_numpy()
    keep = apply_rule(rule, codes, oof)
    by_country = {}
    for c in np.unique(country):
        ent = np.flatnonzero(country == c)
        mask = np.isin(codes, ent)
        remap = np.full(sample.height, -1)
        remap[ent] = np.arange(len(ent))
        by_country[str(c)] = round(macro_f05(remap[codes[mask]], n_true[ent], keep[mask], y[mask]), 5)
    log.info("out-of-fold macro F0.5 by country: %s", by_country)

    gain = np.mean([b.feature_importance("gain") for b in boosters], axis=0)
    report = {
        "trained_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        "blocking": blocking,
        "sample_s1": sample.height, "sample_pairs": len(y), "sample_positives": int(y.sum()),
        "oof_f05": round(score, 5), "rule": rule, "rule_trials": trials, "oof_f05_by_country": by_country,
        "fold_rounds": [b.best_iteration for b in boosters],
        "feature_gain": dict(sorted(zip(names, gain.round(1).tolist()), key=lambda kv: -kv[1])),
        "provenance": provenance(data_dir, "train"),
    }
    if country_check:
        report["unseen_country"] = _country_transfer(X, y, codes, n_true, country, cfg,
                                                     max(1, int(np.mean(report["fold_rounds"]))), rule)
    report["train_seconds"] = round(time.time() - t_start, 1)
    return Bundle(cfg, names, rule, boosters, report)


def _blocking_report(pairs: pl.DataFrame, n_true: pl.DataFrame) -> dict:
    """Pair recall and the F0.5 a perfect matcher could reach on these candidates."""
    found = pairs.filter(pl.col("y") == 1).group_by("i").len("found")
    t = n_true.join(found, left_on="row", right_on="i", how="left").with_columns(pl.col("found").fill_null(0))
    ceiling = f05_counts(t["found"], t["n_true"], t["found"])
    out = {"pairs": pairs.height, "pairs_per_s1": round(pairs.height / t.height, 2),
           "pair_recall": round(t["found"].sum() / max(1, t["n_true"].sum()), 5),
           "f05_ceiling": round(float(ceiling.mean()), 5), "by_country": {}}
    for c in t["country_key"].unique().sort():
        m = (t["country_key"] == c).to_numpy()
        out["by_country"][c] = {"pair_recall": round(t["found"].filter(m).sum() / max(1, t["n_true"].filter(m).sum()), 5),
                                "f05_ceiling": round(float(ceiling[m].mean()), 5)}
    log.info("blocking: %s", out)
    return out


def _country_transfer(X, y, codes, n_true, country, cfg, rounds, rule) -> dict:
    """Leave-one-country-out on the training sample: train on the other countries,
    score the held-out one. Estimates quality on a country never seen in training
    (the test set adds France)."""
    pair_country = country[codes]
    out = {}
    for c in np.unique(country):
        held = pair_country == c
        if held.all():
            continue
        b = fit(X[~held], y[~held], cfg, rounds=rounds)
        prob = b.predict(X[held])
        ent = np.flatnonzero(country == c)
        remap = np.full(len(country), -1)
        remap[ent] = np.arange(len(ent))
        cc, nt = remap[codes[held]], n_true[ent]
        best = choose_rule(cc, nt, prob, y[held])
        out[str(c)] = {"f05_with_chosen_rule": round(macro_f05(cc, nt, apply_rule(rule, cc, prob), y[held]), 5),
                       "f05_best_rule": round(best[0], 5), "best_rule": best[1]}
        log.info("unseen-country check, held out %s: %s", c, out[str(c)])
    return out


def predict(data_dir: Path, bundle: Bundle) -> tuple[pl.Series, pl.DataFrame, pl.DataFrame, dict]:
    """Matches and candidates for every test Source 1 record, checked against the
    submission rules. Returns (ids, match lists, candidate lists, report)."""
    raw = data.load_split(data_dir, "test")
    recs, pairs = candidate_table(raw, bundle.config)
    del raw
    t0 = time.time()
    prob = np.concatenate([bundle.predict(matrix(f, bundle.features))
                           for f in feature_chunks(recs, pairs, bundle.config.feature_chunk)]) \
        if pairs.height else np.zeros(0)
    log.info("scored %d pairs in %.1fs", pairs.height, time.time() - t0)
    if bundle.config.exclusive:
        prob = np.where(owner_mask(pairs, prob), prob, 0.0)
    keep = apply_rule(bundle.rule, pairs["i"].to_numpy(), prob) & (prob > 0)
    matched = pairs.select("i", "j").filter(pl.Series(keep))

    id_of = recs["entity_id"].to_numpy()
    s1 = recs.filter(pl.col("src") == 1)
    ids = s1["entity_id"]
    match_lists = data.id_lists(ids, matched, id_of)
    cand_lists = data.id_lists(ids, pairs.select("i", "j"), id_of)
    data.check_submission(ids, match_lists, cand_lists)

    counts = s1.select("row", "country_key").join(
        matched.group_by("i").len("n"), left_on="row", right_on="i", how="left").with_columns(pl.col("n").fill_null(0))
    by_country = {c: {"s1": g.height, "with_matches": int((g["n"] > 0).sum()), "matched_ids": int(g["n"].sum())}
                  for (c,), g in counts.group_by(["country_key"], maintain_order=True)}
    for c, v in by_country.items():
        log.info("test %s: %s", c, v)
    report = {"predicted_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"), "test_s1": len(ids),
              "test_pairs": pairs.height, "matched_pairs": matched.height, "by_country": by_country,
              "provenance": provenance(data_dir, "test")}
    return ids, match_lists, cand_lists, report
