"""Training and prediction: blocking -> pair features -> LightGBM -> decision rule.

Training produces a Bundle (fold models, feature list, decision rule, config,
metrics, data fingerprints). Prediction only reads the Bundle, so the test run
uses exactly the settings that were validated.
"""
import hashlib
import inspect
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
NON_FEATURES = {"i", "j", "y", "jsrc"}


# ---------------------------------------------------------------- stages

def peak_gb() -> float:
    """Peak resident memory of this process so far, in GB."""
    peak = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    return peak / (1 << 30) if sys.platform == "darwin" else peak / (1 << 20)


def _cache_key(inputs: dict, cfg: Config) -> str:
    """Identifies a candidate table: input file hashes, blocking settings and the
    source of every module that shapes records or candidates."""
    h = hashlib.sha256(json.dumps([inputs, cfg.k, cfg.k_rev, cfg.max_df], sort_keys=True).encode())
    for module in (candidates, sys.modules[features.text.__name__]):
        h.update(Path(module.__file__).read_bytes())
    for fn in (features._prepare_chunk, features.prepare, features.competition, features.add_group_features):
        h.update(inspect.getsource(fn).encode())
    return h.hexdigest()[:16]


def candidate_table(raw: pl.DataFrame, cfg: Config, inputs: dict,
                    cache: Path | None = None) -> tuple[pl.DataFrame, pl.DataFrame]:
    """Prepared records and their candidate pairs with retrieval-context features.

    With `cache`, results are stored under a key derived from the input hashes,
    the blocking settings and the code, and reused when all of them match.
    """
    where = cache / _cache_key(inputs, cfg) if cache else None
    if where and (where / "pairs.parquet").is_file():
        recs, pairs = pl.read_parquet(where / "recs.parquet"), pl.read_parquet(where / "pairs.parquet")
        log.info("loaded %d records and %d candidate pairs from cache %s", recs.height, pairs.height, where)
        return recs, pairs
    t0 = time.time()
    recs = features.prepare(raw)
    log.info("normalised %d records in %.1fs (peak %.1f GB)", recs.height, time.time() - t0, peak_gb())
    t0 = time.time()
    pairs = candidates.generate(recs, cfg.k, cfg.k_rev, cfg.max_df)
    pairs = features.add_group_features(pairs, recs["src"].to_numpy())
    log.info("%d candidate pairs (%.1f per Source 1 record) in %.1fs (peak %.1f GB)", pairs.height,
             pairs.height / max(1, (recs["src"] == 1).sum()), time.time() - t0, peak_gb())
    if where:
        where.mkdir(parents=True, exist_ok=True)
        recs.write_parquet(where / "recs.parquet")
        pairs.write_parquet(where / ".pairs.parquet")
        (where / ".pairs.parquet").rename(where / "pairs.parquet")  # marks the entry complete
        log.info("cached candidate table in %s", where)
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
class Stage:
    """One matcher: the fold models of a cross-validation and their feature names."""
    features: list[str]
    boosters: list[lgb.Booster]

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Mean probability of the fold models."""
        return np.mean([b.predict(X, num_iteration=b.best_iteration) for b in self.boosters], axis=0)


@dataclass
class Bundle:
    config: Config
    stages: list[Stage]
    rule: dict
    report: dict
    oof: pl.DataFrame | None = None  # out-of-fold predictions, saved for error analysis

    def save(self, directory: Path) -> None:
        directory.mkdir(parents=True, exist_ok=True)
        stages = []
        for s, stage in enumerate(self.stages, 1):
            files = []
            for n, b in enumerate(stage.boosters):
                name = f"stage{s}_fold{n}.txt"
                b.save_model(directory / name, num_iteration=b.best_iteration)
                files.append(name)
            stages.append({"features": stage.features, "models": files})
        meta = {"config": self.config.to_dict(), "stages": stages, "rule": self.rule, "report": self.report}
        (directory / BUNDLE_FILE).write_text(json.dumps(meta, indent=2))
        if self.oof is not None:
            self.oof.write_parquet(directory / "oof.parquet")
        log.info("saved bundle to %s", directory)

    @classmethod
    def load(cls, directory: Path) -> "Bundle":
        path = directory / BUNDLE_FILE
        if not path.is_file():
            raise data.DataError(f"no model bundle at {directory}; run `train` first")
        meta = json.loads(path.read_text())
        stages = []
        for st in meta["stages"]:
            boosters = [lgb.Booster(model_file=str(directory / f)) for f in st["models"]]
            if any(b.num_feature() != len(st["features"]) for b in boosters):
                raise data.DataError(f"{directory}: model features do not match the bundle")
            stages.append(Stage(st["features"], boosters))
        return cls(Config.from_dict(meta["config"]), stages, meta["rule"], meta["report"])


def score_pairs(recs: pl.DataFrame, pairs: pl.DataFrame, stage: Stage, size: int) -> np.ndarray:
    """Probabilities of one stage for every pair; features are built chunk by
    chunk and discarded, so memory stays flat whatever the number of pairs."""
    out = np.empty(pairs.height, dtype=np.float32)
    for s, f in zip(range(0, pairs.height, size), feature_chunks(recs, pairs, size)):
        out[s:s + f.height] = stage.predict(matrix(f, stage.features))
    return out


def score_all(recs: pl.DataFrame, pairs: pl.DataFrame, stages: list[Stage], size: int) -> np.ndarray:
    """Final probabilities: stage 1, then (if present) stage 2 on stage-1 context."""
    prob = score_pairs(recs, pairs, stages[0], size)
    if len(stages) > 1:
        ctx = features.probability_context(pairs.select("i", "j").with_columns(pl.Series("p1", prob)))
        prob = score_pairs(recs, pl.concat([pairs, ctx.drop("i", "j")], how="horizontal"), stages[1], size)
    return prob


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

def train(data_dir: Path, cfg: Config, country_check: bool = True, cache: Path | None = None) -> Bundle:
    """Candidates for the whole training split, a matcher cross-validated on a
    sample of Source 1 records, and the decision rule that maximises macro F0.5."""
    t_start = time.time()
    raw = data.load_split(data_dir, "train")
    truth = data.load_truth(data_dir / "train" / "train_ground_truth.tsv", raw)
    prov = provenance(data_dir, "train")
    recs, pairs = candidate_table(raw, cfg, prov["inputs"], cache)
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
    in_sample = pairs["i"].is_in(sample["row"].implode()).to_numpy()
    table = build_features(recs, pairs.filter(pl.Series(in_sample)), cfg.feature_chunk)
    names = [c for c in table.columns if c not in NON_FEATURES]
    X, y = matrix(table, names), table["y"].to_numpy()
    codes = code_of[table["i"].to_numpy()]
    n_true = sample["n_true"].to_numpy()
    log.info("training sample: %d Source 1 records, %d pairs, %d features, %d positives",
             sample.height, len(y), len(names), int(y.sum()))

    oof, boosters = cross_validate(X, y, codes, cfg)
    stages = [Stage(names, boosters)]
    stage1_score = choose_rule(codes, n_true, oof, y)[0]
    log.info("stage 1 out-of-fold macro F0.5 %.4f", stage1_score)
    country = sample["country_key"].to_numpy()
    if country_check:  # on stage-1 features, whose inputs never saw the held-out country
        transfer = _country_transfer(X, y, codes, n_true, country, cfg,
                                     max(1, int(np.mean([b.best_iteration for b in boosters]))))

    if cfg.stages > 1:
        # Stage-1 probabilities for every training pair: out-of-fold for the sample,
        # fold-model average for pairs of Source 1 records outside the sample.
        t0 = time.time()
        p1 = np.empty(pairs.height, dtype=np.float32)
        p1[in_sample] = oof
        p1[~in_sample] = score_pairs(recs, pairs.filter(pl.Series(~in_sample)), stages[0], cfg.feature_chunk)
        ctx = features.probability_context(pairs.select("i", "j").with_columns(pl.Series("p1", p1)))
        ctx = ctx.filter(pl.Series(in_sample)).drop("i", "j")
        log.info("stage-1 context for %d pairs in %.0fs (peak %.1f GB)", pairs.height, time.time() - t0, peak_gb())
        del X
        table = pl.concat([table, ctx], how="horizontal")
        names = names + ctx.columns
        X = matrix(table, names)
        oof, boosters = cross_validate(X, y, codes, cfg)
        stages.append(Stage(names, boosters))

    score, rule, trials = choose_rule(codes, n_true, oof, y)
    log.info("out-of-fold macro F0.5 %.4f with rule %s; best trials %s", score, rule, trials)
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
        "stage1_oof_f05": round(stage1_score, 5),
        "oof_f05": round(score, 5), "rule": rule, "rule_trials": trials, "oof_f05_by_country": by_country,
        "fold_rounds": [[b.best_iteration for b in st.boosters] for st in stages],
        "feature_gain": dict(sorted(zip(names, gain.round(1).tolist()), key=lambda kv: -kv[1])),
        "provenance": prov,
    }
    if country_check:
        report["unseen_country_stage1"] = transfer
    report["train_seconds"] = round(time.time() - t_start, 1)
    bundle = Bundle(cfg, stages, rule, report)
    bundle.oof = table.select("i", "j", "y").with_columns(
        pl.Series("oof", oof.astype(np.float32)), pl.Series("entity", codes))
    return bundle


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


def _country_transfer(X, y, codes, n_true, country, cfg, rounds) -> dict:
    """Leave-one-country-out on the training sample: train on the other countries,
    score the held-out one with the rule chosen on the training countries' own
    out-of-fold predictions. Estimates quality on a country never seen in
    training (the test set adds France)."""
    pair_country = country[codes]
    out = {}
    for c in np.unique(country):
        held = pair_country == c
        if held.all():
            continue
        # Rule chosen without the held-out country: out-of-fold on the others.
        tr_ent = np.flatnonzero(country != c)
        tr_remap = np.full(len(country), -1)
        tr_remap[tr_ent] = np.arange(len(tr_ent))
        tr_codes = tr_remap[codes[~held]]
        inner = cross_validate(X[~held], y[~held], tr_codes, cfg)[0]
        rule = choose_rule(tr_codes, n_true[tr_ent], inner, y[~held])[1]
        b = fit(X[~held], y[~held], cfg, rounds=rounds)
        prob = b.predict(X[held])
        ent = np.flatnonzero(country == c)
        remap = np.full(len(country), -1)
        remap[ent] = np.arange(len(ent))
        cc, nt = remap[codes[held]], n_true[ent]
        best = choose_rule(cc, nt, prob, y[held])
        out[str(c)] = {"f05_with_rule_from_other_country": round(macro_f05(cc, nt, apply_rule(rule, cc, prob), y[held]), 5),
                       "rule_from_other_country": rule,
                       "f05_best_rule": round(best[0], 5), "best_rule": best[1]}
        log.info("unseen-country check, held out %s: %s", c, out[str(c)])
    return out


def predict(data_dir: Path, bundle: Bundle, cache: Path | None = None
            ) -> tuple[pl.Series, pl.DataFrame, pl.DataFrame, dict]:
    """Matches and candidates for every test Source 1 record, checked against the
    submission rules. Returns (ids, match lists, candidate lists, report)."""
    raw = data.load_split(data_dir, "test")
    prov = provenance(data_dir, "test")
    recs, pairs = candidate_table(raw, bundle.config, prov["inputs"], cache)
    del raw
    t0 = time.time()
    prob = score_all(recs, pairs, bundle.stages, bundle.config.feature_chunk) if pairs.height else np.zeros(0)
    log.info("scored %d pairs with %d stage(s) in %.1fs (peak %.1f GB)", pairs.height, len(bundle.stages),
             time.time() - t0, peak_gb())
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
              "provenance": prov}
    return ids, match_lists, cand_lists, report
