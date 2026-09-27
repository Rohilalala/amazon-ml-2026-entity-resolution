"""Training and prediction: blocking -> pair features -> LightGBM -> decision rule.

Training produces a Bundle (fold models, feature list, decision rule, config,
metrics, data fingerprints). Prediction only reads the Bundle, so the test run
uses exactly the settings that were validated.
"""
import gc
import hashlib
import inspect
import json
import logging
import platform
import resource
import subprocess
import sys
import tempfile
import time
from dataclasses import dataclass, field, replace
from importlib import metadata
from pathlib import Path

import lightgbm as lgb
import numpy as np
import polars as pl
from sklearn.model_selection import GroupKFold

from . import alignment, candidates, data, features, groupfeat, indic, locale_norm, pair_encoder, stage4, text, \
    unseen_rules
from .config import Config
from .metric import f05_counts

log = logging.getLogger(__name__)

THRESHOLDS = np.round(np.arange(0.05, 0.96, 0.01), 2)
MISSED_GRID = (0.0, 0.05, 0.1, 0.2, 0.3, 0.5, 0.8)
BUNDLE_FILE = "bundle.json"
NATIVE_FILE = "native_words.json"
VOCAB_FILE = "vocab_{}.parquet"
PACKAGES = ("numpy", "polars", "scipy", "scikit-learn", "lightgbm", "rapidfuzz", "anyascii", "torch", "transformers")
STAGE4_FILE = "stage4.txt"
NON_FEATURES = {"i", "j", "y", "jsrc"}
# Stage 3 rescores only pairs whose stage-2 probability reaches this value.
STAGE3_ACTIVE = 0.001
# Stage-3 group features that let a candidate's siblings vouch for a deviation they
# share with it (house number, name tokens). True copies share their corruption in
# training, but decoy businesses in the test set come as copy groups that share
# their difference too; the "robust" stage 3 is trained without these features.
def robust_stage3_features(names: list[str]) -> list[str]:
    return [n for n in names if "nov" not in n and "h_dev" not in n]


STAGE3_CHOICES = ("full", "robust", "none")
# Record columns saved with the test scores: ids, country and what unseen_rules reads.
RECORD_COLUMNS = ("row", "entity_id", "src", "country_key", *unseen_rules.COLUMNS)


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


def _locale_rules(raw: pl.DataFrame) -> dict:
    return locale_norm.learn(raw.select(
        pl.col("country").str.strip_chars().str.to_lowercase().alias("country_key"), "src",
        text.clean_address(raw["business_address"]).alias("addr")))


def candidate_table(raw: pl.DataFrame, cfg: Config, inputs: dict,
                    cache: Path | None = None) -> tuple[pl.DataFrame, pl.DataFrame]:
    """Prepared records and their candidate pairs with retrieval-context features.

    With `cache`, results are stored under a key derived from the input hashes,
    the blocking settings and the code, and reused when all of them match.
    """
    # Locale rules (label-free, learned from this split's addresses) are part of what
    # the candidate table depends on; they are empty for the US/India training data,
    # so they only enter the cache key when they change something.
    rules = _locale_rules(raw)
    if rules:
        log.info("locale rules: %s", rules)
        inputs = {**inputs, "locale_rules": rules}
    where = cache / _cache_key(inputs, cfg) if cache else None
    if where and (where / "pairs.parquet").is_file():
        recs, pairs = pl.read_parquet(where / "recs.parquet"), pl.read_parquet(where / "pairs.parquet")
        log.info("loaded %d records and %d candidate pairs from cache %s", recs.height, pairs.height, where)
        return recs, pairs
    t0 = time.time()
    recs = locale_norm.apply(features.prepare(raw), rules)
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


def feature_chunks(recs: pl.DataFrame, pairs: pl.DataFrame, size: int, tables: dict):
    """Yield feature frames for consecutive chunks of `pairs`."""
    for s in range(0, pairs.height, size):
        yield features.pair_features(recs, pairs.slice(s, size), tables)


def build_matrix(recs, pairs, size, tables) -> tuple[np.ndarray, list[str]]:
    """Feature matrix for `pairs` (row order kept), filled chunk by chunk so only one
    chunk's feature frame is alive at a time."""
    t0 = time.time()
    X, names = None, None
    for s, f in zip(range(0, pairs.height, size), feature_chunks(recs, pairs, size, tables)):
        if X is None:
            names = [c for c in f.columns if c not in NON_FEATURES]
            X = np.empty((pairs.height, len(names)), dtype=np.float32)
        X[s:s + f.height] = matrix(f, names)
    log.info("features for %d pairs in %.1fs (peak %.1f GB)", pairs.height, time.time() - t0, peak_gb())
    return X, names


def matrix(frame: pl.DataFrame, names: list[str]) -> np.ndarray:
    return frame.select(names).to_numpy().astype(np.float32, copy=False)


def fit(X, y, cfg: Config, valid=None, rounds=None, stage: int = 1) -> lgb.Booster:
    """One LightGBM model: early-stopped on `valid`, or a fixed number of rounds."""
    train_set = lgb.Dataset(X, y, free_raw_data=True)
    if valid is None:
        return lgb.train(cfg.lgb(stage), train_set, rounds)
    return lgb.train(cfg.lgb(stage), train_set, cfg.num_boost_round,
                     valid_sets=[lgb.Dataset(valid[0], valid[1], reference=train_set)],
                     callbacks=[lgb.early_stopping(cfg.early_stopping_rounds, verbose=False)])


def fold_ids(groups: np.ndarray, cfg: Config) -> np.ndarray:
    """Fold of every pair, grouped by Source 1 record so no record's pairs are split
    between training and validation. Computed once and shared by all stages."""
    folds = np.empty(len(groups), dtype=np.int8)
    splitter = GroupKFold(n_splits=cfg.folds, shuffle=True, random_state=cfg.seed)
    for n, (_, va) in enumerate(splitter.split(groups, groups=groups)):
        folds[va] = n
    return folds


def cross_validate(X, y, folds, cfg: Config, stage: int = 1) -> tuple[np.ndarray, list[lgb.Booster]]:
    """Out-of-fold probabilities over the given fold assignment. The features are
    binned once into one LightGBM dataset and every fold trains on a subset of it,
    so no fold copies the feature matrix."""
    oof, boosters = np.zeros(len(y), dtype=np.float64), []
    params = cfg.lgb(stage)
    full = lgb.Dataset(X, y, params=params, free_raw_data=False).construct()
    for n in range(cfg.folds):
        tr, va = np.flatnonzero(folds != n), np.flatnonzero(folds == n)
        t0 = time.time()
        b = lgb.train(params, full.subset(tr), cfg.num_boost_round, valid_sets=[full.subset(va)],
                      callbacks=[lgb.early_stopping(cfg.early_stopping_rounds, verbose=False)])
        for s in range(0, len(va), cfg.feature_chunk):
            rows = va[s:s + cfg.feature_chunk]
            oof[rows] = b.predict(X[rows], num_iteration=b.best_iteration)
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


def expected_f05_mask(group: np.ndarray, prob: np.ndarray, missed: float, none_scale: float = 1.0) -> np.ndarray:
    """Per Source 1 record, keep the top-k candidates that maximise expected F0.5.

    With candidates sorted by probability p1 >= p2 >= ..., keeping the top k scores
    E[F] ~ 1.25 * (p1 + .. + pk) / (0.25 * (sum(p) + missed) + k), where `missed`
    is the expected number of true matches blocking did not retrieve. Keeping
    nothing scores 1 only for a singleton: E[F] ~ prod(1 - p) when `missed` is 0,
    shrunk by exp(-missed). The best k (possibly 0) is chosen per record.
    `none_scale` > 1 favours "keep nothing" (used for countries unseen in training).
    """
    df = pl.DataFrame({"g": group, "p": prob, "idx": np.arange(len(prob))}).sort(["g", "p"], descending=[False, True])
    df = df.with_columns(
        k=pl.int_range(1, pl.len() + 1).over("g"),
        cum=pl.col("p").cum_sum().over("g"),
        total=pl.col("p").sum().over("g"),
        none=((1 - pl.col("p")).clip(1e-12).log().sum().over("g").exp() * np.exp(-missed) * none_scale),
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


def _empty_share(group: np.ndarray, keep: np.ndarray, n_s1: int) -> float:
    """Share of `n_s1` Source 1 records (including those without candidates) left empty."""
    return 1.0 - len(np.unique(group[keep])) / max(1, n_s1)


def decide(i: np.ndarray, j: np.ndarray, prob: np.ndarray, pair_country: np.ndarray, s1_country: np.ndarray,
           bundle: "Bundle", stage4_countries: list = ()) -> tuple[np.ndarray, dict]:
    """Final keep mask. Each Source 2/3 record goes to its most likely Source 1 record
    (exclusive owner rule), then the decision rule. For a country absent from training
    (France), the expected-F0.5 rule's "keep nothing" weight is raised until its share
    of empty Source 1 records matches the seen countries' share in the same run, or
    cfg.unseen_empty_target when set: an unseen country's probabilities are
    over-confident, and this uses no labels. Pairs of `stage4_countries` carry stage-4
    probabilities and are decided with the stage-4 rule."""
    cfg = bundle.config
    if cfg.exclusive:
        top = pl.DataFrame({"j": j, "p": prob}).select(pl.col("p").max().over("j"))["p"].to_numpy()
        prob = np.where(prob >= top, prob, 0.0)

    def rule_of(c):
        return bundle.stage4_rule if c in stage4_countries else bundle.rule
    if len(stage4_countries):
        m4 = np.isin(pair_country, list(stage4_countries))
        keep = np.zeros(len(prob), dtype=bool)
        for m, rule in ((m4, bundle.stage4_rule), (~m4, bundle.rule)):
            keep[m] = apply_rule(rule, i[m], prob[m]) & (prob[m] > 0)
    else:
        keep = apply_rule(bundle.rule, i, prob) & (prob > 0)
    info = {}
    seen = set(bundle.report.get("oof_f05_by_country", {}))
    countries, n_s1 = np.unique(s1_country, return_counts=True)
    unseen = [c for c in countries if c not in seen and rule_of(c)["kind"] == "expected_f05"]
    if not unseen or cfg.unseen_none_cap <= 1 or not seen:
        return keep, info
    share = {c: _empty_share(i[pair_country == c], keep[pair_country == c], n) for c, n in zip(countries, n_s1)}
    seen_share = min((share[c] for c in countries if c in seen), default=None)
    target = seen_share if cfg.unseen_empty_target is None else cfg.unseen_empty_target
    for c, n in zip(countries, n_s1):
        if c not in unseen:
            continue
        m = pair_country == c
        gi, pc, missed = i[m], prob[m], rule_of(c)["missed"]

        def mask(scale):
            return expected_f05_mask(gi, pc, missed, scale) & (pc > 0)
        lo, hi = 1.0, float(cfg.unseen_none_cap)
        if share[c] >= target:
            scale = 1.0
        elif _empty_share(gi, mask(hi), n) <= target:
            scale = hi
        else:
            for _ in range(25):  # bisection on the empty share, which grows with the scale
                mid = (lo + hi) / 2
                lo, hi = (mid, hi) if _empty_share(gi, mask(mid), n) < target else (lo, mid)
            scale = hi
        keep[m] = mask(scale)
        info[str(c)] = {"none_scale": round(scale, 4), "empty_share_before": round(share[c], 5),
                        "empty_share_after": round(_empty_share(gi, keep[m], n), 5), "target": round(target, 5),
                        "seen_share": None if seen_share is None else round(seen_share, 5)}
        log.info("unseen country %s: none_scale %.3f, empty share %.4f -> %.4f (target %.4f)",
                 c, scale, share[c], info[str(c)]["empty_share_after"], target)
    return keep, info


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
    native_words: dict | None = None  # Indic -> English dictionary learned in training
    variants: dict = field(default_factory=dict)  # alternative stage-3 models, e.g. {"robust": Stage}
    oof: pl.DataFrame | None = None  # out-of-fold predictions, saved for error analysis
    vocab: dict | None = None  # training-pool vocabulary tables (alignment.vocab_tables)
    stage4: Stage | None = None  # stage-4 stacker (its pair encoder lives in <bundle>/encoder)
    stage4_rule: dict | None = None

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
        variants = {}
        for name, stage in self.variants.items():
            files = []
            for n, b in enumerate(stage.boosters):
                fname = f"stage3_{name}_fold{n}.txt"
                b.save_model(directory / fname, num_iteration=b.best_iteration)
                files.append(fname)
            variants[name] = {"features": stage.features, "models": files}
        meta = {"config": self.config.to_dict(), "stages": stages, "variants": variants,
                "rule": self.rule, "report": self.report}
        if self.stage4:
            files = stage4_files(len(self.stage4.boosters))
            for b, f in zip(self.stage4.boosters, files):
                b.save_model(directory / f)
            meta["stage4"] = {"features": self.stage4.features, "models": files, "rule": self.stage4_rule}
        (directory / BUNDLE_FILE).write_text(json.dumps(meta, indent=2))
        (directory / NATIVE_FILE).write_text(json.dumps(self.native_words or {}, ensure_ascii=False,
                                                        sort_keys=True), encoding="utf-8")
        if self.oof is not None:
            self.oof.write_parquet(directory / "oof.parquet")
        save_vocab(directory, self.vocab or {})
        log.info("saved bundle to %s", directory)

    @classmethod
    def load(cls, directory: Path) -> "Bundle":
        path = directory / BUNDLE_FILE
        if not path.is_file():
            raise data.DataError(f"no model bundle at {directory}; run `train` first")
        meta = json.loads(path.read_text())
        def stage(st):
            boosters = [lgb.Booster(model_file=str(directory / f)) for f in st["models"]]
            if any(b.num_feature() != len(st["features"]) for b in boosters):
                raise data.DataError(f"{directory}: model features do not match the bundle")
            return Stage(st["features"], boosters)
        native = directory / NATIVE_FILE
        words = json.loads(native.read_text(encoding="utf-8")) if native.is_file() else {}
        vocab = {k: pl.read_parquet(directory / VOCAB_FILE.format(k)) for k in alignment.VOCAB
                 if (directory / VOCAB_FILE.format(k)).is_file()}
        st4 = meta.get("stage4")
        return cls(Config.from_dict(meta["config"]), [stage(st) for st in meta["stages"]], meta["rule"],
                   meta["report"], words, {k: stage(v) for k, v in meta.get("variants", {}).items()},
                   vocab=vocab or None, stage4=stage(st4) if st4 else None, stage4_rule=st4 and st4["rule"])


def stage4_files(n: int) -> list[str]:
    """Stacker file names: stage4.txt for one model, stage4_m{k}.txt for seed members."""
    return [STAGE4_FILE] if n == 1 else [f"stage4_m{k}.txt" for k in range(n)]


def save_vocab(directory: Path, vocab: dict) -> None:
    for k, v in vocab.items():
        v.write_parquet(directory / VOCAB_FILE.format(k))


def train_vocab(data_dir: Path, bundle: Bundle) -> dict:
    """Training-pool vocabulary tables for a bundle trained before they were saved with
    it: the training records prepared exactly as in training (same Indic dictionary)."""
    raw = data.load_split(data_dir, "train")
    text.set_native_words(bundle.native_words or {})
    recs = locale_norm.apply(features.prepare(raw), _locale_rules(raw))
    return alignment.vocab_tables(recs)


def apply_feature_caps(x: np.ndarray, names: list[str], caps: dict, mask: np.ndarray | None) -> None:
    """Clip the named feature columns of `x` in place, on `mask` rows (all when None).

    Test pools can carry a pattern at a density training never saw: decoy copy
    groups make amb_s23_addr_2 read like copy-consensus. A predict-time cap
    restores the training-time meaning of the feature for the chosen countries.
    """
    for name, cap in caps.items():
        if name in names:
            col = names.index(name)
            if mask is None:
                np.minimum(x[:, col], cap, out=x[:, col])
            else:
                x[mask, col] = np.minimum(x[mask, col], cap)


def score_pairs(recs: pl.DataFrame, pairs: pl.DataFrame, stage: Stage, size: int, tables: dict,
                store: np.ndarray | None = None, caps: dict | None = None,
                cap_mask: np.ndarray | None = None) -> np.ndarray:
    """Probabilities of one stage for every pair; features are built chunk by
    chunk, so memory stays flat whatever the number of pairs. With `store`, each
    chunk's feature rows are also written there for reuse by the next stage.
    `caps` clips feature columns (on `cap_mask` rows) before prediction and
    before storing, so later stages see the capped values too."""
    out = np.empty(pairs.height, dtype=np.float32)
    for s, f in zip(range(0, pairs.height, size), feature_chunks(recs, pairs, size, tables)):
        x = matrix(f, stage.features)
        if caps:
            apply_feature_caps(x, stage.features, caps,
                               None if cap_mask is None else cap_mask[s:s + len(x)])
        if store is not None:
            store[s:s + len(x)] = x
        out[s:s + len(x)] = stage.predict(x)
    return out


def _stage3_frame(ij: pl.DataFrame, prob: np.ndarray, recs: pl.DataFrame) -> pl.DataFrame:
    """Stage-3 inputs: the stage-2 probability ('oof') first, then the group features."""
    f3 = groupfeat.stage3_features(ij.with_columns(pl.Series("oof", prob)), recs, "oof")
    return f3.select(pl.Series("oof", prob.astype(np.float32)), pl.exclude("i", "j", "oof"))


def stage3_scores(recs: pl.DataFrame, pairs: pl.DataFrame, prob: np.ndarray, stage: Stage) -> np.ndarray:
    """Stage-3 probabilities: pairs at or above STAGE3_ACTIVE are rescored from
    group features; the rest keep their stage-2 probability."""
    act = prob >= STAGE3_ACTIVE
    f3 = _stage3_frame(pairs.select("i", "j").filter(pl.Series(act)), prob[act], recs)
    out = prob.copy()
    out[act] = stage.predict(matrix(f3, stage.features))
    return out


def cap_mask_for(recs: pl.DataFrame, pairs: pl.DataFrame, countries: list) -> np.ndarray | None:
    """Rows of `pairs` whose Source 1 record is in one of `countries` (None = all rows)."""
    if not countries:
        return None
    country = recs["country_key"].to_numpy()[pairs["i"].to_numpy()]
    return np.isin(country, list(countries))


def score_all(recs: pl.DataFrame, pairs: pl.DataFrame, stages: list[Stage], size: int, tables: dict,
              variants: dict | None = None, caps: dict | None = None,
              cap_countries: list | None = None) -> dict:
    """Probabilities through every stage: {"stage2": ..., "full": ..., <variant>: ...}
    ("full" is the last stage). Stage-1 features are kept in a disk-backed array so
    stage 2 appends its context instead of recomputing them. `caps` clips feature
    columns at predict time for the `cap_countries` (all countries when empty)."""
    n, first = pairs.height, stages[0]
    mask = cap_mask_for(recs, pairs, cap_countries or []) if caps else None
    with tempfile.TemporaryDirectory(prefix="ber-features-") as tmp:
        store = np.lib.format.open_memmap(Path(tmp) / "stage1.npy", mode="w+", dtype=np.float32,
                                          shape=(n, len(first.features))) if len(stages) > 1 else None
        prob = score_pairs(recs, pairs, first, size, tables, store, caps, mask)
        if len(stages) > 1:
            second = stages[1]
            if second.features[:len(first.features)] != first.features:
                raise data.DataError("stage-2 features must extend the stage-1 features")
            ctx = features.probability_context(pairs.select("i", "j").with_columns(pl.Series("p1", prob)))
            extra = ctx.select(second.features[len(first.features):]).to_numpy().astype(np.float32)
            for s in range(0, n, size):
                prob[s:s + size] = second.predict(np.hstack([store[s:s + size], extra[s:s + size]]))
            del store
    out = {"stage2": prob, "full": prob}
    if len(stages) > 2:
        out["full"] = stage3_scores(recs, pairs, prob, stages[2])
        for name, stage in (variants or {}).items():
            out[name] = stage3_scores(recs, pairs, prob, stage)
    return out


def _dict_hash(d: dict) -> str:
    return hashlib.sha256(json.dumps(d, ensure_ascii=False, sort_keys=True).encode()).hexdigest()[:16]


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

def evaluate_blocking(data_dir: Path, cfg: Config, cache: Path | None = None) -> dict:
    """Candidate generation on the training split only: pair recall and F0.5
    ceiling per country, without training a matcher."""
    raw = data.load_split(data_dir, "train")
    truth = data.load_truth(data_dir / "train" / "train_ground_truth.tsv", raw)
    prov = provenance(data_dir, "train")
    native = indic.learn_dictionary(raw, truth)
    text.set_native_words(native)
    recs, pairs = candidate_table(raw, cfg, {**prov["inputs"], "native_words": _dict_hash(native)}, cache)
    pairs = pairs.select("i", "j").join(truth.with_columns(pl.lit(1, dtype=pl.Int8).alias("y")),
                                        on=["i", "j"], how="left").with_columns(pl.col("y").fill_null(0))
    s1 = recs.filter(pl.col("src") == 1).select("row", "country_key")
    n_true = s1.join(truth.group_by("i").len("n_true"), left_on="row", right_on="i", how="left") \
        .with_columns(pl.col("n_true").fill_null(0))
    return _blocking_report(pairs, n_true)


def sample_and_holdout(n_true_all: pl.DataFrame, cfg: Config) -> tuple[pl.DataFrame, pl.DataFrame]:
    """Source 1 records the matcher is trained on (their candidates come from the full pool),
    and the holdout: records used by neither stage, scored exactly as the test set is."""
    shuffled = n_true_all.sample(fraction=1.0, shuffle=True, seed=cfg.seed)
    sample = shuffled.head(min(cfg.train_s1_sample, n_true_all.height)).sort("row")
    return sample, shuffled.slice(sample.height, cfg.holdout_s1).sort("row")


def train(data_dir: Path, cfg: Config, country_check: bool = True, cache: Path | None = None) -> Bundle:
    """Candidates for the whole training split, a matcher cross-validated on a
    sample of Source 1 records, and the decision rule that maximises macro F0.5."""
    t_start = time.time()
    raw = data.load_split(data_dir, "train")
    truth = data.load_truth(data_dir / "train" / "train_ground_truth.tsv", raw)
    prov = provenance(data_dir, "train")
    native = indic.learn_dictionary(raw, truth)
    text.set_native_words(native)
    log.info("learned %d Indic -> English dictionary entries", len(native))
    recs, pairs = candidate_table(raw, cfg, {**prov["inputs"], "native_words": _dict_hash(native)}, cache)
    del raw
    pairs = pairs.join(truth.with_columns(pl.lit(1, dtype=pl.Int8).alias("y")), on=["i", "j"], how="left") \
        .with_columns(pl.col("y").fill_null(0))

    s1 = recs.filter(pl.col("src") == 1).select("row", "country_key")
    n_true_all = s1.join(truth.group_by("i").len("n_true"), left_on="row", right_on="i", how="left") \
        .with_columns(pl.col("n_true").fill_null(0))
    blocking = _blocking_report(pairs, n_true_all)

    sample, holdout = sample_and_holdout(n_true_all, cfg)
    code_of = np.full(recs.height, -1, dtype=np.int64)
    code_of[sample["row"].to_numpy()] = np.arange(sample.height)
    in_sample = pairs["i"].is_in(sample["row"].implode()).to_numpy()
    in_hold = pairs["i"].is_in(holdout["row"].implode()).to_numpy()
    hp = pairs.filter(pl.Series(in_hold))  # holdout pairs, kept for the end of training
    tables = alignment.pool_tables(recs)
    ids = pairs.filter(pl.Series(in_sample)).select("i", "j", "y")
    X, names = build_matrix(recs, pairs.filter(pl.Series(in_sample)), cfg.feature_chunk, tables)
    y = ids["y"].to_numpy()
    codes = code_of[ids["i"].to_numpy()]
    folds = fold_ids(codes, cfg)
    n_true = sample["n_true"].to_numpy()
    log.info("training sample: %d Source 1 records, %d pairs, %d features, %d positives",
             sample.height, len(y), len(names), int(y.sum()))

    oof, boosters = cross_validate(X, y, folds, cfg)
    stages = [Stage(names, boosters)]
    stage_scores = [choose_rule(codes, n_true, oof, y)[0]]
    log.info("stage 1 out-of-fold macro F0.5 %.4f", stage_scores[-1])
    country = sample["country_key"].to_numpy()
    if country_check:  # on stage-1 features, whose inputs never saw the held-out country
        transfer = _country_transfer(X, y, codes, n_true, country, cfg,
                                     max(1, int(np.mean([b.best_iteration for b in boosters]))))

    variants = {}
    oof_stage2 = oof_robust = None
    if cfg.stages > 1:
        # Stage-1 probabilities for every training pair: out-of-fold for the sample,
        # fold-model average for pairs of Source 1 records outside the sample.
        t0 = time.time()
        p1 = np.empty(pairs.height, dtype=np.float32)
        p1[in_sample] = oof
        p1[~in_sample] = score_pairs(recs, pairs.filter(pl.Series(~in_sample)), stages[0], cfg.feature_chunk, tables)
        full_ctx = features.probability_context(pairs.select("i", "j").with_columns(pl.Series("p1", p1)))
        ctx = full_ctx.filter(pl.Series(in_sample)).drop("i", "j")
        hold_ctx = full_ctx.filter(pl.Series(in_hold)).drop("i", "j")
        log.info("stage-1 context for %d pairs in %.0fs (peak %.1f GB)", pairs.height, time.time() - t0, peak_gb())
        # Only the sample and holdout rows are needed from here on.
        del full_ctx, p1, pairs
        gc.collect()
        X = np.hstack([X, ctx.to_numpy().astype(np.float32)])
        names = names + ctx.columns
        del ctx
        oof, boosters = cross_validate(X, y, folds, cfg, stage=2)
        stages.append(Stage(names, boosters))
        stage_scores.append(choose_rule(codes, n_true, oof, y)[0])
        oof_stage2 = oof
        log.info("stage 2 out-of-fold macro F0.5 %.4f", stage_scores[-1])
    del X

    if cfg.stages > 2:
        # Group features over the stage-2 out-of-fold probabilities, same folds.
        act = oof >= STAGE3_ACTIVE
        f3 = _stage3_frame(ids.select("i", "j").filter(pl.Series(act)), oof[act], recs)
        names3 = f3.columns
        oof3, boosters3 = cross_validate(matrix(f3, names3), y[act], folds[act], cfg, stage=3)
        oof = oof.copy()
        oof[act] = oof3
        stages.append(Stage(names3, boosters3))
        stage_scores.append(choose_rule(codes, n_true, oof, y)[0])
        log.info("stage 3 out-of-fold macro F0.5 %.4f (%d active pairs)", stage_scores[-1], int(act.sum()))
        names3r = robust_stage3_features(names3)
        oof3r, boosters3r = cross_validate(matrix(f3, names3r), y[act], folds[act], cfg, stage=3)
        variants["robust"] = Stage(names3r, boosters3r)
        oof_robust = oof.copy()
        oof_robust[act] = oof3r
        log.info("stage 3 robust (without %d shared-deviation features) out-of-fold macro F0.5 %.4f",
                 len(names3) - len(names3r), choose_rule(codes, n_true, oof_robust, y)[0])

    score, rule, trials = choose_rule(codes, n_true, oof, y)
    log.info("out-of-fold macro F0.5 %.4f with rule %s; best trials %s", score, rule, trials)
    holdout_report = None
    if holdout.height:
        if len(stages) > 1:
            h_prob = score_pairs(recs, pl.concat([hp, hold_ctx], how="horizontal"), stages[1],
                                 cfg.feature_chunk, tables)
        else:
            h_prob = score_pairs(recs, hp, stages[0], cfg.feature_chunk, tables)
        h_options = {"none": h_prob}
        if len(stages) > 2:
            h_options["full"] = stage3_scores(recs, hp, h_prob, stages[2])
            for name, st in variants.items():
                h_options[name] = stage3_scores(recs, hp, h_prob, st)
        holdout_report = {k: _holdout_report(hp, v, holdout, rule) for k, v in h_options.items()}
        log.info("HOLDOUT (never trained on, scored like the test set) by stage-3 choice: %s", holdout_report)
    keep = apply_rule(rule, codes, oof)
    by_country = {}
    for c in np.unique(country):
        ent = np.flatnonzero(country == c)
        mask = np.isin(codes, ent)
        remap = np.full(sample.height, -1)
        remap[ent] = np.arange(len(ent))
        by_country[str(c)] = round(macro_f05(remap[codes[mask]], n_true[ent], keep[mask], y[mask]), 5)
    log.info("out-of-fold macro F0.5 by country: %s", by_country)

    report = {
        "trained_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        "blocking": blocking,
        "sample_s1": sample.height, "sample_pairs": len(y), "sample_positives": int(y.sum()),
        "oof_f05_by_stage": [round(v, 5) for v in stage_scores],
        "oof_f05": round(score, 5), "rule": rule, "rule_trials": trials, "oof_f05_by_country": by_country,
        "fold_rounds": [[b.best_iteration for b in st.boosters] for st in stages],
        "feature_gain": [dict(sorted(zip(st.features, np.mean([b.feature_importance("gain") for b in st.boosters],
                                                               axis=0).round(1).tolist()), key=lambda kv: -kv[1]))
                         for st in stages],
        "provenance": prov,
    }
    if country_check:
        report["unseen_country_stage1"] = transfer
    report["holdout"] = holdout_report
    report["train_seconds"] = round(time.time() - t_start, 1)
    if oof_robust is not None:
        report["oof_f05_stage3_robust"] = round(choose_rule(codes, n_true, oof_robust, y)[0], 5)
    bundle = Bundle(cfg, stages, rule, report, native, variants, vocab={k: tables[k] for k in alignment.VOCAB})
    extra = [pl.Series(n, v.astype(np.float32)) for n, v in (("oof_stage2", oof_stage2), ("oof_robust", oof_robust))
             if v is not None]
    bundle.oof = ids.with_columns(pl.Series("oof", oof.astype(np.float32)), pl.Series("entity", codes), *extra)
    return bundle


def _holdout_report(pairs: pl.DataFrame, prob: np.ndarray, holdout: pl.DataFrame, rule: dict) -> dict:
    """Macro F0.5 on holdout Source 1 records (all of them, including those with
    no candidates), overall and per country."""
    code_of = pl.DataFrame({"i": holdout["row"], "code": np.arange(holdout.height)})
    codes = pairs.select("i").join(code_of, on="i", how="left", maintain_order="left")["code"].to_numpy()
    keep = apply_rule(rule, codes, prob)
    y = pairs["y"].to_numpy()
    n_true = holdout["n_true"].to_numpy()
    out = {"s1": holdout.height, "f05": round(macro_f05(codes, n_true, keep, y), 5), "by_country": {}}
    country = holdout["country_key"].to_numpy()
    for c in np.unique(country):
        ent = np.flatnonzero(country == c)
        remap = np.full(holdout.height, -1)
        remap[ent] = np.arange(len(ent))
        m = np.isin(codes, ent)
        out["by_country"][str(c)] = round(macro_f05(remap[codes[m]], n_true[ent], keep[m], y[m]), 5)
    return out


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
        inner = cross_validate(X[~held], y[~held], fold_ids(tr_codes, cfg), cfg)[0]
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


def train_stage4(data_dir: Path, bundle_dir: Path, base: Path | None, cache: Path | None = None) -> dict:
    """Stage 4 for a trained bundle: fine-tune the pair encoder from the `base` checkpoint (or,
    with base None, keep the bundle's encoder/, e.g. an adapted one), then cross-validate the
    stacker on the matcher's training sample and save both.

    Encoder data: training Source 1 records outside the matcher's sample and holdout
    (stage4.encoder_pairs), minus pairs whose candidate belongs to an active sample pair.
    Stacker rows: the active sample pairs, with the out-of-fold probability each is decided
    on (oof.parquet, stage-3 choice per country as configured) and the encoder's logit;
    feature set cfg.stage4_features; folds grouped by Source 1 record as in `train`; one
    cross-validation per seed of cfg.stage4_seeds (fold assignment fixed by cfg.seed), the
    out-of-fold probabilities averaged; the decision rule is chosen on that average, and each
    seed's final model is refit on all active pairs with that seed's mean best iteration."""
    bundle = Bundle.load(bundle_dir)
    cfg, params = bundle.config, bundle.config.ce_params
    names = stage4.FEATURE_SETS[cfg.stage4_features]
    if not (bundle_dir / "oof.parquet").is_file():
        raise data.DataError(f"{bundle_dir}/oof.parquet is needed to train stage 4")
    enc_dir = bundle_dir / pair_encoder.ENCODER_DIR
    if base is None and not (enc_dir / "model.safetensors").is_file():
        raise data.DataError(f"no encoder in {enc_dir}; pass --base to fine-tune one")
    raw = data.load_split(data_dir, "train")
    truth = data.load_truth(data_dir / "train" / "train_ground_truth.tsv", raw)
    prov = provenance(data_dir, "train")
    native = bundle.native_words or {}
    text.set_native_words(native)
    pair_text = pair_encoder.texts(raw)
    recs, pairs = candidate_table(raw, cfg, {**prov["inputs"], "native_words": _dict_hash(native)}, cache)
    del raw
    s1 = recs.filter(pl.col("src") == 1).select("row", "country_key")
    n_true_all = s1.join(truth.group_by("i").len("n_true"), left_on="row", right_on="i", how="left") \
        .with_columns(pl.col("n_true").fill_null(0))
    sample, holdout = sample_and_holdout(n_true_all, cfg)
    oof = pl.read_parquet(bundle_dir / "oof.parquet")
    if not oof["i"].is_in(sample["row"].implode()).all():
        raise data.DataError("oof.parquet does not match this training split and configuration")
    country = recs["country_key"].to_numpy()[oof["i"].to_numpy()]
    oof_probs = {"stage2": oof["oof_stage2"].to_numpy(), "full": oof["oof"].to_numpy()}
    if "oof_robust" in oof.columns:
        oof_probs["robust"] = oof["oof_robust"].to_numpy()
    prob = select_probability(oof_probs, bundle, country)
    act = prob >= cfg.stage4_active
    A = oof.filter(pl.Series(act)).select("i", "j", "y", "entity")
    enc_report = {"encoder": "kept from the bundle"}
    if base is not None:
        used = pl.concat([sample["row"], holdout["row"]])
        fit_pairs, val_pairs = stage4.encoder_pairs(pairs, truth, n_true_all.filter(~pl.col("row").is_in(used.implode())),
                                                    params, A["j"].unique())
        enc_report = pair_encoder.train(base, enc_dir, pair_text, fit_pairs, val_pairs, params)
    string_recs = recs.select(stage4.RECORD_COLUMNS) if cfg.stage4_features != "A" else None
    del recs, pairs
    gc.collect()
    ce = pair_encoder.score(enc_dir, pair_text, A["i"].to_numpy(), A["j"].to_numpy(), params)
    F = A.with_columns(prob=pl.Series(prob[act]), ce_logit=pl.Series(ce))
    empty = pair_encoder.empty_address(pair_text)
    F = stage4.features(F, empty) if string_recs is None else stage4.features_acnrst(F, empty, string_recs)
    x, y, codes = stage4.matrix(F, names), A["y"].to_numpy(), oof["entity"].to_numpy()
    del F, string_recs
    cfg4 = replace(cfg, folds=stage4.FOLDS)
    folds = fold_ids(A["entity"].to_numpy(), cfg4)
    oofs, fold_rounds = [], []
    for sd in cfg.stage4_seeds:
        o, boosters = cross_validate(x, y, folds, replace(cfg4, seed=sd))
        oofs.append(o)
        fold_rounds.append([b.best_iteration for b in boosters])
    p4 = prob.astype(np.float64)
    p4[act] = np.mean(oofs, axis=0)
    n_true, y_all = sample["n_true"].to_numpy(), oof["y"].to_numpy()
    f4, rule, trials = choose_rule(codes, n_true, p4, y_all)
    rounds = [max(1, int(np.mean(r))) for r in fold_rounds]
    members = [fit(x, y, replace(cfg, seed=sd), rounds=n) for sd, n in zip(cfg.stage4_seeds, rounds)]
    bundle.stage4, bundle.stage4_rule = Stage(names, members), rule
    report = {"encoder": enc_report, "features": cfg.stage4_features, "seeds": cfg.stage4_seeds,
              "active_pairs": int(act.sum()), "fold_rounds": fold_rounds, "final_rounds": rounds,
              "oof_f05": round(f4, 5), "rule": rule, "rule_trials": trials,
              "oof_f05_without_stage4": round(choose_rule(codes, n_true, prob, y_all)[0], 5)}
    log.info("stage 4: %s", report)
    files = stage4_files(len(members))
    for b, f in zip(members, files):
        b.save_model(bundle_dir / f)
    set_stage4_meta(bundle_dir, bundle.stage4_rule, {**report, "trained_at": time.strftime("%Y-%m-%dT%H:%M:%S%z")},
                    names, files)
    return report


def train_encoder(data_dir: Path, bundle_dir: Path, init: Path, out_dir: Path, lr: float, draw: int,
                  exclude: list, seed: int, cache: Path | None = None) -> dict:
    """Continued fine-tuning of a pair encoder (`init`) on fresh training Source 1 records: those
    outside the matcher's sample and holdout and outside earlier draws (`exclude`: their
    s1_draw.parquet files); `draw` of them at random (0 = all), 10,000 of those kept for the
    validation AUC. Each record brings its top 8 candidates by retrieval score plus every true pair
    among its candidates; fit pairs are those of rank <= 6 or true, minus pairs whose candidate
    belongs to an active pair of the matcher's sample. One pass (pair_encoder.train) at `lr`.
    The records drawn are saved to <out_dir>/s1_draw.parquet for the next round's `exclude`, the
    fit pairs to fit_pairs.parquet (the anchor pool of test-time adaptation)."""
    bundle = Bundle.load(bundle_dir)
    cfg, params = bundle.config, bundle.config.ce_params
    raw = data.load_split(data_dir, "train")
    truth = data.load_truth(data_dir / "train" / "train_ground_truth.tsv", raw)
    prov = provenance(data_dir, "train")
    text.set_native_words(bundle.native_words or {})
    pair_text = pair_encoder.texts(raw)
    recs, pairs = candidate_table(raw, cfg, {**prov["inputs"], "native_words": _dict_hash(bundle.native_words or {})}, cache)
    del raw
    s1 = recs.filter(pl.col("src") == 1).select("row", "country_key")
    n_true_all = s1.join(truth.group_by("i").len("n_true"), left_on="row", right_on="i", how="left") \
        .with_columns(pl.col("n_true").fill_null(0))
    sample, holdout = sample_and_holdout(n_true_all, cfg)
    oof = pl.read_parquet(bundle_dir / "oof.parquet", columns=["i", "j", "oof_stage2", "oof"] +
                          (["oof_robust"] if "oof_robust" in pl.read_parquet_schema(bundle_dir / "oof.parquet") else []))
    probs = {"stage2": oof["oof_stage2"].to_numpy(), "full": oof["oof"].to_numpy()}
    if "oof_robust" in oof.columns:
        probs["robust"] = oof["oof_robust"].to_numpy()
    prob = select_probability(probs, bundle, recs["country_key"].to_numpy()[oof["i"].to_numpy()])
    exclude_j = oof.filter(pl.Series(prob >= cfg.stage4_active))["j"].unique()
    used = pl.concat([sample["row"], holdout["row"], *[pl.read_parquet(Path(f) / "s1_draw.parquet")["i"] for f in exclude]])
    pool = n_true_all.filter(~pl.col("row").is_in(used.implode())).sort("row")
    fresh = (pool.sample(draw, seed=seed) if draw else pool).sort("row")
    val = fresh.sample(min(10_000, fresh.height // 10), seed=seed + 1)["row"]
    P = (pairs.select("i", "j", "score").filter(pl.col("i").is_in(fresh["row"].implode()))
         .join(truth.with_columns(pl.lit(1, dtype=pl.Int8).alias("y")), on=["i", "j"], how="left")
         .with_columns(pl.col("y").fill_null(0), pl.col("score").rank("ordinal", descending=True).over("i").alias("rank"))
         .filter((pl.col("rank") <= 8) | (pl.col("y") == 1)).sort("i", "rank"))
    is_val, fits = pl.col("i").is_in(val.implode()), (pl.col("rank") <= 6) | (pl.col("y") == 1)
    fit = P.filter(~is_val & fits & ~pl.col("j").is_in(exclude_j.implode())).select("i", "j", "y")
    report = pair_encoder.train(init, out_dir, pair_text, fit, P.filter(is_val & fits).select("i", "j", "y"),
                                {**params, "lr": lr})
    fresh.select(pl.col("row").alias("i"), is_val=pl.col("row").is_in(val.implode())).write_parquet(out_dir / "s1_draw.parquet")
    # the anchor pool of test-time adaptation (adapt-labels --anchor-pool)
    fit.join(fresh.select(pl.col("row").alias("i"), pl.col("country_key").alias("country")), on="i", how="left",
             maintain_order="left").write_parquet(out_dir / "fit_pairs.parquet")
    report.update(init=str(init.name), lr=lr, fresh_s1=fresh.height, val_s1=val.len())
    (out_dir / "encoder_report.json").write_text(json.dumps(report, indent=2))
    return report


def set_stage4_meta(bundle_dir: Path, rule: dict, report: dict, features: list = stage4.FEATURES,
                    models: list = (STAGE4_FILE,)) -> None:
    """Record stage 4 (stacker files, features, rule, how it was made) in a saved bundle.json."""
    path = bundle_dir / BUNDLE_FILE
    meta = json.loads(path.read_text())
    meta["config"] = Config.from_dict(meta["config"]).to_dict()  # writes the stage-4 settings in use
    meta["stage4"] = {"features": list(features), "models": list(models), "rule": rule}
    meta["report"]["stage4"] = report
    path.write_text(json.dumps(meta, indent=2))


def predict(data_dir: Path, bundle: Bundle, cache: Path | None = None, bundle_dir: Path | None = None
            ) -> tuple[pl.Series, pl.DataFrame, pl.DataFrame, dict]:
    """Matches and candidates for every test Source 1 record, checked against the
    submission rules. Returns (ids, match lists, candidate lists, report)."""
    if bundle.config.stage4_scope != "none" and (bundle.stage4 is None or bundle_dir is None
                                                 or not (bundle_dir / pair_encoder.ENCODER_DIR).is_dir()):
        raise data.DataError(f"stage4_scope={bundle.config.stage4_scope!r} needs a bundle with stage 4 (encoder/ "
                             f"and {STAGE4_FILE}); see `stage4-import` / `stage4-train`")
    raw = data.load_split(data_dir, "test")
    prov = provenance(data_dir, "test")
    native = bundle.native_words or {}
    text.set_native_words(native)
    recs, pairs = candidate_table(raw, bundle.config, {**prov["inputs"], "native_words": _dict_hash(native)}, cache)
    pair_text = pair_encoder.texts(raw) if bundle.config.stage4_scope != "none" else None
    del raw
    t0 = time.time()
    tables = alignment.pool_tables(recs)
    if bundle.vocab:
        tables = alignment.with_vocab(tables, bundle.vocab)
        log.info("training-pool vocabulary tables for %s", bundle.vocab["floor"]["country_key"].to_list())
    cfg = bundle.config
    if cfg.feature_caps:
        log.info("predict-time feature caps %s on countries %s", cfg.feature_caps,
                 cfg.feature_cap_countries or "ALL")
    probs = score_all(recs, pairs, bundle.stages, cfg.feature_chunk, tables, bundle.variants,
                      cfg.feature_caps, cfg.feature_cap_countries) \
        if pairs.height else {"stage2": np.zeros(0), "full": np.zeros(0)}
    log.info("scored %d pairs with %d stage(s) in %.1fs (peak %.1f GB)", pairs.height, len(bundle.stages),
             time.time() - t0, peak_gb())
    del tables
    keep_cols = list(RECORD_COLUMNS)
    if bundle.stage4 is not None and stage4.feature_set(bundle.stage4.features) != "A":
        keep_cols += [c for c in stage4.RECORD_COLUMNS if c not in keep_cols]
    records, pairs = recs.select(keep_cols), pairs.select("i", "j")
    del recs
    pair_country = records["country_key"].to_numpy()[pairs["i"].to_numpy()]
    prob = select_probability(probs, bundle, pair_country)
    scored = pairs.with_columns([pl.Series(f"prob_{k}", v.astype(np.float32)) for k, v in probs.items()])
    if pair_text is not None:
        # Every active pair is encoded, whatever the scope, so `decide` can switch scope.
        records = records.with_columns(pl.Series("addr_empty", pair_encoder.empty_address(pair_text)))
        act = np.flatnonzero(prob >= cfg.stage4_active)
        ce = np.full(pairs.height, np.nan, dtype=np.float32)
        t0 = time.time()
        ce[act] = pair_encoder.score(bundle_dir / pair_encoder.ENCODER_DIR, pair_text, pairs["i"].to_numpy()[act],
                                     pairs["j"].to_numpy()[act], cfg.ce_params)
        log.info("pair encoder: %d active pairs in %.0fs (peak %.1f GB)", len(act), time.time() - t0, peak_gb())
        del pair_text
        pairs = pairs.with_columns(pl.Series("ce_logit", ce))
        scored = scored.with_columns(pl.Series("ce_logit", ce))
    scores_dir = bundle_dir / "test_scores" if bundle_dir else None
    if scores_dir:
        scores_dir.mkdir(parents=True, exist_ok=True)
        scored.write_parquet(scores_dir / "pairs.parquet")
        records.write_parquet(scores_dir / "records.parquet")
    del scored
    return finish(records, pairs, prob, bundle, prov)


def select_probability(probs: dict, bundle: Bundle, pair_country: np.ndarray) -> np.ndarray:
    """The probability each pair is decided on: the configured stage-3 choice, with
    per-country overrides (stage3_by_country) applied by the Source 1 record's country."""
    def column(choice):
        key = "stage2" if choice == "none" else choice
        if key not in probs:
            raise data.DataError(f"stage-3 choice {choice!r} not available; have {sorted(probs)}")
        return probs[key]
    prob = column(bundle.config.stage3_use).copy()
    for country, choice in bundle.config.stage3_by_country.items():
        mask = pair_country == country
        prob[mask] = column(choice)[mask]
    return prob


def load_scores(bundle_dir: Path) -> tuple[pl.DataFrame, pl.DataFrame, dict]:
    """Records, candidate pairs (with the pair-encoder logit when predict saved one) and every
    saved probability column from the last predict()."""
    d = bundle_dir / "test_scores"
    if not (d / "pairs.parquet").is_file():
        raise data.DataError(f"no saved test scores in {d}; run `predict` first")
    scored = pl.read_parquet(d / "pairs.parquet")
    probs = {c.removeprefix("prob_"): scored[c].to_numpy() for c in scored.columns if c.startswith("prob_")}
    pairs = scored.select(c for c in ("i", "j", "ce_logit") if c in scored.columns)
    return pl.read_parquet(d / "records.parquet"), pairs, probs


def finish(records: pl.DataFrame, pairs: pl.DataFrame, prob: np.ndarray, bundle: Bundle, prov: dict | None = None
           ) -> tuple[pl.Series, pl.DataFrame, pl.DataFrame, dict]:
    """Decision rule, id lists and submission checks from scored pairs. With stage 4 in scope,
    `pairs` carries the pair-encoder logit (ce_logit) and `records` the empty-address flag."""
    country = records["country_key"].to_numpy()
    i, j = pairs["i"].to_numpy(), pairs["j"].to_numpy()
    s1 = records.filter(pl.col("src") == 1)
    cfg = bundle.config
    s4 = stage4.countries(cfg.stage4_scope, set(bundle.report.get("oof_f05_by_country", {})), np.unique(country[i]))
    stage4_report = {"scope": cfg.stage4_scope, "countries": [str(c) for c in s4]}
    if s4:
        act = (prob >= cfg.stage4_active) & np.isin(country[i], s4)
        ce = pairs["ce_logit"].to_numpy() if "ce_logit" in pairs.columns else None
        if bundle.stage4 is None or ce is None or np.isnan(ce[act]).any():
            raise data.DataError("saved test scores lack pair-encoder logits; re-run `predict` with stage4_scope set")
        if stage4.feature_set(bundle.stage4.features) != "A" and not set(stage4.RECORD_COLUMNS) <= set(records.columns):
            raise data.DataError("saved test records lack the string columns the stage-4 stacker reads; re-run `predict`")
        prob = stage4.probability(bundle.stage4, i, j, prob, ce, records["addr_empty"].to_numpy(), act, records)
        stage4_report["active_pairs"] = int(act.sum())
        log.info("stage 4 rescored %d active pairs of %s", int(act.sum()), stage4_report["countries"])
    pairs = pairs.select("i", "j")
    keep, calibration = decide(i, j, prob, country[i], s1["country_key"].to_numpy(), bundle, s4)
    if bundle.config.unseen_rules:
        missing = [c for c in unseen_rules.COLUMNS if c not in records.columns]
        if missing:
            raise data.DataError(f"saved test scores lack {missing} (needed by unseen_rules); re-run `predict`")
        seen = set(bundle.report.get("oof_f05_by_country", {}))
        for c in np.unique(country[i]):
            if c in seen:
                continue
            m = np.flatnonzero(country[i] == c)
            keep[m], counts = unseen_rules.correct(records, i[m], j[m], prob[m], keep[m])
            calibration.setdefault(str(c), {}).update(counts)
            log.info("unseen country %s: post-decision rules %s", c, counts)
    matched = pairs.filter(pl.Series(keep))
    id_of = records["entity_id"].to_numpy()
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
              "unseen_country_calibration": calibration, "stage4": stage4_report, "provenance": prov}
    return ids, match_lists, cand_lists, report
