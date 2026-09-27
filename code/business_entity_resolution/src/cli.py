"""Command line entry point.

    python -m src.cli train   --data DATA --model artifacts/model
    python -m src.cli predict --data DATA --model artifacts/model --out OUTPUT
    python -m src.cli run     --data DATA --model artifacts/model --out OUTPUT
    python -m src.cli block   --data DATA          (candidate recall on train, no training)
    python -m src.cli decide  --model artifacts/model --out OUTPUT   (re-decide from saved test scores)
    python -m src.cli vocab   --data DATA --model artifacts/model   (add training-pool vocabulary tables to a bundle)
    python -m src.cli configure --model artifacts/model --set NAME=JSON   (change a setting saved in a bundle)
    python -m src.cli stage4-train  --data DATA --model artifacts/model --base BASE   (fine-tune + stack stage 4)
    python -m src.cli stage4-import --model artifacts/model --encoder DIR --stacker FILE.txt --stacker-meta FILE.json

DATA is the dataset/ folder holding train/ and test/. `run` = train + predict.
Exit codes: 0 success, 2 bad input or output data.
"""
import argparse
import dataclasses
import json
import logging
import shutil
import sys
from pathlib import Path

import lightgbm as lgb

from . import data, pair_encoder, pipeline, stage4
from .config import Config

log = logging.getLogger("ber")


def _config(args) -> Config:
    return Config(k=args.k, k_rev=args.k_rev, max_df=args.max_df, train_s1_sample=args.sample,
                  folds=args.folds, seed=args.seed)


def _train(args) -> pipeline.Bundle:
    bundle = pipeline.train(args.data, _config(args), country_check=not args.skip_country_check,
                            cache=args.cache)
    bundle.save(args.model)
    return bundle


def _predict(args, bundle=None) -> None:
    bundle = bundle or pipeline.Bundle.load(args.model)
    if getattr(args, "feature_cap", None):
        caps = {k: float(v) for k, v in (item.split("=", 1) for item in args.feature_cap)}
        bundle.config = dataclasses.replace(bundle.config, feature_caps=caps,
                                            feature_cap_countries=(args.cap_countries or "").split(",")
                                            if args.cap_countries else [])
    _, matches, cands, report = pipeline.predict(args.data, bundle, cache=args.cache, bundle_dir=args.model)
    _write(args, matches, cands, report)


def _decide(args) -> None:
    """Rewrite the outputs from the scores saved by the last predict (no rescoring)."""
    bundle = pipeline.Bundle.load(args.model)
    if args.unseen_none_cap is not None:
        bundle.config = dataclasses.replace(bundle.config, unseen_none_cap=args.unseen_none_cap)
    if args.stage3 is not None:
        bundle.config = dataclasses.replace(bundle.config, stage3_use=args.stage3)
    if args.stage3_country:
        overrides = dict(item.split("=", 1) for item in args.stage3_country)
        bad = [v for v in overrides.values() if v not in pipeline.STAGE3_CHOICES]
        if bad:
            raise data.DataError(f"--stage3-country values must be one of {pipeline.STAGE3_CHOICES}, got {bad}")
        bundle.config = dataclasses.replace(bundle.config, stage3_by_country=overrides)
    if args.stage4_scope is not None:
        bundle.config = dataclasses.replace(bundle.config, stage4_scope=args.stage4_scope)
    records, pairs, probs = pipeline.load_scores(args.model)
    pair_country = records["country_key"].to_numpy()[pairs["i"].to_numpy()]
    _, matches, cands, report = pipeline.finish(records, pairs, pipeline.select_probability(probs, bundle, pair_country),
                                                bundle)
    report["stage3_use"] = bundle.config.stage3_use
    report["stage3_by_country"] = bundle.config.stage3_by_country
    _write(args, matches, cands, report)


def _stage4_import(args) -> None:
    """Put an already fine-tuned encoder and a trained stage-4 stacker into a bundle."""
    pipeline.Bundle.load(args.model)  # the bundle must exist
    meta = json.loads(args.stacker_meta.read_text())
    if meta.get("features") != stage4.FEATURES or lgb.Booster(model_file=str(args.stacker)).num_feature() != len(stage4.FEATURES):
        raise data.DataError(f"{args.stacker}: stacker features are not {stage4.FEATURES}")
    if not (args.encoder / "model.safetensors").is_file():
        raise data.DataError(f"no model.safetensors in {args.encoder}")
    shutil.copytree(args.encoder, args.model / pair_encoder.ENCODER_DIR, dirs_exist_ok=True)
    shutil.copyfile(args.stacker, args.model / pipeline.STAGE4_FILE)
    pipeline.set_stage4_meta(args.model, meta["rule"], {
        "imported": {"encoder": args.encoder.name, "stacker": args.stacker.name, "rounds": meta.get("rounds"),
                     "encoder_sha256": pipeline._sha256(args.encoder / "model.safetensors"),
                     "stacker_sha256": pipeline._sha256(args.stacker)}})
    bundle = pipeline.Bundle.load(args.model)
    log.info("stage 4 imported into %s: %d trees, rule %s", args.model, bundle.stage4.boosters[0].num_trees(),
             bundle.stage4_rule)


def _vocab(args) -> None:
    """Save the training pool's vocabulary tables into a bundle trained without them."""
    bundle = pipeline.Bundle.load(args.model)
    pipeline.save_vocab(args.model, pipeline.train_vocab(args.data, bundle))
    log.info("saved training-pool vocabulary tables to %s", args.model)


def _configure(args) -> None:
    """Set config fields in a saved bundle (models untouched), e.g. unseen_rules=true."""
    path = args.model / pipeline.BUNDLE_FILE
    if not path.is_file():
        raise data.DataError(f"no model bundle at {args.model}")
    meta = json.loads(path.read_text())
    names = {f.name for f in dataclasses.fields(Config)}
    for item in args.set:
        name, eq, value = item.partition("=")
        if not eq:
            raise data.DataError(f"--set expects NAME=JSON, got {item!r}")
        if name not in names:
            raise data.DataError(f"unknown setting {name!r}")
        try:
            meta["config"][name] = json.loads(value)
        except json.JSONDecodeError as exc:
            raise data.DataError(f"--set {name}: value is not JSON ({exc})") from None
    stage4.countries(Config.from_dict(meta["config"]).stage4_scope, set(), [])  # validates the scope
    path.write_text(json.dumps(meta, indent=2))
    log.info("bundle config now %s", meta["config"])


def _write(args, matches, cands, report) -> None:
    data.write_id_lists(args.out / "matching_results.tsv", data.MATCH_COLUMN, matches)
    data.write_id_lists(args.out / "candidate_pairs.tsv", data.CANDIDATE_COLUMN, cands)
    (args.model / "predict_report.json").write_text(json.dumps(report, indent=2))
    log.info("wrote %s", args.out)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(prog="python -m src.cli", description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="command", required=True)
    for name in ("train", "predict", "run", "block", "decide"):
        p = sub.add_parser(name)
        p.add_argument("--data", type=Path, required=name != "decide", help="dataset/ folder")
        p.add_argument("--model", type=Path, default=Path("artifacts/model"), help="bundle folder")
        p.add_argument("--cache", type=Path, default=None,
                       help="reuse prepared records and candidates across runs (keyed on data, settings and code)")
        if name == "predict":
            p.add_argument("--feature-cap", action="append", default=None, metavar="NAME=VALUE",
                           help="clip a stage-1/2 feature at predict time, e.g. amb_s23_addr_2=1 (repeatable)")
            p.add_argument("--cap-countries", default=None,
                           help="comma-separated countries the caps apply to (default: all)")
        if name in ("predict", "run", "decide"):
            p.add_argument("--out", type=Path, required=True, help="output folder for the TSVs")
        if name == "decide":
            p.add_argument("--unseen-none-cap", type=float, default=None,
                           help="override the unseen-country calibration cap (1 disables it)")
            p.add_argument("--stage3", choices=pipeline.STAGE3_CHOICES, default=None,
                           help="which probabilities decide: full, robust or none (stage 2)")
            p.add_argument("--stage3-country", action="append", default=None, metavar="COUNTRY=CHOICE",
                           help="per-country override, e.g. india=full (repeatable)")
            p.add_argument("--stage4-scope", choices=stage4.SCOPES, default=None,
                           help="countries decided on stage 4: none, seen (with training labels) or all")
        if name != "predict":
            d = Config()
            p.add_argument("--k", type=int, default=d.k, help="candidates per Source 1 record and source")
            p.add_argument("--k-rev", type=int, default=d.k_rev, help="Source 1 candidates per Source 2/3 record")
            p.add_argument("--max-df", type=int, default=d.max_df, help="skip blocking keys more common than this")
            p.add_argument("--sample", type=int, default=d.train_s1_sample, help="Source 1 records used to train")
            p.add_argument("--folds", type=int, default=d.folds)
            p.add_argument("--seed", type=int, default=d.seed)
            p.add_argument("--skip-country-check", action="store_true",
                           help="skip the leave-one-country-out estimate")
    p = sub.add_parser("vocab")
    p.add_argument("--data", type=Path, required=True, help="dataset/ folder")
    p.add_argument("--model", type=Path, default=Path("artifacts/model"), help="bundle folder")
    p = sub.add_parser("stage4-train")
    p.add_argument("--data", type=Path, required=True, help="dataset/ folder")
    p.add_argument("--model", type=Path, default=Path("artifacts/model"), help="bundle folder (with oof.parquet)")
    p.add_argument("--base", type=Path, required=True, help="local copy of the base encoder checkpoint")
    p.add_argument("--cache", type=Path, default=None, help="reuse the prepared training records and candidates")
    p = sub.add_parser("stage4-import")
    p.add_argument("--model", type=Path, default=Path("artifacts/model"), help="bundle folder")
    p.add_argument("--encoder", type=Path, required=True, help="fine-tuned encoder folder (weights + tokenizer)")
    p.add_argument("--stacker", type=Path, required=True, help="stage-4 LightGBM model file")
    p.add_argument("--stacker-meta", type=Path, required=True, help="JSON with its features and decision rule")
    p = sub.add_parser("configure")
    p.add_argument("--model", type=Path, default=Path("artifacts/model"), help="bundle folder")
    p.add_argument("--set", action="append", required=True, metavar="NAME=JSON",
                   help='config field and JSON value, e.g. feature_caps=\'{"amb_s23_addr_2": 1}\' (repeatable)')
    args = ap.parse_args(argv)
    logging.basicConfig(level=logging.INFO, stream=sys.stderr,
                        format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    try:
        if args.command == "block":
            report = pipeline.evaluate_blocking(args.data, _config(args), cache=args.cache)
            print(json.dumps(report, indent=2))
        elif args.command == "train":
            _train(args)
        elif args.command == "decide":
            _decide(args)
        elif args.command == "predict":
            _predict(args)
        elif args.command == "vocab":
            _vocab(args)
        elif args.command == "configure":
            _configure(args)
        elif args.command == "stage4-train":
            pipeline.train_stage4(args.data, args.model, args.base, cache=args.cache)
        elif args.command == "stage4-import":
            _stage4_import(args)
        else:
            _predict(args, _train(args))
    except data.DataError as exc:
        log.error("%s", exc)
        return 2
    return 0


if __name__ == "__main__":
    sys.exit(main())
