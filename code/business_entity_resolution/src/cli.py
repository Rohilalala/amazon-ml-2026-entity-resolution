"""Command line entry point.

    python -m src.cli train   --data DATA --model artifacts/model
    python -m src.cli predict --data DATA --model artifacts/model --out OUTPUT
    python -m src.cli run     --data DATA --model artifacts/model --out OUTPUT
    python -m src.cli block   --data DATA          (candidate recall on train, no training)
    python -m src.cli decide  --model artifacts/model --out OUTPUT   (re-decide from saved test scores)

DATA is the dataset/ folder holding train/ and test/. `run` = train + predict.
Exit codes: 0 success, 2 bad input or output data.
"""
import argparse
import dataclasses
import json
import logging
import sys
from pathlib import Path

from . import data, pipeline
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
    _, matches, cands, report = pipeline.predict(args.data, bundle, cache=args.cache, bundle_dir=args.model)
    _write(args, matches, cands, report)


def _decide(args) -> None:
    """Rewrite the outputs from the scores saved by the last predict (no rescoring)."""
    bundle = pipeline.Bundle.load(args.model)
    if args.unseen_none_cap is not None:
        bundle.config = dataclasses.replace(bundle.config, unseen_none_cap=args.unseen_none_cap)
    records, pairs, prob = pipeline.load_scores(args.model)
    _, matches, cands, report = pipeline.finish(records, pairs, prob, bundle)
    _write(args, matches, cands, report)


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
        if name in ("predict", "run", "decide"):
            p.add_argument("--out", type=Path, required=True, help="output folder for the TSVs")
        if name == "decide":
            p.add_argument("--unseen-none-cap", type=float, default=None,
                           help="override the unseen-country calibration cap (1 disables it)")
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
        else:
            _predict(args, _train(args))
    except data.DataError as exc:
        log.error("%s", exc)
        return 2
    return 0


if __name__ == "__main__":
    sys.exit(main())
