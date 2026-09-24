"""Command line entry point.

    python -m src.cli train   --data DATA --model artifacts/model
    python -m src.cli predict --data DATA --model artifacts/model --out OUTPUT
    python -m src.cli run     --data DATA --model artifacts/model --out OUTPUT

DATA is the dataset/ folder holding train/ and test/. `run` = train + predict.
Exit codes: 0 success, 2 bad input or output data.
"""
import argparse
import json
import logging
import sys
from pathlib import Path

from . import data, pipeline
from .config import Config

log = logging.getLogger("ber")


def _config(args) -> Config:
    base = Config()
    return Config(k=args.k, k_rev=args.k_rev, block_by_country=not args.no_country_blocking,
                  folds=args.folds, seed=args.seed, lgb_params=base.lgb_params)


def _train(args) -> pipeline.Bundle:
    bundle = pipeline.train(args.data, _config(args), country_check=not args.skip_country_check)
    bundle.save(args.model)
    return bundle


def _predict(args, bundle=None) -> None:
    bundle = bundle or pipeline.Bundle.load(args.model)
    ids, matches, cands, report = pipeline.predict(args.data, bundle)
    data.write_ids(args.out / "matching_results.tsv", data.MATCH_COLUMN, ids, matches)
    data.write_ids(args.out / "candidate_pairs.tsv", data.CANDIDATE_COLUMN, ids, cands)
    (args.model / "predict_report.json").write_text(json.dumps(report, indent=2))
    log.info("wrote %s", args.out)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(prog="python -m src.cli", description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="command", required=True)
    for name in ("train", "predict", "run"):
        p = sub.add_parser(name)
        p.add_argument("--data", type=Path, required=True, help="dataset/ folder")
        p.add_argument("--model", type=Path, default=Path("artifacts/model"), help="bundle folder")
        if name != "train":
            p.add_argument("--out", type=Path, required=True, help="output folder for the TSVs")
        if name != "predict":
            d = Config()
            p.add_argument("--k", type=int, default=d.k)
            p.add_argument("--k-rev", type=int, default=d.k_rev)
            p.add_argument("--folds", type=int, default=d.folds)
            p.add_argument("--seed", type=int, default=d.seed)
            p.add_argument("--no-country-blocking", action="store_true")
            p.add_argument("--skip-country-check", action="store_true",
                           help="skip the leave-one-country-out estimate")
    args = ap.parse_args(argv)
    logging.basicConfig(level=logging.INFO, stream=sys.stderr,
                        format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    try:
        if args.command == "train":
            _train(args)
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
