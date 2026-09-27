"""Command line entry point.

    python -m src.cli train   --data DATA --model artifacts/model
    python -m src.cli predict --data DATA --model artifacts/model --out OUTPUT
    python -m src.cli run     --data DATA --model artifacts/model --out OUTPUT
    python -m src.cli block   --data DATA          (candidate recall on train, no training)
    python -m src.cli decide  --model artifacts/model --out OUTPUT   (re-decide from saved test scores)
    python -m src.cli vocab   --data DATA --model artifacts/model   (add training-pool vocabulary tables to a bundle)
    python -m src.cli configure --model artifacts/model --set NAME=JSON   (change a setting saved in a bundle)
    python -m src.cli stage4-train  --data DATA --model artifacts/model --base BASE   (fine-tune + stack stage 4)
    python -m src.cli stage4-import --model artifacts/model --encoder DIR --stacker FILE.txt [--stacker ...] --stacker-meta FILE.json
    python -m src.cli post-import --model artifacts/final --retriever DIR --recall-encoder DIR --head FILE --head-meta FILE --lists FILE
    python -m src.cli final   --data DATA --model artifacts/final --out OUTPUT   (the leaderboard run)
    python -m src.cli encoder-train --data DATA --model BUNDLE --init DIR --out DIR [--draw N --exclude DIR]
    python -m src.cli recall-train  --data DATA --model artifacts/final
    python -m src.cli adapt-labels --data DATA --model BUNDLE --decision TSV --anchor-pool FILE --round N --out DIR
    python -m src.cli adapt-train  --data DATA --model BUNDLE --init DIR --labels DIR --out DIR

DATA is the dataset/ folder holding train/ and test/. `run` = train + predict.
Exit codes: 0 success, 2 bad input or output data.
"""
import argparse
import dataclasses
import json
import logging
import os
import shutil
import sys
from pathlib import Path

import lightgbm as lgb
import polars as pl

from . import adapt, data, final, pair_encoder, pipeline, postprocess, recall, stage4
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
    """Put an already fine-tuned encoder and trained stage-4 stacker members into a bundle
    (replacing any stage 4 it had)."""
    if not (args.model / pipeline.BUNDLE_FILE).is_file():
        raise data.DataError(f"no model bundle at {args.model}")
    meta = json.loads(args.stacker_meta.read_text())
    names = meta.get("features")
    stage4.feature_set(names or [])  # a known feature set
    for f in args.stacker:
        if lgb.Booster(model_file=str(f)).num_feature() != len(names):
            raise data.DataError(f"{f}: stacker features are not the {len(names)} of {args.stacker_meta}")
    if not (args.encoder / "model.safetensors").is_file():
        raise data.DataError(f"no model.safetensors in {args.encoder}")
    shutil.copytree(args.encoder, args.model / pair_encoder.ENCODER_DIR, dirs_exist_ok=True)
    os.chmod(args.model / pair_encoder.ENCODER_DIR / "model.safetensors", 0o644)
    files = pipeline.stage4_files(len(args.stacker))
    for f, name in zip(args.stacker, files):
        shutil.copyfile(f, args.model / name)
    pipeline.set_stage4_meta(args.model, meta["rule"], {
        "imported": {"encoder": args.encoder.name, "stacker": [f.name for f in args.stacker], "rounds": meta.get("rounds"),
                     "seeds": meta.get("seeds"), "encoder_sha256": pipeline._sha256(args.encoder / "model.safetensors"),
                     "stacker_sha256": [pipeline._sha256(f) for f in args.stacker]}}, names, files)
    bundle = pipeline.Bundle.load(args.model)
    log.info("stage 4 imported into %s: %d member(s), %d features, rule %s", args.model, len(bundle.stage4.boosters),
             len(names), bundle.stage4_rule)


def _post_import(args) -> None:
    """Put the recall models (pretrained retriever, recall pair encoder, head) and the rule lists
    of the post-decision chain into a bundle."""
    pipeline.Bundle.load(args.model)
    rdir = args.model / final.RECALL_DIR
    meta = json.loads(args.head_meta.read_text())
    if meta.get("features") != recall.HEAD_FEATURES:
        raise data.DataError(f"{args.head_meta}: head features are not {recall.HEAD_FEATURES}")
    for name, src in ((recall.RETRIEVER_DIR, args.retriever), (recall.ENCODER_DIR, args.recall_encoder)):
        if not (src / "model.safetensors").is_file():
            raise data.DataError(f"no model.safetensors in {src}")
        shutil.copytree(src, rdir / name, dirs_exist_ok=True)
        os.chmod(rdir / name / "model.safetensors", 0o644)
    shutil.copyfile(args.head, rdir / recall.HEAD_FILE)
    (rdir / recall.HEAD_META).write_text(json.dumps(meta, indent=1))
    lists = pl.read_parquet(args.lists)
    if lists.columns != ["step", "action", "rule", "s1", "m", "src"]:
        raise data.DataError(f"{args.lists}: columns must be step, action, rule, s1, m, src")
    shutil.copyfile(args.lists, args.model / final.LISTS_FILE)
    log.info("post-decision pieces imported into %s: head t_keep %s t_none %s, %d list rows", args.model,
             meta["t_keep"], meta["t_none"], lists.height)


def _final(args) -> None:
    """The leaderboard run: stages 1-4, recall adds and the post-decision chain."""
    matches, cands, report = final.predict(args.data, args.model, cache=args.cache, fixed_recall=args.recall_pairs)
    _write(args, matches, cands, report)


def _adapt_labels(args) -> None:
    """Pseudo-labels and anchors of one test-time adaptation round, from the previous round's run."""
    bundle = pipeline.Bundle.load(args.model)
    records, pairs, probs = pipeline.load_scores(args.model)
    country = records["country_key"].to_numpy()[pairs["i"].to_numpy()]
    prob = pipeline.select_probability(probs, bundle, country)
    act = prob >= bundle.config.stage4_active
    i, j, ce = pairs["i"].to_numpy(), pairs["j"].to_numpy(), pairs["ce_logit"].to_numpy()
    p4 = stage4.probability(bundle.stage4, i, j, prob, ce, records["addr_empty"].to_numpy(), act, records)
    T = pl.DataFrame({"i": i[act], "j": j[act], "country": country[act], "p4": p4[act], "ce": ce[act]})
    raw = data.load_split(args.data, "test")
    recs = postprocess.TestRecords(raw.select("entity_id", "src", "country", "business_name", "business_address"))
    ids = records.select("row", "entity_id")

    def read(path):
        return postprocess.pairs(data.read_tsv(path, ["source1_entity_id", data.MATCH_COLUMN]))
    decision = read(args.decision)
    neg = None
    if args.negatives_from:
        pre = [read(a).join(read(b), on=["s1", "m"], how="anti") if b else read(a)
               for a, _, b in (x.partition(":") for x in args.negatives_from)]
        neg = adapt.to_rows(adapt.negatives_of([pl.DataFrame({"s1": q["s1"], "m": q["m"]}) for q in pre], decision,
                                               recs, args.drop_scrambles), ids)
    fit = pl.read_parquet(args.anchor_pool)
    S, A = adapt.build(T, adapt.to_rows(decision, ids), neg, fit, adapt.ROUNDS[args.round])
    args.out.mkdir(parents=True, exist_ok=True)
    S.write_parquet(args.out / "pseudo.parquet")
    A.write_parquet(args.out / "anchor.parquet")
    log.info("round %d: %d pseudo pairs (%d held out), %d anchors -> %s", args.round, S.height, int(S["is_ho"].sum()),
             A.height, args.out)


def _adapt_train(args) -> None:
    """Fine-tune the pair encoder on one round's pseudo-labels and anchors."""
    bundle = pipeline.Bundle.load(args.model)
    text_train = pair_encoder.texts(data.load_split(args.data, "train"))
    text_test = pair_encoder.texts(data.load_split(args.data, "test"))
    report = adapt.train(args.init, args.out, text_train, text_test, pl.read_parquet(args.labels / "anchor.parquet"),
                         pl.read_parquet(args.labels / "pseudo.parquet"), bundle.config.ce_params)
    (args.out / "adapt_report.json").write_text(json.dumps(report, indent=2))


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
    p.add_argument("--base", type=Path, default=None,
                   help="encoder checkpoint to fine-tune from; omit to keep the bundle's encoder/ and train the stacker only")
    p.add_argument("--cache", type=Path, default=None, help="reuse the prepared training records and candidates")
    p = sub.add_parser("stage4-import")
    p.add_argument("--model", type=Path, default=Path("artifacts/model"), help="bundle folder")
    p.add_argument("--encoder", type=Path, required=True, help="fine-tuned encoder folder (weights + tokenizer)")
    p.add_argument("--stacker", type=Path, action="append", required=True,
                   help="stage-4 LightGBM model file (repeat for seed members, averaged)")
    p.add_argument("--stacker-meta", type=Path, required=True, help="JSON with its features and decision rule")
    p = sub.add_parser("post-import")
    p.add_argument("--model", type=Path, default=Path("artifacts/model"), help="bundle folder")
    p.add_argument("--retriever", type=Path, required=True, help="pretrained multilingual-e5-small folder")
    p.add_argument("--recall-encoder", type=Path, required=True, help="pair encoder that scores the recall pairs")
    p.add_argument("--head", type=Path, required=True, help="recall head LightGBM model file")
    p.add_argument("--head-meta", type=Path, required=True, help="JSON with its features and thresholds")
    p.add_argument("--lists", type=Path, required=True, help="lists.parquet of the post-decision steps E-H")
    p = sub.add_parser("final")
    p.add_argument("--data", type=Path, required=True, help="dataset/ folder")
    p.add_argument("--model", type=Path, default=Path("artifacts/final"), help="final bundle folder")
    p.add_argument("--out", type=Path, required=True, help="output folder for the TSVs")
    p.add_argument("--cache", type=Path, default=None, help="reuse prepared records and candidates across runs")
    p.add_argument("--recall-pairs", type=Path, default=None,
                   help="score this saved recall new-pair list instead of retrieving (byte-exact leaderboard run)")
    p = sub.add_parser("encoder-train")
    p.add_argument("--data", type=Path, required=True, help="dataset/ folder")
    p.add_argument("--model", type=Path, required=True, help="bundle (oof.parquet, ce_params)")
    p.add_argument("--init", type=Path, required=True, help="encoder to continue from")
    p.add_argument("--out", type=Path, required=True, help="new encoder folder")
    p.add_argument("--lr", type=float, default=2e-5)
    p.add_argument("--draw", type=int, default=0, help="fresh Source 1 records to draw (0 = all remaining)")
    p.add_argument("--exclude", type=Path, action="append", default=[], help="earlier encoder folder whose draw to skip")
    p.add_argument("--seed", type=int, default=21)
    p.add_argument("--cache", type=Path, default=None)
    p = sub.add_parser("recall-train")
    p.add_argument("--data", type=Path, required=True, help="dataset/ folder")
    p.add_argument("--model", type=Path, required=True, help="bundle with recall/retriever and recall/encoder")
    p.add_argument("--head-s1", type=int, default=200_000, help="training Source 1 records the head learns from")
    p.add_argument("--cache", type=Path, default=None)
    p = sub.add_parser("adapt-labels")
    p.add_argument("--data", type=Path, required=True, help="dataset/ folder")
    p.add_argument("--model", type=Path, required=True, help="previous round's bundle, after `final` (its test_scores)")
    p.add_argument("--decision", type=Path, required=True, help="previous round's final matching_results.tsv")
    p.add_argument("--negatives-from", action="append", default=None, metavar="BEFORE[:AFTER]",
                   help="matching_results.tsv kept before a filter (minus AFTER when given); its pairs outside the "
                        "decision become negatives (repeatable)")
    p.add_argument("--drop-scrambles", action="store_true", help="keep generator typo scrambles out of the negatives")
    p.add_argument("--anchor-pool", type=Path, required=True, help="parquet (i, j, y, country) of labelled training pairs")
    p.add_argument("--round", type=int, choices=sorted(adapt.ROUNDS), required=True)
    p.add_argument("--out", type=Path, required=True)
    p = sub.add_parser("adapt-train")
    p.add_argument("--data", type=Path, required=True, help="dataset/ folder")
    p.add_argument("--model", type=Path, required=True, help="bundle whose ce_params apply")
    p.add_argument("--init", type=Path, required=True, help="encoder the round starts from")
    p.add_argument("--labels", type=Path, required=True, help="adapt-labels output folder")
    p.add_argument("--out", type=Path, required=True, help="adapted encoder folder")
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
        elif args.command == "post-import":
            _post_import(args)
        elif args.command == "final":
            _final(args)
        elif args.command == "encoder-train":
            pipeline.train_encoder(args.data, args.model, args.init, args.out, args.lr, args.draw, args.exclude, args.seed,
                                   cache=args.cache)
        elif args.command == "recall-train":
            final.train_recall_head(args.data, args.model, n_head=args.head_s1, cache=args.cache)
        elif args.command == "adapt-labels":
            _adapt_labels(args)
        elif args.command == "adapt-train":
            _adapt_train(args)
        else:
            _predict(args, _train(args))
    except data.DataError as exc:
        log.error("%s", exc)
        return 2
    return 0


if __name__ == "__main__":
    sys.exit(main())
