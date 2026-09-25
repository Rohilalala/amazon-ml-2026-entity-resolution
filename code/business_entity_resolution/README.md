# Business Entity Resolution

Finds, for every Source 1 business record, the Source 2 and Source 3 records
that describe the same real-world business. Output is the two tab-separated
files the challenge asks for: `matching_results.tsv` and `candidate_pairs.tsv`.

## Setup

Python 3.12. From this folder:

```bash
python3.12 -m venv .venv
.venv/bin/pip install -r requirements.txt        # runtime, exact versions
.venv/bin/pip install -r requirements-dev.txt    # + pytest, for the test suite
```

Hardware used: one laptop, 24 GB RAM, CPU only. No GPU and no network access
are needed at run time.

## Reproduce end to end

`DATA` is the challenge `dataset/` folder (it contains `train/` and `test/`).

```bash
.venv/bin/python -m src.cli run --data DATA --model artifacts/model --out ../../output
```

This trains, validates, saves the model bundle to `artifacts/model/`, and writes
`../../output/matching_results.tsv` and `../../output/candidate_pairs.tsv`.

The two stages can also run separately:

```bash
.venv/bin/python -m src.cli train   --data DATA --model artifacts/model
.venv/bin/python -m src.cli predict --data DATA --model artifacts/model --out ../../output
```

`predict` reads every setting (blocking parameters, feature list, decision
rule) from the bundle, so it cannot drift from what was validated. The same
inputs and bundle give byte-identical outputs.

Check the outputs with the challenge validator before uploading:

```bash
python3 utils/validate_submission.py --matching output/matching_results.tsv \
    --candidate output/candidate_pairs.tsv --test-dir dataset/test
```

## Pipeline

| Stage | Module | What it does |
|---|---|---|
| Load | `src/data.py` | Reads TSVs as text (quote characters are data), checks columns, id prefixes, duplicates and ground-truth references |
| Normalise | `src/text.py`, `src/indic.py` | Indic words translated with a dictionary learned from training pairs, then ASCII transliteration; placeholders removed, `&`→`and`, single letters joined (S.A.R.L.→sarl), abbreviations expanded, state names to codes, core name without legal forms, phonetic skeletons |
| Locale | `src/locale_norm.py` | Label-free: address words that are strongly asymmetric between Source 1 and Source 2/3 within a country (France's region/department phrases) are dropped on both sides |
| Block | `src/candidates.py` | Per country and source: records become sparse vectors of rare keys (skeleton tokens and pairs, transliteration-aware variants, name prefix/suffix, numbers, rarest address words and their pairs, cross keys). Top-10 by IDF cosine per source, plus reverse top-3 |
| Features | `src/features.py`, `src/alignment.py` | 78 pair features: fuzzy scores, token and number alignment with corpus frequency, ambiguity counts, character similarity, house-number agreement, retrieval context on both sides |
| Model | `src/pipeline.py`, `src/groupfeat.py` | Three LightGBM stages cross-validated with folds grouped by Source 1 record: pair features; + stage-1 competition context; + group features comparing each candidate with the likely members of its group. Test prediction = mean of the fold models |
| Decide | `src/pipeline.py` | Exclusive owner rule, then per-record expected-F0.5 selection; stage-3 or stage-2 probabilities chosen per country (`stage3_use`, `stage3_by_country`); unseen countries calibrated to the seen countries' empty-prediction share |
| Write | `src/data.py` | Enforces every submission rule, then writes atomically |

Settings live in `src/config.py`; the CLI overrides `--k`, `--k-rev`,
`--max-df`, `--sample`, `--folds` and `--seed`.

Other commands:

```bash
.venv/bin/python -m src.cli block  --data DATA --cache CACHE          # blocking recall on train only
.venv/bin/python -m src.cli decide --model artifacts/model --out OUT \
    --stage3 none --stage3-country india=full                         # re-decide from saved test scores
```

## What training reports

`artifacts/model/bundle.json` records:

- blocking recall and the F0.5 ceiling it allows, overall and per country
- out-of-fold macro F0.5, overall and per country, and the decision rules tried
- a leave-one-country-out estimate (train on one country, score the other) for
  the country that appears only in the test set
- feature importance
- input file SHA-256 hashes, package versions and the git commit

`artifacts/model/predict_report.json` has per-country prediction counts for the
test set. Every stage logs its duration and peak memory.

## Tests

```bash
.venv/bin/python -m pytest
```

Covers normalisation (including transliteration and state names), the metric
(the worked example in the problem statement, and the vectorised form against
the reference), the decision rules, input and output checks, and a full
train→predict run on a generated dataset with an unseen third country, checked
with the challenge validator.

## Constraints respected

- Only the provided data is used: no external lookups, geocoding or APIs.
- The model is LightGBM (MIT licence).
- Country is used for blocking only and is never a model feature, so the model
  applies unchanged to a country missing from training.
