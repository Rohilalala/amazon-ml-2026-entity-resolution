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

## Reproduce end to end

`DATA` is the challenge `dataset/` folder (it contains `train/` and `test/`).

```bash
.venv/bin/python -m src.cli run --data DATA --model artifacts/model --out ../../output
```

This trains, validates, saves the model bundle to `artifacts/model/`, and writes
`../../output/matching_results.tsv` and `../../output/candidate_pairs.tsv`.
Runtime on a laptop CPU is minutes; no GPU or network access is used.

The two stages can also run separately:

```bash
.venv/bin/python -m src.cli train   --data DATA --model artifacts/model
.venv/bin/python -m src.cli predict --data DATA --model artifacts/model --out ../../output
```

`predict` reads every setting (blocking parameters, feature list, threshold)
from the bundle, so it cannot drift from what was validated. Same inputs and
bundle give byte-identical outputs.

Check the outputs with the challenge validator before uploading:

```bash
python3 utils/validate_submission.py --matching output/matching_results.tsv \
    --candidate output/candidate_pairs.tsv --test-dir dataset/test
```

## Pipeline

| Stage | Module | What it does |
|---|---|---|
| Load | `src/data.py` | Reads TSVs as text (`QUOTE_NONE`), checks columns, id prefixes, duplicates and ground-truth references |
| Normalise | `src/text.py` | Removes accents, `&`→`and`, expands abbreviations (Pvt/Ltd/Corp, Rd/St), builds a core name without legal forms, extracts numbers and postcodes |
| Block | `src/candidates.py` | Per country: top-k nearest Source 2/3 records by TF-IDF character n-gram cosine on name and on name+address, plus a reverse pass from Source 2/3 to Source 1 |
| Features | `src/features.py` | ~50 country-agnostic pair features: TF-IDF cosines, fuzzy string scores, token/number Jaccard, postcode and house-number agreement, acronyms, and how the pair ranks against competing candidates |
| Model | `src/pipeline.py` | LightGBM, cross-validated with folds grouped by Source 1 entity; the test prediction is the mean of the fold models |
| Decide | `src/pipeline.py` | Keep pairs above a threshold tuned on out-of-fold predictions for the exact macro F0.5 metric; optionally each Source 2/3 record goes to its best Source 1 record only |
| Write | `src/data.py` | Enforces every submission rule, then writes atomically |

Settings live in `src/config.py`; the CLI overrides `--k`, `--k-rev`, `--folds`,
`--seed` and `--no-country-blocking`.

## What training reports

`artifacts/model/bundle.json` records:

- blocking recall and the F0.5 ceiling it allows
- out-of-fold macro F0.5, overall and per country
- a leave-one-country-out estimate (train on one country, score the other) for
  the country that appears only in the test set
- feature importance
- input file SHA-256 hashes, package versions and the git commit

`artifacts/model/predict_report.json` has per-country prediction counts for the
test set.

## Tests

```bash
.venv/bin/python -m pytest
```

Covers normalisation, the metric (including the worked example in the problem
statement), input and output checks, and a full train→predict run on a
generated dataset with an unseen third country.

## Constraints respected

- Only the provided data is used: no external lookups, geocoding or APIs.
- The model is LightGBM (MIT licence).
- Country is used for blocking only and is never a model feature, so the model
  applies unchanged to a country missing from training.
