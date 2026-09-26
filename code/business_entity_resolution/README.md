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

`requirements.txt` includes `torch` and `transformers` for stage 4 (below).

Hardware used: one laptop (Apple M-series, 24 GB unified memory). Stages 1-3 run on
the CPU. Stage 4's pair encoder runs on the Apple GPU (PyTorch MPS), on CUDA when
present, else on the CPU (much slower). No network access is needed at run time:
every model is read from the bundle, with `HF_HUB_OFFLINE=1`.

## Reproduce end to end

`DATA` is the challenge `dataset/` folder (it contains `train/` and `test/`).

```bash
.venv/bin/python -m src.cli run --data DATA --model artifacts/model --out ../../output
```

This trains, validates, saves the model bundle to `artifacts/model/`, and writes
`../../output/matching_results.tsv` and `../../output/candidate_pairs.tsv`.
It uses the defaults in `src/config.py`, which save the training-pool vocabulary
tables but leave the other predict-time corrections off (`feature_caps={}`,
`unseen_rules=false`). The submitted outputs come from the v8 bundle below.

The two stages can also run separately:

```bash
.venv/bin/python -m src.cli train   --data DATA --model artifacts/model
.venv/bin/python -m src.cli predict --data DATA --model artifacts/model --out ../../output
```

`predict` reads every setting (blocking parameters, feature list, decision
rule) from the bundle, so it cannot drift from what was validated. The same
inputs and bundle give byte-identical outputs.

### The submitted bundle (v8)

`artifacts/v8` is the trained v7 bundle with predict-time corrections and no
retraining. It was made from `artifacts/v7` (the bundle `train` saved for v7: models,
dictionary, report) with:

```bash
cp -R artifacts/v7 artifacts/v8
.venv/bin/python -m src.cli vocab --data DATA --model artifacts/v8
.venv/bin/python -m src.cli configure --model artifacts/v8 \
    --set 'feature_caps={"amb_s23_addr_2": 1.0}' --set 'feature_cap_countries=[]' \
    --set 'stage3_use="none"' --set 'stage3_by_country={"india": "full"}' \
    --set 'unseen_empty_target=0.05597295153414383' --set 'unseen_rules=true'
.venv/bin/python -m src.cli predict --data DATA --model artifacts/v8 --out ../../output
```

- `vocab` saves the training pool's token frequency tables (`vocab_*.parquet`) into
  the bundle; `train` saves them itself. Predict then reads the leftover-token
  features of every country with a training pool at training-time frequencies.
- `feature_caps` clips the pool ambiguity count `amb_s23_addr_2` at 1: the test
  pool's decoy copy groups inflate it into false copy-consensus.
- `unseen_empty_target` is the empty-prediction share the unseen country was
  calibrated to with the v7 probabilities (the seen countries' share then),
  pinned so the corrections above do not move it.
- `unseen_rules` turns on the post-decision rules of `src/unseen_rules.py` for
  countries absent from training.

### Stage 4 and the v9 bundle

`artifacts/v9` is the v8 bundle plus stage 4: a fine-tuned multilingual pair encoder
(`encoder/`) and a LightGBM stacker (`stage4.txt`) that rescores the pairs whose decided
probability is at least `stage4_active` (0.002). The encoder reads the raw text of both
records, `lower(business_name) | lower(business_address)`, accents and legal forms kept,
and gives one match logit. The stacker's 8 features: the logit of the decided
probability, the encoder logit, an empty-candidate-address flag, and for both scores the
rank and the gap to the best other active candidate of the same Source 1 record, plus the
number of active candidates of that record. It has its own decision rule
(expected F0.5, `missed` 0.05). `stage4_scope` chooses where it applies:

- `none`: no stage 4 (the v8 decision);
- `seen`: countries with training labels (US, India); a country absent from training
  (France) is decided exactly as in v8;
- `all`: every country; an unseen country then goes through the same empty-share
  calibration and `unseen_rules` as in v8, on the stage-4 probabilities.

The encoder was fine-tuned from `intfloat/multilingual-e5-small` (MIT licence, revision
614241f622f53c4eeff9890bdc4f31cfecc418b3, 117.65M parameters with the classification head)
and the stacker trained on the v7 out-of-fold probabilities; `stage4-import` puts those
already-trained files into a bundle without retraining:

```bash
mkdir artifacts/v9 && cp artifacts/v8/*.* artifacts/v9/        # v8 without test_scores/
.venv/bin/python -m src.cli stage4-import --model artifacts/v9 \
    --encoder ENCODER_DIR --stacker stage4_A.txt --stacker-meta stage4_A.json
.venv/bin/python -m src.cli configure --model artifacts/v9 --set 'stage4_scope="seen"'
.venv/bin/python -m src.cli predict --data DATA --model artifacts/v9 --out ../../output
.venv/bin/python -m src.cli decide  --model artifacts/v9 --out ../../output_all --stage4-scope all
```

`predict` encodes every active pair of every country (8.08M test pairs, about 40 minutes
on the Apple GPU, on top of about 45 minutes for stages 1-3) and saves the logits with the
test scores, so `decide --stage4-scope` switches scope in a few minutes without rescoring.
The encoder runs in a separate process: on macOS, torch's CPU kernels hang in a process
where LightGBM has already run (each brings its own OpenMP runtime).

Encoder logits do not depend on how pairs are batched (checked bit for bit on the Apple
GPU), so predict gives the same output on the same device. CUDA or CPU arithmetic differs
slightly and can move a few decisions. The weights ship in float32 (470 MB): stored as
bfloat16 or float16 they changed 20-23% of the logits on a 30k-pair check.

**Training stage 4 from scratch** (instead of importing) needs the base checkpoint once,
downloaded outside the run (network only for this step):

```bash
hf download intfloat/multilingual-e5-small --revision 614241f622f53c4eeff9890bdc4f31cfecc418b3 \
    --local-dir models/multilingual-e5-small
.venv/bin/python -m src.cli stage4-train --data DATA --model artifacts/v9 \
    --base models/multilingual-e5-small --cache CACHE
```

It reads the bundle's `oof.parquet` and `ce_params` (`src/config.py`):

1. Encoder data: 360,000 training Source 1 records drawn outside the matcher's sample and
   holdout (20,000 of them kept for a validation AUC), each with its top 6 candidates by
   retrieval score plus every true pair among its candidates; fit pairs whose candidate
   belongs to an active pair of the matcher's sample are dropped, so the encoder never saw
   the candidates stage 4 is trained on.
2. Fine-tuning: one pass, batch 128, AdamW (lr 3e-5, weight decay 0.01), 5% linear warmup
   then linear decay, gradient clip 1.0, word embeddings frozen, bfloat16 autocast,
   max length 128; batches of similar token length. About 72 minutes on the Apple GPU.
3. Stacker: the active sample pairs (2.32M) with their out-of-fold decided probability and
   the new encoder's logit; 5-fold cross-validation grouped by Source 1 record, the
   decision rule chosen on the out-of-fold stage-4 probability, and
   the final model refit on all active pairs with the mean best iteration.

The shipped weights came from the same recipe run as separate steps, with two
differences: the learning-rate schedule also advanced with elapsed time (a 75-minute
budget, which the pass finished inside), and 22,899 more fit pairs were dropped whose
candidate appears in an active holdout pair (the holdout measured the gain: +0.00179,
95% CI [0.00156, 0.00202], US and India). GPU training is not bit-reproducible, so a
retrained encoder gives close but not identical logits.

### Files the submission zip needs

`artifacts/` is not in git, so the submission zip must carry the bundle. `predict`
needs these files of `artifacts/v8/`: `bundle.json`, `native_words.json`, every
`stage*_fold*.txt` (including `stage3_robust_fold*.txt`) and the three
`vocab_*.parquet`. Leave out `oof.parquet` (error analysis only, 184 MB) and
`test_scores/` (rewritten by `predict`, 1.2 GB). A bundle without the `vocab_*.parquet`
files still loads but reads the leftover-token features at test-pool frequencies. The
v9 bundle also needs `stage4.txt` and `encoder/` (`config.json`, `model.safetensors`,
`tokenizer.json`, `tokenizer_config.json`). Check the unzipped copy:

```bash
.venv/bin/python -c "from pathlib import Path; from src.pipeline import Bundle; \
b = Bundle.load(Path('artifacts/v8')); assert b.vocab and b.config.unseen_rules; print('ok')"
```

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
| Features | `src/features.py`, `src/alignment.py` | 78 pair features: fuzzy scores, token and number alignment with corpus frequency (the training pool's, for countries that have one), ambiguity counts, character similarity, house-number agreement, retrieval context on both sides; optional predict-time caps (`feature_caps`) |
| Model | `src/pipeline.py`, `src/groupfeat.py` | Three LightGBM stages cross-validated with folds grouped by Source 1 record: pair features; + stage-1 competition context; + group features comparing each candidate with the likely members of its group. Test prediction = mean of the fold models |
| Stage 4 | `src/pair_encoder.py`, `src/stage4.py` | Optional (`stage4_scope`): a fine-tuned multilingual pair encoder scores the raw text of every pair whose decided probability is at least 0.002; a LightGBM stacker combines its logit with that probability and both scores' rank and gap within the Source 1 record |
| Decide | `src/pipeline.py` | Exclusive owner rule, then per-record expected-F0.5 selection; stage-3 or stage-2 probabilities chosen per country (`stage3_use`, `stage3_by_country`); unseen countries calibrated to the seen countries' empty-prediction share (or a pinned `unseen_empty_target`), then label-free rules for them (`src/unseen_rules.py`: legal-form conflict drops, true-variant word and name-tail adds under the owner rule) |
| Write | `src/data.py` | Enforces every submission rule, then writes atomically |

Settings live in `src/config.py`; the CLI overrides `--k`, `--k-rev`,
`--max-df`, `--sample`, `--folds` and `--seed`.

Other commands:

```bash
.venv/bin/python -m src.cli block  --data DATA --cache CACHE          # blocking recall on train only
.venv/bin/python -m src.cli decide --model artifacts/model --out OUT \
    --stage3 none --stage3-country india=full                         # re-decide from saved test scores
.venv/bin/python -m src.cli vocab  --data DATA --model artifacts/model  # add training-pool tables to a bundle
.venv/bin/python -m src.cli configure --model artifacts/model --set NAME=JSON   # change a saved setting
.venv/bin/python -m src.cli decide --model artifacts/v9 --out OUT --stage4-scope all  # stage-4 scope from saved scores
.venv/bin/python -m src.cli stage4-train  --data DATA --model artifacts/model --base BASE  # fine-tune + stack
.venv/bin/python -m src.cli stage4-import --model artifacts/model --encoder DIR \
    --stacker FILE.txt --stacker-meta FILE.json                        # add trained stage-4 files
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
the reference), the decision rules, the training-pool vocabulary tables (identity
on the training pool, per-country swap, `vocab` equals what `train` saved), each
unseen-country rule on a small example, bundle configuration, input and output
checks, and a full
train→predict run on a generated dataset with an unseen third country, checked
with the challenge validator. Stage 4: the pair text and empty-address flag, the 8
features against hand-computed values, scope routing, the per-country decision rule,
config round trip, and `stage4-train` → `predict` → `decide` with a tiny randomly
initialised encoder on the CPU (seen countries equal under `seen` and `all`, the unseen
country equal under `seen` and `none`), plus `stage4-import`.

## Constraints respected

- Only the provided data is used: no external lookups, geocoding or APIs.
- The models are LightGBM (MIT licence) and, for stage 4, a fine-tuned
  `intfloat/multilingual-e5-small` (MIT licence, 117.65M parameters), fine-tuned on the
  training files only and read offline from the bundle.
- Country is used for blocking only and is never a model feature, so the model
  applies unchanged to a country missing from training.
