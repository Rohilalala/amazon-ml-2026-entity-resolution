# Business Entity Resolution (Amazon ML Challenge 2026)

For every business record of Source 1, find the records of Source 2 and Source 3 that describe the
same real-world business. The test pool holds 1,732,544 Source 1 records and 11.7M records in all,
in the US, India and France; France never appears in the training data. Submissions are scored by
macro F0.5 over Source 1 records. The records come from a data generator: true copies carry typos,
abbreviations and appended words, and decoys are near-copies of a real business with a changed house
number or one swapped name word (sister organisations).

This repository is the full pipeline behind our final leaderboard score, **0.989667**: blocking, a
three-stage LightGBM matcher, a fine-tuned multilingual pair encoder with a LightGBM stacker, dense
recall with a pretrained text embedding, test-time adaptation of the encoder, and label-free
post-decision rules for the country absent from training. The shipped model files reproduce the
uploaded leaderboard file byte for byte; [Reproducibility](#reproducibility) says which runs measured
this and which command has not yet been run end to end.

## Results

Public leaderboard (macro F0.5), in the order the pieces were added:

| Step | Leaderboard |
|---|---|
| 2-stage LightGBM matcher with house-number features | 0.964386 |
| 3 stages, rarest-key blocking, France locale normalisation and empty-share calibration (v7) | 0.971137 |
| + US ambiguity cap (`amb_s23_addr_2` capped at 1) | 0.974 |
| + stage 4: fine-tuned pair encoder (v1) and stacker | 0.985 |
| + encoder v2 and the 57-feature stacker | 0.985 |
| + dense recall adds; then encoder v3 | 0.986 |
| + France sister-organisation filter (strict, then v2) | 0.987, 0.988 |
| + test-time adaptation of the encoder, round 1 | 0.989 |
| + round 2, France 'groupe' tail-copy and different-street rules | 0.989538 |
| + round 3, France / India / US list rules (steps F-H) | **0.989667** (final) |

Three-decimal scores are the rounded values the portal showed; six decimals where the exact value was
recorded. The training-pool vocabulary tables (v8 bundle) have no leaderboard score of their own; the
v9 bundle, v8 plus stage 4, carries them.

On 100,000 training Source 1 records held out from every model and scored exactly like the test
set (US and India), the final stacker reaches macro F0.5 0.98959 (US 0.98995, India 0.98906); the
v7 matcher alone reached 0.9873 there. The full log is in [docs/SUBMISSIONS.md](docs/SUBMISSIONS.md)
and the method write-up up to stage 4 in [docs/METHODOLOGY.md](docs/METHODOLOGY.md).

## How it works

```
test_source{1,2,3}.tsv                                      11.7M records
  │ normalise: Indic dictionary + transliteration, legal forms, phonetic skeletons, locale rules
  ▼
blocking: IDF cosine over rare name/address keys,                47.2M candidate pairs
          top 10 per source + reverse top 3
  ▼
stages 1-3: LightGBM on 78 pair features, then competition context, then group features
  ▼
stage 4: pairs with probability >= 0.002                          8.08M pairs
         pair encoder (multilingual-e5-small fine-tuned, adapted to the test pool in 3 rounds)
         + LightGBM stacker (57 features, 3 seed members)
  ▼
decision: exclusive owner rule, expected-F0.5 per record;         5.80M matches
          France: empty-share calibration + label-free name/address rules
  ▼
B  dense recall: pretrained e5-small kNN -> 4.16M new pairs,      +21,197
   scored by a pair encoder + LightGBM head
C  France sister-organisation filter                              -652
D  restore generator typo scrambles among C's drops               +396
E  France different-street drops, 'groupe' tail-copy adds         -326 / +3,299
F  France 'cie'/'compagnie' and tail-swap adds, India pool copies +2,073
G  sister swaps at any position, source caps, US state conflicts  -35 / +10
H  France region conflicts, near-tie ownership abstention         -215
  ▼
matching_results.tsv, candidate_pairs.tsv                         5,828,069 matches
```

Country is used for blocking and for the unseen-country rules, never as a model feature. Every
model is trained on the provided training files only; the test pool enters only through label-free
steps (calibration, rules, and test-time adaptation on the pipeline's own confident decisions).

## Quickstart

The code lives in [`code/business_entity_resolution`](code/business_entity_resolution); its
[README](code/business_entity_resolution/README.md) documents every command and setting.

**Environment.** Python 3.12:

```bash
cd code/business_entity_resolution
python3.12 -m venv .venv
.venv/bin/pip install -r requirements.txt -r requirements-dev.txt
.venv/bin/python -m pytest          # 96 tests on generated data (CPU; the GPU is used when present)
```

The 3 tests that run the challenge validator skip unless it sits at
`student_resource/utils/validate_submission.py` next to `code/`.

**Data.** The challenge data comes only from the Amazon ML Challenge 2026 and is not redistributed
here. `DATA` below is the challenge `dataset/` folder (with `train/` and `test/`).

**Model files.** `artifacts/` is not in git, and the final bundle, `artifacts/final/` (1.6 GB), is
not public: `lists.parquet`, `adapt/round3/*.parquet` and `recall/new_pairs_leaderboard.parquet`
hold record ids or record rows of the challenge test data. It is distributed only with our
submission zip (to the challenge organisers), at `code/business_entity_resolution/artifacts/final/`
inside the zip; after unpacking, run the bundle check in the code README. Every prediction command
below needs it. It holds:

| File | What |
|---|---|
| `bundle.json`, `stage{1,2,3}_fold*.txt`, `stage3_robust_fold*.txt`, `vocab_*.parquet`, `native_words.json` | stages 1-3, settings, decision rules |
| `encoder/` | round-3 adapted pair encoder (470 MB) |
| `stage4_m{0,1,2}.txt` | stage-4 stacker, seed members 42, 7, 99 |
| `recall/retriever/` | pretrained `intfloat/multilingual-e5-small` (MIT licence), unchanged |
| `recall/encoder/`, `recall/head.txt`, `recall/head.json` | pair encoder and LightGBM head for the recall pairs |
| `recall/new_pairs_leaderboard.parquet` | the recall new-pair list of the leaderboard run (optional, see Reproducibility) |
| `lists.parquet` | pair lists of post-decision steps E-H (7,797 rows) |
| `adapt/round3/pseudo.parquet`, `anchor.parquet` | the labels round 3 of test-time adaptation learned from (see Reproducibility) |
| `NOTICE` | the MIT licence notice of `intfloat/multilingual-e5-small`, from which `recall/retriever/`, `encoder/` and `recall/encoder/` come |

**Predict** (the leaderboard run):

```bash
.venv/bin/python -m src.cli final --data DATA --model artifacts/final --out ../../output --cache ../../cache
```

`--cache` keeps the prepared records and candidate pairs (keyed on the data, settings and code), so
a second run skips normalisation and blocking. Step B retrieves its new pairs itself (embeddings,
kNN, pair rule). `--recall-pairs artifacts/final/recall/new_pairs_leaderboard.parquet` scores the
leaderboard run's saved list instead, which skips the retrieval (about 41 minutes); the candidate
file is then the one our leaderboard-list run wrote (sha256 `183f2278…`), where the retrieved list
gives `95366357…` (see Reproducibility).

**Train.** Every model has a command; the code README lists them in order. In short:

| Model | Command |
|---|---|
| stages 1-3 | `train` (then `vocab`, `configure` for the predict-time corrections) |
| pair encoder v1 + stacker | `stage4-train --base <multilingual-e5-small>` |
| encoders v2, v3 (continued fine-tuning on fresh records) | `encoder-train` |
| test-time adaptation rounds 1-3 | previous round's decision (`predict` for round 1, saved chain frames after that) → `adapt-labels` → `adapt-train` → `stage4-train` (stacker only) |
| recall head | `recall-train` |
| post-decision lists | `src/rules.py` (code for E, F, G's sister and state rules, H's region rule; see below) |

## Hardware and timings

One laptop: Apple M-series, 24 GB unified memory, macOS; no other machine was tested, so 24 GB is
the tested minimum. LightGBM and polars run on the CPU; the encoders run on the Apple GPU through
PyTorch MPS (CUDA or CPU otherwise, much slower on CPU). The pipeline logs peak memory in GiB
(2^30 bytes); `/usr/bin/time` figures below are in GB (10^9 bytes).

The one full `final` run (28 Sep, with `--cache` from an earlier run, so without normalisation and
blocking, and with `--recall-pairs`, so without retrieval) took 6,329 s (1 h 45 min); maximum
resident set 11.8 GB, macOS peak memory footprint 23.1 GB. Its steps:

| Step | Time | Peak (pipeline log) |
|---|---|---|
| stages 1-3 over 47.2M pairs | 39 min | 10.6 GiB |
| pair encoder over 8.08M active pairs | 40 min | 10.6 GiB |
| stage-4 stacker features and prediction | 1.5 min | |
| recall encoder + head over 4.16M new pairs | 23 min | |
| post-decision chain B-H | 21 s | |

Not in that run, measured separately:

- Retrieval, which the default command runs: 11.7M records embedded at 7,225 records/s on average,
  41 min with the kNN (2,453 s), peak 13.3 GiB (14.2 GB resident). This was the retrieval code run
  from a script, not through `final`; the peak of a whole default run has not been measured.
- Normalisation and blocking (a first run, or a run without `--cache`): not measured on the final
  code; about 50 minutes per split on an earlier version ([docs/METHODOLOGY.md](docs/METHODOLOGY.md)).

Disk: the bundle 1.6 GB; `test_scores/`, written into the bundle by every run, about 1.5 GB;
`embeddings.npy` 9 GB during retrieval (deleted after the kNN); the test candidate cache 1.4 GB.

## Reproducibility

- **Byte-identical output: what was measured.** One `final` run end to end, with `--cache` and
  `--recall-pairs artifacts/final/recall/new_pairs_leaderboard.parquet`, on the code as of 28 Sep
  00:58 (before commit 53ad459; later commits changed the code), wrote the uploaded leaderboard
  file (sha256 `0a76fb8fa921d23d8c3ec208849d740e5d1a531c573185229b1c7156b88c978c`) and passed the
  official validator. The default command above, with retrieval, has not yet been run end to end in
  one piece: its retrieval was run on its own, and steps B-H on the leaderboard run's saved stage-4
  scores with the retrieved list give the same sha256 (see "Recall pair list"; rechecked on the
  current code). Each stage was also checked against the files of the leaderboard run: stacker
  probabilities of all 8.08M active pairs, recall head probabilities, every step of the
  post-decision chain, the round-3 pseudo-labels, the retriever embeddings of a 200,000-record
  chunk, one kNN list (France Source 3), and eight of the list rules on the decisions they were
  derived on.
- **Apple GPU (MPS).** Encoder and retriever inference is deterministic run to run on the same
  machine with the same batch settings (checked bit for bit). CUDA or CPU arithmetic differs slightly
  and can move a few decisions. Whether GPU training is bit-reproducible was not measured; the
  byte-identical path uses the saved round-3 weights.
- **Recall pair list.** The retrieval rule caps reverse-only pairs at 10 per record by cosine; float16
  cosines tie often, and the leaderboard run ranked tied cosines in the row order of a join, which
  is not fixed (the same ranking without a tie-break gave different lists on two runs of this code).
  The code breaks ties by source and record, so its list is deterministic; it differs from the
  leaderboard run's list in 368 of 4,156,489 pairs (367 records, each a swap between pairs of equal
  cosine at the cap). None of them is added, so `matching_results.tsv` is still byte-identical
  (measured: steps B-H on the leaderboard run's stage-4 scores with the list the code retrieved);
  `candidate_pairs.tsv` differs in those 368 ids (sha256 `95366357…`). `--recall-pairs` scores the
  leaderboard list instead; the candidate file is then the one our leaderboard-list run wrote (sha256
  `183f2278…`). No candidate file was uploaded, so there is no leaderboard candidate file to match.
- **Rule lists.** The leaderboard run derived each list of steps E-H on an earlier decision of the
  same run and applied the fixed list later, so the lists ship as data (`lists.parquet`).
  `src/rules.py` reproduces the France different-street drops, 'groupe' tail-copy adds,
  'cie'/'compagnie' and tail-word swap adds, sister swaps at any position and region-conflict
  drops, the India pool-copy list and the US state-conflict list exactly on those decisions.
  Source-cap drops, France no-legal-form groupe adds and the near-tie ownership list ship as data
  only: their builders read intermediate tables of the leaderboard run, and the near-tie
  list has no saved builder at all.

## Limitations

- Steps E-H replay fixed pair lists of this test pool (`lists.parquet`). On other data they match
  nothing; there only the rules with code apply (`final --derive-lists`), and three list types
  (source-cap drops, France no-legal-form 'groupe' adds, near-tie abstentions) have no code.
- The France rules and word lists were tuned without labels on this test pool, the only France
  data there is; nothing measured how they carry to other data.
- Retraining may not be bit-for-bit: GPU training reproducibility was not measured, and the code
  README lists where the training commands differ from the runs that made the shipped weights.
- Only one machine was used (Apple M-series, 24 GB); other hardware was not tried.

## Repository layout

```
README.md                         this file
LICENSE                           MIT
code/business_entity_resolution/  the pipeline: src/, tests/, requirements, README
docs/METHODOLOGY.md               method write-up up to stage 4 (v9); later steps are in the READMEs
docs/SUBMISSIONS.md               version and leaderboard log
archive/                          archived experiment scripts (not runnable, kept for history)
```

## Team

Team MEOW MEOW: Aditya Rohilla, Priyanshu Ghosh.

## Licence

The code is released under the MIT licence ([LICENSE](LICENSE)). The pretrained text model,
`intfloat/multilingual-e5-small` (revision 614241f622f53c4eeff9890bdc4f31cfecc418b3), is published
under the MIT licence by its authors; our fine-tuned encoders are derived from it, and the bundle
carries its licence notice (`NOTICE`). LightGBM is MIT licensed. The challenge data belongs to the organisers and is not included.
