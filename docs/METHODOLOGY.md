# Business Entity Resolution: Method up to Stage 4 (Amazon ML Challenge 2026)

**Team Name:** MEOW MEOW  
**Team Members:** Aditya Rohilla, Priyanshu Ghosh  
**Written:** 25 September 2026, for the v9 bundle

This write-up covers the pipeline up to stage 4 (the v9 bundle, an earlier submission). The
final submission (0.989667) adds dense recall, test-time adaptation of the encoder and the
post-decision rules of steps B-H; those are described in the repository README and the code
README. Where this document says "submitted", it means the v9 submission.

---

## 1. Executive Summary

A retrieve-then-rank pipeline:

1. IDF-weighted retrieval over rare token *combinations* for blocking.
2. A three-stage LightGBM matcher.
3. Stage 4: a fine-tuned multilingual pair encoder (intfloat/multilingual-e5-small,
   MIT licence) reads the raw text of the likely pairs, and a LightGBM stacker
   combines its score with the matcher's probability.
4. A per-record rule that maximises expected F0.5.

Blocking finds 98.7% of true pairs at 25 candidates per Source 1 record, a
perfect-matcher ceiling of 0.996. On 100k Source 1 records held out from all
training and scored exactly like the test set, the matcher reaches a macro
F0.5 of **0.9873**. With the production decision rules, stage 4 raises the
same holdout from 0.9870 to **0.9888** (+0.0018, 95% CI +0.0016 to +0.0020).

The main contributions are:

- Phonetic skeletons plus an Indic-to-English word dictionary learned from the
  training pairs, so that transliterated names line up with their English
  forms.
- Token-alignment and ambiguity features that separate decoy businesses from
  noisy true copies.
- Label-free adaptation for the unseen country (France) and for the test
  set's denser decoys.

---

## 2. Methodology

### 2.1 Problem Analysis

Findings from the full training set (2.2M Source 1, 5.0M Source 2, 5.3M Source 3 records):

- **Structure:**
  - No Source 2/3 record matches more than one Source 1 record, and no true
    match crosses countries.
  - 5.6% of Source 1 records are singletons; the rest have 1–11 matches
    (mostly 3–4), split across Source 2 and Source 3.
  - 26% of Source 2/3 records match nothing. These are decoys: near-copies of
    a real business with a changed house number or name word.
- **Names** come from a small vocabulary, so single words ("Quality", "Prime")
  are shared by tens of thousands of businesses. Noise seen:
  - typos, abbreviations (Pvt/Ltd/Corp) and dotted forms (S.A.R.L., L.L.C.)
  - legal-suffix changes, swapped word order, appended words ("Center")
  - prefixes (M/s, Shri, `***`), "formerly X"
  - website-style names (`qualityaceculture.com`)
- **Scripts:** 23% of India Source 2 names and 13% of India Source 3 names are
  in Devanagari, Bengali, Kannada, Malayalam or other Indic scripts. Some
  addresses carry state names in Devanagari. Before the dictionary below,
  these pairs caused 41% of India's blocking misses.
- **Addresses:**
  - reordered components and abbreviations (Rd/St/Cir)
  - state name vs code (Arizona/AZ, Uttar Pradesh/UP, Kerala/Keralam)
  - placeholders (`<NULL>`, `N/A`, `None`) and extra prefixes (H.no, Plot, Door No)
  - no usable address in about 4% of true pairs; these are the hardest remaining cases
- **Test set:**
  - 1.73M Source 1 records, 15% in France, which never appears in training.
  - France Source 1 addresses always end in a long region phrase ("…, Bordeaux,
    Nouvelle-Aquitaine"). France Source 2/3 records carry the region, a
    department ("Gironde") or neither.
  - 40% of test Source 2/3 records are unowned, against 26% in training. In
    the US these decoys often come as small groups of copies sharing a nearby
    house number.

### 2.2 Solution Strategy

**Approach Type:** Blocking + three-stage LightGBM classifier + fine-tuned pair encoder with a
LightGBM stacker (stage 4) + expected-F0.5 decision rule.

**Core Innovation:**

- Rare-combination key retrieval built on phonetic skeletons and a learned
  transliteration dictionary.
- Pair features that align leftover tokens and numbers and count
  name/address ambiguity in the pool.
- Label-free test-time adaptation: locale normalisation, empty-share
  calibration, and a per-country choice of the group-feature stage.
- Predict-time corrections for the test pool's shift, without retraining
  (section 4): an ambiguity-count cap, training-pool token frequencies, and
  post-decision rules for the country absent from training.
- A cross-encoder over the raw record text as a second opinion on the likely
  pairs, stacked with the matcher by a small LightGBM (section 4, stage 4).

Only the provided data is used: no external databases, geocoding or APIs. The
dictionary and locale rules are learned from the provided files. The matcher
and the stage-4 stacker are LightGBM (MIT licence). The stage-4 pair encoder is
the pretrained `intfloat/multilingual-e5-small` (MIT licence, 117.65M
parameters with the classification head), fine-tuned on the training files
only and read offline from the submitted bundle. Country restricts the search
and selects data-driven normalisation, but it is never a model feature.

---

## 3. Candidate Generation (Blocking)

**Normalisation:**

- Indic words are first translated with a dictionary learned from training
  pairs. Indic-script Source 2/3 names are word-by-word transliterations of
  their Source 1 name, so aligning the words in order over 551k pairs gives
  1,362 entries such as अर्बन→urban. Indic address runs map to the state
  voted by the matched Source 1 address.
- Remaining non-Latin text is transliterated to ASCII.
- Accents and placeholders are removed, `&` becomes "and", split digit runs
  are re-joined ("1-59"→"159") and ordinals dropped ("14th"→"14").
- Runs of single letters are joined ("s a r l"→"sarl"), state names map to
  codes, and abbreviations are expanded.
- A *core name* drops legal forms and honorifics, also matched by skeleton,
  so transliterated "praivet" is dropped like "private".
- Each token gets a phonetic **skeleton**: c/q→k, z→s, w→v, ph→f, then h and
  vowels are dropped and repeated consonants collapsed. A second,
  transliteration-aware skeleton also merges soft c/g, b/v, m/n, j/s and d/t.

**Locale normalisation (label-free).** For every country we compare how often
each address word appears in Source 1 against Source 2/3. Words that are
strongly source-asymmetric are dropped from that country's addresses on both
sides, together with the articles joining them. On the test set this finds
exactly France's region and department words and the "N°" markers. On the
US/India training data it finds nothing, so the trained models see unchanged
inputs.

**Blocking keys used**, each hashed to 64 bits:

| Key family | Example |
|---|---|
| core-name token skeleton, and its transliteration-aware variant | `trstd` |
| unordered pair of core-name tokens (both skeletons) | `trstd\|sftvr` |
| whole core name without spaces (both skeletons), plus its first/last 6 letters | matches `trustedsoftware.com` |
| address number (leading zeros removed, glued ranges split) | `18220` |
| address word skeleton: the record's 5 *rarest* words, not its first 5 | `sprs` |
| unordered pair of address words | `sprs\|ln` |
| number × name token, number × address word, name token × address word | `18220\|trstd` |

Before key building, digit look-alikes inside name words are mapped back to
letters ("pi1ates"→"pilates").

**Search:**

- Keys are weighted by IDF within each country and source block. Keys seen
  once, or more than 3,000 times, are dropped.
- Each record becomes an L2-normalised sparse vector.
- For every Source 1 record we keep the top 10 Source 2 and top 10 Source 3
  records by cosine, using chunked sparse matrix products in parallel threads.
- A reverse pass adds the top 3 Source 1 records for every Source 2/3 record.
- Search never crosses countries.

- **Candidate pairs generated:** 55.8M on train (25.3 per Source 1 record);
  47.2M on test (27.3 per Source 1 record).
- **How true matches were kept:**
  - Combination keys stay rare even when their parts are common.
  - Skeletons and the learned dictionary absorb spelling, transliteration and
    typo noise.
  - Independent key families (name-only, address-only, mixed) keep a record
    findable when one field is corrupted or empty.
  - The reverse pass rescues pairs ranked low from the Source 1 side.

  Measured on the full training set: **pair recall 0.9867** (US 0.9906, India
  0.9808); **F0.5 ceiling 0.9960**.

---

## 4. Matching Model

**Features used:** 78 in stage 1, 87 in stage 2, 47 in stage 3.

- **Name features:**
  - rapidfuzz ratio, partial, token-sort and token-set ratios, and
    Jaro-Winkler, on the name and the core name
  - skeleton similarity, space-free name similarity (website names),
    acronym match
  - letter-bag cosine and character-trigram TF-IDF cosine
- **Token alignment** (names and addresses separately):
  - After exact matches are removed, the number of leftover tokens on each side.
  - The best edit similarity of each leftover to the other side's leftovers:
    a typo scores high, a substituted word low.
  - The leftovers' corpus frequency and a "common word substituted" flag.
  - These separate decoys ("Emerald Financial" vs "Emerald Industries") from
    noisy true copies.
- **Address features:**
  - fuzzy ratios and address-word Jaccard
  - number alignment over all address numbers: coverage, smallest numeric gap,
    smallest edit distance, prefix relation
  - house-number equality, equality after removing zeros, edit similarity,
    log numeric gap and containment (665 inside 6654)
- **Ambiguity:** how many Source 1 and Source 2/3 records in the pool share
  this core name, skeleton or address. Name-only evidence is weak when six
  businesses carry the name.
- **Context:**
  - the retrieval cosine
  - rank and gap to the best candidate among the Source 1 record's candidates,
    and among the Source 1 records competing for the same Source 2/3 record
  - candidate counts; source and script flags
- **Stage 2** adds the stage-1 probability and how it compares with rivals on
  both sides: rank, gap, sums, and the margin over the runner-up owner of the
  Source 2/3 record.
- **Stage 3** compares a candidate with the other likely members of its
  Source 1 record's group. It uses the probability-weighted similarity of
  name, address and house number to the anchors, agreement across sources,
  and whether the candidate's deviations from Source 1 are shared by the
  anchors.

**Model type:** LightGBM gradient-boosted trees, with binary log-loss and early stopping.

- Stages 1–2: 127 leaves, learning rate 0.08.
- Stage 3: 63 leaves, learning rate 0.05, trained on pairs with stage-2
  probability ≥ 0.001.
- Training uses 500k sampled Source 1 records; their candidates come from the
  full pool.
- 4-fold cross-validation, grouped by Source 1 record, with one fold
  assignment shared by all stages.
- Stage-1 probabilities for all 55.8M training pairs (out-of-fold for the
  sample, fold-model average otherwise) feed stage 2.
- Test predictions are the mean of the fold models.

**Stage-3 choice by country:**

- Stage 3 adds +0.0007 on the holdout, but it lets siblings vouch for each
  other.
- On the test set, US decoy businesses arrive as copy groups sharing a nearby
  house number: all-unowned twin groups are 18 times the training rate. There,
  stage 3 adds 0.07 extra house-variant matches per record over stage 2, which
  are very likely decoys.
- India's test profile matches training, so India is decided on stage 3. The
  US and the unseen France are decided on stage 2.

**Threshold selection method:**

- Each Source 2/3 record is first given to its most likely Source 1 record
  only, as in the ground truth.
- A per-record rule keeps the top-k candidates maximising expected F0.5,
  E[F] ≈ 1.25·Σᵢ≤ₖ pᵢ / (0.25·Σp + k), and keeps nothing when Π(1 − p) is
  larger.
- It was chosen over 91 global thresholds on out-of-fold predictions with the
  exact metric, singletons included.
- For France only, the "keep nothing" term is raised (at most 4×) until
  France's share of empty predictions matches the seen countries in the same
  run (5.2%→5.6%). This uses no labels. The submitted bundle pins this target
  (see below).

**Test-pool corrections (v8, submitted; the v7 models are not retrained):**

- **Ambiguity cap.** `amb_s23_addr_2` counts the Source 2/3 records in the
  pool that share the candidate's address. In training a count of 2 or more
  means copies of the same business; the test pool's decoy copy groups inflate
  it into false copy-consensus. Prediction clips it at 1 (`feature_caps`),
  which restores its training-time meaning. The US leaderboard confirmed it
  (+0.003).
- **Training-pool token frequencies.** The leftover-token features read each
  token's log document frequency, log(df/n), in the pool being scored. Stage 1
  uses these values almost as token identities, and the test pool's vocabulary
  differs (decoy words such as "group", "holdings", "partners" are more
  frequent), so readings flip. For every country with a training pool,
  prediction looks the frequencies up in tables built from the training
  records (`vocab_*.parquet` in the bundle; identity on training data). The
  mirror experiment, training records read with test-pool tables, loses 0.0037
  F0.5. A country without a training pool keeps its own test-pool tables.
- **Pinned calibration target.** The corrections above raise the seen
  countries' empty share, which would push France's "keep nothing" weight to
  its 4× cap. The target is pinned at 0.0560 (`unseen_empty_target`), the
  seen countries' share with the v7 probabilities, so France is calibrated
  exactly as before (weight 3.49).
- **Post-decision rules for a country absent from training**
  (`src/unseen_rules.py`, label-free, using names, house numbers and streets
  only):
  - legal-form conflict drop: a kept pair at a nearby house number whose name
    carries a different legal form (sarl vs sas) is a sister company when the
    Source 1 record also keeps a pair at its own house number (785 pairs);
  - true-variant word add: a candidate whose core name is the Source 1 core
    plus "fils", "associes", "cie" or "services", at the same house number
    (2,178 pairs);
  - generator-tail add: the Source 1 name with its last word replaced by, or
    followed by, a tail the generator uses for true copies ("france",
    "et associes", "groupe"), at the same house number and street (7,678
    pairs).

  Adds keep the exclusive owner rule. The word lists were measured without
  labels on this test set's unseen country, and country names never appear in
  the logic.

**Stage 4: pair encoder and stacker (v9, submitted).**

- **Pair encoder.** Base checkpoint `intfloat/multilingual-e5-small` (MIT
  licence, Hugging Face revision 614241f622f53c4eeff9890bdc4f31cfecc418b3,
  117.65M parameters), fine-tuned as a cross-encoder with one match logit.
  Each record is its raw text, lowercased: `business_name | business_address`,
  accents, punctuation and legal forms kept. The pair is the two texts,
  truncated to 128 tokens.
- **Encoder training data** (training files only): 360,000 training Source 1
  records drawn outside the matcher's 500k sample and 100k holdout (20,000 kept
  apart for validation), each with its top 6 candidates by retrieval score plus
  every true pair among its candidates. Fit pairs whose Source 2/3 record
  appears in any active pair of the matcher's sample or holdout are dropped, so
  the encoder never saw the candidates of the pairs the stacker is trained or
  measured on. That leaves 1,817,581 fit pairs (1,145,172 positive).
- **Fine-tuning:** one pass, batch 128, binary cross-entropy, AdamW (learning
  rate 3e-5, weight decay 0.01), 5% linear warmup then linear decay, gradient
  clipping at 1.0, word embeddings frozen, bfloat16 autocast, batches of
  similar token length. 72 minutes on the laptop's Apple GPU (PyTorch MPS).
  Validation AUC 0.9979, against 0.9192 for the retrieval score alone.
- **Stacker:** rescores only the *active* pairs, whose decided probability is
  at least 0.002 (99.98% of the true pairs among the candidates). Eight
  features: the logit of the decided probability; the encoder logit; an
  empty-candidate-address flag; for both scores, the rank and the gap to the
  best other active candidate of the same Source 1 record; and the number of
  active candidates of that record. The flag matters because the pair filter
  above removed most negatives of empty-address candidates (they sit in many
  pools), so the encoder is over-confident there and the stacker learns by how
  much. LightGBM with the matcher's parameters, trained on the 2.32M active
  pairs of the matcher's sample with their out-of-fold probabilities and the
  encoder's logits, 5-fold cross-validation grouped by Source 1 record, and the
  decision rule chosen on the out-of-fold stage-4 probability (expected F0.5,
  missed-match term 0.05); final model refit on all active pairs (119 trees).
- **Holdout** (the 100k Source 1 records, used by neither the encoder nor the
  stacker): 0.98702 → 0.98881 macro F0.5 with the production decision (US
  0.98722 → 0.98920, India 0.98673 → 0.98824); among the active pairs, false
  positives 726 → 488 and missed true pairs 7,266 → 6,311. Trained on one
  country and scored on the other, the gain holds (US → India +0.0014).
- **Test:** 8.08M active test pairs, about 40 minutes on the Apple GPU at
  batch 1024 with bfloat16 autocast; stages 1–3 stay on the CPU. The
  fine-tuned weights ship in float32 (470 MB): stored in 16 bits they changed
  about a fifth of the logits.
- **Scope and France.** The setting `stage4_scope` chooses the countries
  decided on stage 4: `seen` (countries with training labels) or `all`. The
  encoder never saw France, and no label can measure it there (its agreement
  with the matcher's keeps is 96.1% in France against 99.1–99.4% in the US and
  India). Under `seen`, France is decided exactly as in v8 (stage-2
  probabilities, the pinned calibration, the unseen-country rules). Under
  `all`, France gets the stage-4 probabilities, then the same calibration (its
  "keep nothing" weight reaches the 4× cap) and the same unseen-country rules.
  **The v9 bundle uses `stage4_scope = seen`; the final bundle uses `all`.**

---

## 5. Results & Error Analysis

- **F_0.5 Score (macro):**
  - Honest holdout (100k Source 1 records used by no stage, scored like the
    test set): **0.9873** (India 0.9866, US 0.9878).
  - Out-of-fold by stage: 0.9846 / 0.9861 / 0.9869.
  - Blocking ceiling: 0.9960.

| Version | Change | Holdout / OOF F0.5 |
|---|---|---|
| v1 | 1 stage, 100k records | 0.960 (OOF) |
| v2 | 2 stages, 300k records | 0.969 (OOF) |
| v3 | house-number similarity features | 0.974 (OOF), 0.975 (holdout) |
| v5 | Indic dictionary, rarest-word keys, token-alignment and ambiguity features, stage 3, 500k records | 0.9851 (holdout) |
| v7 | transliteration-aware and address-pair keys, locale normalisation, per-country stage-3 choice | 0.9873 (holdout) |
| v8 | v7 models + test-pool corrections (ambiguity cap, training-pool token frequencies, unseen-country rules) | not measured on the holdout (the corrections target the test pool) |
| v9 | v8 + stage 4 (fine-tuned multilingual-e5-small pair encoder + LightGBM stacker) | 0.98881 vs 0.98702 for the production decision (holdout) |

- **Common false positives (wrong merges):**
  - decoys with the same name and street but a nearby house number
  - the same business name at another branch
  - singletons matched to a decoy (1.6% of singletons)
  - Precision on the out-of-fold sample is above 0.99.
- **Common false negatives (missed matches):**
  - Candidates with an empty address whose name is shared by several Source 1
    records (chains). The model is calibrated there and the text cannot say
    which branch owns the record.
  - Names replaced by an unrelated word with the same address.
  - Heavily typo'd names.
  - True pairs never retrieved (1.3%).

---

## 6. Conclusion

Treating blocking as weighted retrieval over rare token combinations made
12.5M records tractable on one laptop at 98.7% recall. Most of the remaining
gain came from error analysis:

- decoys differing only in a house number
- transliterated names
- substituted name words

On the test side, two shifts could only be seen by comparing test statistics
with training: an unseen address format, and decoys arriving in copy groups.
Label-free checks of those statistics decided how the model is applied per
country.

---

## Appendix

### A. Code Artefacts

`code/business_entity_resolution/`:

| Path | Role |
|---|---|
| `src/cli.py` | Entry point: `run` (train + predict), `train`, `predict`, `block` (blocking recall only), `decide` (re-apply the decision rule to saved scores), `vocab` (add training-pool token tables to a bundle), `configure` (change a saved setting), `stage4-train` (fine-tune the encoder and train the stacker into a bundle), `stage4-import` (add an already trained encoder and stacker to a bundle); the commands of the final run (`final`, `check`, `post-import`, `encoder-train`, `recall-train`, `adapt-labels`, `adapt-train`) are in the code README |
| `src/data.py` | TSV loading with schema checks; submission-rule checks; atomic writes |
| `src/text.py` | Normalisation, transliteration, state codes, skeletons |
| `src/indic.py` | Indic-to-English word dictionary learned from training pairs |
| `src/locale_norm.py` | Label-free source-asymmetric address-word removal |
| `src/candidates.py` | Key generation and parallel sparse retrieval |
| `src/features.py`, `src/alignment.py` | Record preparation; pair, alignment, ambiguity and competition features |
| `src/groupfeat.py` | Stage-3 group features |
| `src/pipeline.py` | Training (grouped cross-validation, three stages, rule selection, honest holdout), prediction, decision, model bundle |
| `src/unseen_rules.py` | Post-decision rules for a country absent from training |
| `src/pair_encoder.py` | Stage-4 pair text, encoder fine-tuning and inference (PyTorch, in a separate process) |
| `src/stage4.py` | Stage-4 features, stacker scoring, scope routing, encoder training-pair selection |
| `src/metric.py` | Macro F0.5, reference and vectorised forms |
| `src/config.py` | All settings; saved inside the model bundle |
| `tests/` | synthetic end-to-end runs (stage 4 with a tiny encoder on the CPU) checked with the challenge validator; the code README gives the current count |

To reproduce both output files:

```bash
cd code/business_entity_resolution
python3.12 -m venv .venv && .venv/bin/pip install -r requirements.txt
.venv/bin/python -m src.cli run --data <dataset dir> --model artifacts/model \
    --out ../../output --sample 500000 --folds 4
```

On a 24 GB laptop (Apple M-series), blocking takes about 50 minutes per
split, training about 1 hour and test scoring about 50 minutes, all on the
CPU. Stage 4 runs on the Apple GPU (PyTorch MPS; CUDA or, much slower, the CPU
elsewhere): fine-tuning about 72 minutes, test encoding about 40 minutes.
Peak memory is under 21 GB. Pass `--cache DIR` to reuse the candidate tables
between runs. No network is used at run time.

The v9 submission came from the v9 bundle: the v8 bundle (the v7 bundle plus
the training-pool token tables and the corrected settings) plus stage 4. With
the v7 bundle in `artifacts/v7`:

```bash
cp -R artifacts/v7 artifacts/v8
.venv/bin/python -m src.cli vocab --data <dataset dir> --model artifacts/v8
.venv/bin/python -m src.cli configure --model artifacts/v8 \
    --set 'feature_caps={"amb_s23_addr_2": 1.0}' --set 'feature_cap_countries=[]' \
    --set 'stage3_use="none"' --set 'stage3_by_country={"india": "full"}' \
    --set 'unseen_empty_target=0.05597295153414383' --set 'unseen_rules=true'
mkdir artifacts/v9 && cp artifacts/v8/*.* artifacts/v9/
.venv/bin/python -m src.cli stage4-import --model artifacts/v9 --encoder <fine-tuned encoder dir> \
    --stacker stage4_A.txt --stacker-meta stage4_A.json
.venv/bin/python -m src.cli configure --model artifacts/v9 --set 'stage4_scope="seen"'
.venv/bin/python -m src.cli predict --data <dataset dir> --model artifacts/v9 --out ../../output
```

`stage4-train --data <dataset dir> --model artifacts/v9 --base <base checkpoint>`
retrains stage 4 instead of importing it; the base checkpoint is downloaded
once beforehand (`hf download intfloat/multilingual-e5-small --revision
614241f622f53c4eeff9890bdc4f31cfecc418b3`). The v9 bundle is not distributed;
the final bundle (`artifacts/final`, see the repository README) is, and its
leaderboard run is `final`.

### B. Additional Results

- **Blocking per country (train):** US recall 0.9906, ceiling 0.9973; India
  recall 0.9808, ceiling 0.9941.
- **Leave-one-country-out (stage 1, earlier version):** trained on India and
  scored on US 0.944; trained on US and scored on India 0.864.
- **Test predictions:**
  - Share of Source 1 records with at least one match: US 94.4%, India 94.1%,
    France 94.4%.
  - Matches per matched record: 3.5–3.6, consistent with the 3.46 true
    matches per Source 1 record in training.
- **Measured and not adopted:**
  - exact F-measure maximisers and dependent-label rules (±0.00003)
  - label-shift prior correction (EM, BBSE): negative on an unseen country
  - transitive or twin propagation over Source 2/3 (−0.006 to −0.018)
  - bipartite assignment (the one-sided form equals our owner rule)
  - a larger candidate count alone (+0.0003, not significant)
