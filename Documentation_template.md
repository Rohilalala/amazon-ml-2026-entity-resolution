# ML Challenge 2026: Business Entity Resolution Solution Template

**Team Name:** [Your Team Name]  
**Team Members:** [List all team members]  
**Submission Date:** 25 September 2026

---

## 1. Executive Summary

A three-step pipeline: rare-key retrieval blocking, a two-stage LightGBM
pair classifier, and a per-record decision rule that maximises expected
F0.5. Blocking recovers 97.0% of true pairs at 25 candidates per Source 1
record. The matcher reaches a macro F0.5 of 0.974 on out-of-fold validation,
against a blocking ceiling of 0.990. The main ideas are blocking on
*combinations* of tokens (single words are shared by thousands of
businesses), a phonetic skeleton that bridges Indic-script transliterations
and English spellings, and a second stage that sees how each pair's
probability compares with its competitors.

---

## 2. Methodology

### 2.1 Problem Analysis

Findings from the full training set (2.2M Source 1, 5.0M Source 2, 5.3M
Source 3 records):

- **Structure.** No Source 2/3 record is matched to more than one Source 1
  record, and no true match crosses countries. 5.6% of Source 1 records are
  singletons; the rest have 1–11 matches (mostly 3–4), split across Source 2
  and Source 3. 26% of Source 2/3 records match nothing (decoys).
- **Names** come from a limited vocabulary, so single words ("Quality",
  "Prime", "Global") are shared by tens of thousands of businesses. Noise:
  typos, abbreviations (Pvt, Ltd, Corp), legal-suffix changes, swapped word
  order, appended words ("Center", "Services"), prefixes (M/s, Shri, `***`,
  `--`), "formerly X", and website-style names (`qualityaceculture.com`).
- **Scripts.** About 10% of Indian Source 2/3 names are in Devanagari,
  Malayalam, Kannada or Bengali script, and some addresses contain state
  names in Devanagari (उत्तर प्रदेश).
- **Addresses:** reordered components, abbreviations (Rd/St/Cir), state
  name vs code (Arizona/AZ, Uttar Pradesh/UP, Kerala/Keralam), placeholders
  (`<NULL>`, `N/A`, `None`), extra prefixes (H.no, Plot, Door No), PO boxes,
  and 12% of candidate records with no usable address at all.
- **Decoys** copy a real business's name and street with a nearby house
  number (228 vs 249, 6654 vs 665). These were the largest source of wrong
  merges before house-number features were added.
- **Test set:** 1.73M Source 1 records, 15% of them in France, which never
  appears in training.

### 2.2 Solution Strategy

**Approach Type:** Blocking + two-stage classifier + expected-F0.5 decision rule  
**Core Innovation:** IDF-weighted retrieval over rare *combination* keys
built from phonetic skeletons, plus a second-stage model on competition
features (how a pair ranks against the other candidates on both sides).

Only the provided data is used. There are no external lookups, geocoding or
APIs. The model is LightGBM (MIT licence). Country is used only to restrict
the search and is never a model feature, so the model applies unchanged to
France.

---

## 3. Candidate Generation (Blocking)

**Normalisation.** Non-Latin scripts are transliterated to ASCII (anyascii),
accents and placeholders are removed, `&` becomes "and", digit runs split by
noise are re-joined ("1-59" → "159"), ordinals are dropped ("14th" → "14"),
state names are mapped to codes, and abbreviations are expanded. A *core
name* drops legal forms and honorifics. Every token gets a phonetic
**skeleton**: c/q→k, z→s, w→v, ph→f, then h and vowels are dropped and
repeated consonants collapsed. The skeleton equates transliterations with
English spellings ("helthkeyr"/"healthcare" → "ltkr", "praivet"/"private" →
"prvt").

**Blocking keys used**, each hashed to 64 bits:

| Key | Example |
|---|---|
| core-name token skeleton | `trstd` |
| unordered pair of core-name tokens | `trstd|sftvr` |
| address number | `18220` |
| address word skeleton | `sprs` |
| number × name token | `18220|trstd` |
| number × address word | `1731|sprs` |
| name token × address word | `trstd|fnks` |
| whole core name without spaces | matches `trustedsoftware.com` |

Keys are weighted by IDF within the country. Keys seen once, or more than
1,000 times, are dropped. Each record becomes an L2-normalised sparse vector.
For every Source 1 record we keep the top 10 Source 2 and top 10 Source 3
records by cosine, using chunked sparse matrix products run in parallel
threads. A reverse pass adds the top 3 Source 1 records for every Source 2/3
record. Search stays within one country.

- **Candidate pairs generated:** 55.5M on train (25.2 per Source 1 record);
  47.1M on test (27.2 per Source 1 record).
- **How true matches were kept:** combination keys stay rare even when their
  parts are common; the skeleton absorbs spelling, transliteration and typo
  noise; several independent key families (name-only, address-only, mixed)
  mean a record still matches when one field is corrupted or missing; and
  the reverse pass rescues pairs that rank low from the Source 1 side.
  Measured on the full training set: **pair recall 0.970** (US 0.980, India
  0.955), and an **F0.5 ceiling of 0.990** for a perfect matcher on these
  candidates.

---

## 4. Matching Model

**Features used** (43 in stage 1, 52 in stage 2):

- **Name features:** rapidfuzz ratio, partial ratio, token-sort and
  token-set ratios, and Jaro-Winkler on the normalised name and on the core
  name; ratio and token-set on the skeletons; similarity of the space-free
  names (website names); acronym match; core-token Jaccard; name lengths.
- **Address features:** ratio, partial, token-sort and token-set ratios;
  address-word Jaccard; number Jaccard and shared-number count; house-number
  equality, equality after stripping leading zeros, edit-distance similarity,
  log absolute difference, and containment (665 inside 6654); address
  lengths and number counts.
- **Other:** retrieval cosine; rank and gap to the best candidate among the
  Source 1 record's candidates (overall and per source) and among the Source 1
  records competing for the same Source 2/3 record; candidate counts on both
  sides; source flag; non-Latin script flags for name and address.
- **Stage-2 features:** the stage-1 probability, its rank and gap on both
  sides, the probability sum per Source 1 record and per Source 2/3 record,
  the count of candidates above 0.5, and the margin over the runner-up owner
  of the Source 2/3 record.

**Model type:** LightGBM gradient-boosted trees (binary log-loss, 127
leaves, learning rate 0.08, early stopping). Stage 1 is trained on 300k
sampled Source 1 records, whose candidates come from the full pool, with
4-fold cross-validation grouped by Source 1 record. Stage-1 probabilities for
all 55.5M training pairs (out-of-fold for sampled records, fold-model average
for the rest) feed the stage-2 features. Stage 2 is trained the same way.
Test predictions are the mean of the fold models.

**Threshold selection method:** each Source 2/3 record is first assigned to
its most likely Source 1 record only, as in the ground truth. Then two rules
are scored on out-of-fold predictions with the exact macro F0.5 metric,
singletons included: a global probability threshold (0.05–0.95), and a
per-record rule that keeps the top-k candidates maximising expected F0.5,
E[F] ≈ 1.25·Σᵢ≤ₖ pᵢ / (0.25·Σp + k), with "keep nothing" chosen when
Π(1 − p) is larger. The expected-F0.5 rule scored best and is used.

---

## 5. Results & Error Analysis

- **F_0.5 Score (macro):** 0.9739 out-of-fold (US 0.979, India 0.966), with
  a blocking ceiling of 0.990.

| Version | Change | OOF F0.5 |
|---|---|---|
| v1 | 1 stage, 100k training records | 0.960 |
| v2 | 2 stages, 300k training records | 0.969 |
| v3 | + house-number similarity features | 0.974 |

- **Unseen-country estimate (stage 1, v2):** trained on India and scored on
  US: 0.944 (in-country 0.966). Trained on US and scored on India: 0.864.
  France, being Latin-script, should behave like the milder case.
- **Common false positives (wrong merges):** decoys with the same name and
  street but a different house number (reduced by the house-number
  features), and names with an appended word ("Moore Holdings Holdings").
  Precision on the sample is 0.992.
- **Common false negatives (missed matches):** candidates with an empty
  address, where only a common name is left (37% of in-candidate misses);
  heavily typo'd names; Bengali or Kannada transliterations whose skeletons
  drift from the English spelling; and true pairs never retrieved by blocking
  (3%).

---

## 6. Conclusion

Treating blocking as weighted retrieval over rare token combinations made
the problem tractable at 12.5M records on one laptop without losing recall.
A two-stage matcher that reasons about competing candidates, plus a decision
rule aligned with the metric, gave most of the precision. The largest single
gain came from error analysis: decoys that differ only in the house number.

---

## Appendix

### A. Code Artefacts

`code/business_entity_resolution/`:

| Path | Role |
|---|---|
| `src/cli.py` | Entry point: `train`, `predict`, `run` |
| `src/data.py` | TSV loading with schema checks; submission-rule checks; atomic writes |
| `src/text.py` | Normalisation, transliteration, state codes, skeletons |
| `src/candidates.py` | Key generation and parallel sparse retrieval |
| `src/features.py` | Record preparation, pair and competition features |
| `src/pipeline.py` | Training (cross-validation, two stages, rule selection), prediction, model bundle |
| `src/metric.py` | Macro F0.5, reference and vectorised forms |
| `src/config.py` | All settings; saved inside the model bundle |
| `tests/` | 36 tests, including a synthetic end-to-end run checked by the challenge validator |

To reproduce both output files:

```bash
cd code/business_entity_resolution
python3.12 -m venv .venv && .venv/bin/pip install -r requirements.txt
.venv/bin/python -m src.cli run --data <dataset dir> --model artifacts/model \
    --out ../../output --sample 300000 --folds 4
```

Runtime on a 24 GB, CPU-only laptop is about 2.5 hours for training and
test prediction, with peak memory about 11 GB.

### B. Additional Results

- Blocking per country (train): US recall 0.980, ceiling 0.994; India recall
  0.955, ceiling 0.984.
- Test predictions (v3): US 94.3%, India 94.0% and France 95.4% of Source 1
  records receive at least one match; average matches per matched record are
  3.4–3.6, consistent with the 3.7 seen in training.
