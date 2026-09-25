# Submission log

Scores are out-of-fold macro F0.5 on a sample of training Source 1 records
(folds grouped by record). Leaderboard scores are filled in after upload.

| Version | Commit | Model | Blocking recall / ceiling | OOF F0.5 (US / India) | Leaderboard |
|---|---|---|---|---|---|
| v1 | 443b1cb | 1 stage, 100k records, 3 folds | 0.970 / 0.990 | 0.960 (0.966 / 0.951) | |
| v2 | 66a5f2a | 2 stages, 300k records, 4 folds | 0.970 / 0.990 | 0.969 (0.974 / 0.962) | |
| v3 | 9e96865 | v2 + house-number similarity features | 0.970 / 0.990 | 0.974 (0.979 / 0.966) | 0.9644 (25 Sep 08:36) |
| v5 | 81ec091 | + Indic dictionary, rarest address keys, token-alignment/ambiguity features, stage 3 group model, 500k records | 0.978 / 0.993 | 0.9845 (0.9846 / 0.9843); holdout 0.9851 | |
| v7 | 4647c23 | v6 keys (max_df 3000), France locale normalisation, unseen-country calibration, stage-3 variants | 0.987 / 0.996 | 0.9869; holdout 0.9873 (stage 2 only 0.9866, robust 0.9871) | |
| v7-stage2 | 4647c23 | v7 decided on stage-2 probabilities (drops stage 3, which adds sibling-supported US house variants on test) | same | holdout 0.9866 | |
| v7-usdrop | 4647c23 | v7 + US decoy-copy-group drop rule (v7_lab/probe) | same | train-harmful by design; test probe | |
| v7-final | (this commit) | v7, India decided on stage 3, US and France on stage 2 | 0.987 / 0.996 | holdout 0.9873 (India stage 3) / 0.9872 (US stage 2) | |

Leave-one-country-out (v2, stage 1): trained on India, scored on US 0.944;
trained on US, scored on India 0.864.
