# Submission log

Scores are out-of-fold macro F0.5 on a sample of training Source 1 records
(folds grouped by record). Leaderboard scores are filled in after upload.

| Version | Commit | Model | Blocking recall / ceiling | OOF F0.5 (US / India) | Leaderboard |
|---|---|---|---|---|---|
| v1 | 443b1cb | 1 stage, 100k records, 3 folds | 0.970 / 0.990 | 0.960 (0.966 / 0.951) | |
| v2 | 66a5f2a | 2 stages, 300k records, 4 folds | 0.970 / 0.990 | 0.969 (0.974 / 0.962) | |
| v3 | 9e96865 | v2 + house-number similarity features | 0.970 / 0.990 | 0.974 (0.979 / 0.966) | 0.9644 (25 Sep 08:36) |

Leave-one-country-out (v2, stage 1): trained on India, scored on US 0.944;
trained on US, scored on India 0.864.
