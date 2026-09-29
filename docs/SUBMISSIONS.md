# Submission log

Scores are out-of-fold macro F0.5 on a sample of training Source 1 records
(folds grouped by record). The Leaderboard column holds the portal score of versions uploaded
as they are; the logs below list every upload.

| Version | Commit | Model | Blocking recall / ceiling | OOF F0.5 (US / India) | Leaderboard |
|---|---|---|---|---|---|
| v1 | 443b1cb | 1 stage, 100k records, 3 folds | 0.970 / 0.990 | 0.960 (0.966 / 0.951) | |
| v2 | 66a5f2a | 2 stages, 300k records, 4 folds | 0.970 / 0.990 | 0.969 (0.974 / 0.962) | |
| v3 | 9e96865 | v2 + house-number similarity features | 0.970 / 0.990 | 0.974 (0.979 / 0.966) | 0.9644 (25 Sep 08:36) |
| v5 | 81ec091 | + Indic dictionary, rarest address keys, token-alignment/ambiguity features, stage 3 group model, 500k records | 0.978 / 0.993 | 0.9845 (0.9846 / 0.9843); holdout 0.9851 | |
| v7 | 4647c23 | v6 keys (max_df 3000), France locale normalisation, unseen-country calibration, stage-3 variants | 0.987 / 0.996 | 0.9869; holdout 0.9873 (stage 2 only 0.9866, robust 0.9871) | |
| v7-stage2 | 4647c23 | v7 decided on stage-2 probabilities (drops stage 3, which adds sibling-supported US house variants on test) | same | holdout 0.9866 | |
| v7-usdrop | 4647c23 | v7 + US decoy-copy-group drop rule (a test probe) | same | train-harmful by design; test probe | |
| v7-final | e86e3d6 | v7, India decided on stage 3, US and France on stage 2 | 0.987 / 0.996 | holdout 0.9873 (India stage 3) / 0.9872 (US stage 2) | 0.971137 (25 Sep 22:49) |
| v8 | fb1725d | bundle = v7 + `vocab` tables + `configure` settings (amb_s23_addr_2 cap, training-pool token frequencies for US/India, pinned unseen empty-share target, unseen-country rules); no retraining. Reproduces the lab's file with the same corrections except 4 France rows | same | not measured (test-pool corrections) | |
| v9 | 63eb9fc | v8 bundle + stage 4 (`stage4-import` of the fine-tuned multilingual-e5-small pair encoder and the LightGBM stacker); `stage4_scope` "seen" = output_v9_seen (US/India on stage 4, France as v8), `decide --stage4-scope all` = output_v9_all. US/India rows equal the lab's US/India stage-4 file, France rows equal v8; output_v9_all is byte-identical to the lab's all-country stage-4 file | same | holdout +0.00179 [0.00156, 0.00202] over v7 production (US and India) | |

Leave-one-country-out (v2, stage 1): trained on India, scored on US 0.944;
trained on US, scored on India 0.864.

## Leaderboard log (exact portal values)
| When | File | Score | Note |
|---|---|---|---|
| 25 Sep 08:36 | output_v3 | 0.964386 | |
| 25 Sep 22:49 | output_v7_final | **0.971137** | rank 460 on 26 Sep morning; the base of the 26-27 Sep work |

## Leaderboard log, 26-27 Sep (exact 6-decimal values where known)
| When | File | LB |
|---|---|---|
| 26 Sep 11:36 | v7-final + US amb cap | 0.974 |
| 27 Sep | stage 4 (cross-encoder v1), all countries / US+India | 0.985 / 0.985 |
| 27 Sep | stage 4 v2 (more data + richer stacker) | 0.985 |
| 27 Sep | + dense-recall adds | 0.986 |
| 27 Sep | v3 encoder + recall | 0.986 |
| 27 Sep | + France sister filter (strict) | 0.987 |
| 27 Sep | + France sister filter v2 | 0.988 |
| 27 Sep | + test-time adaptation round 1 | 0.989 |
| 27 Sep | + adaptation round 2 + France groupe/street fixes | 0.989538 |
| 27 Sep ~23:10 | + adaptation round 3 + France, India and US list rules (steps F-H) | **0.989667 (final best)** |
| 27 Sep ~23:20 | the final-best file minus 4,084 least-confident France pairs + 129 France near-tie abstentions | 0.989601 |

Probes (France only): France empty 0.851 (France F about 0.957 at the time); keep-more 0.985; more empties 0.986.
