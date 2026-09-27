# Research and post-processing snapshot (27 Sep 2026)

Scripts behind the leaderboard files after the stage-4 pipeline in `code/business_entity_resolution` (branch `stage4`,
LB 0.985). They are lab scripts with absolute paths, kept here for version history; the integrated pipeline code is the
reference for reproduction. Data, model weights and outputs are not included.

| Folder | What it does | LB effect |
|---|---|---|
| `v8_lab/ce/` | Cross-encoder (multilingual-e5-small, MIT) fine-tuning v1-v3, stage-4 stacker, test builds | 0.974 -> 0.985 |
| `v8_lab/recall/` | Dense retrieval (e5 embeddings, kNN) + recall head for blocking-missed pairs | 0.985 -> 0.986 |
| `v8_lab/hunt/fr_sister/` | France sister-organisation filter (one organisation-type word swapped at the same address) | 0.986 -> 0.988 |
| `v8_lab/adapt/` | Test-time adaptation: fine-tuning on confident test decisions, rounds 1-3 | 0.988 -> 0.9895 |
| `v8_lab/hunt/deep/`, `sweep/`, `extras/`, `restore_scrambles.py` | Small France/India rules (tail copies, street decoys, cie expansions, native-script copies, source-cap limits) | about +0.0001 |
| `v8_lab/hunt/plan/s4ce/` | Stage-4 stacker variants on any logits folder, test decision + build | |
| `v8_lab/day1b/`, `v8_lab/audit_a/` | Machine-wide heavy-job lock, file mixer, per-country rescoring helpers | |
