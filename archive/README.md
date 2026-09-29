# Archived experiment scripts (not runnable)

These are the experiment scripts from the last two days of the challenge (26-27 Sep 2026), kept
for history. They carry absolute paths of the machine they ran on, read intermediate files that
are not in this repository, and were superseded by the pipeline code in
[`code/business_entity_resolution`](../code/business_entity_resolution), which reproduces the final
leaderboard file from the shipped model files (see the main [README](../README.md)). Data, model
weights and outputs are not included.

| Folder | What it explored | Now in |
|---|---|---|
| `lab/ce/` | Pair encoder (multilingual-e5-small, MIT) fine-tuning v1-v3, stage-4 stackers, test builds | `src/pair_encoder.py`, `src/stage4.py`, `encoder-train`, `stage4-train` |
| `lab/recall/` | Dense retrieval (embeddings, kNN) and the recall head for pairs the blocker missed | `src/recall.py`, `recall-train` |
| `lab/hunt/fr_sister/`, `lab/hunt/restore_scrambles.py` | France sister-organisation filter and scramble restore | `src/postprocess.py` |
| `lab/adapt/` | Test-time adaptation of the encoder, rounds 1-3 | `src/adapt.py`, `adapt-labels`, `adapt-train` |
| `lab/hunt/deep/`, `lab/hunt/sweep/`, `lab/hunt/extras/`, `lab/hunt/late/` | France / India / US list rules (tail copies, street decoys, cie expansions, pool copies, source caps) | `src/rules.py`, `lists.parquet` |
| `lab/hunt/fr_exact/` | France exact-copy adds (same core name and house number) | not among the final lists |
| `lab/hunt/sparse/` | Wider sparse blocking (more candidates, higher max_df) on the holdout, next to dense recall | not in the pipeline; the final run uses dense recall (`src/recall.py`) |
| `lab/hunt/plan/s4ce/` | Stage-4 stacker variants, test decision and build | `src/stage4.py` (ACNRST set) |
| `lab/day1b/`, `lab/audit_a/` | Heavy-job lock, file mixer, per-country rescoring helpers | (tooling only, not in the pipeline) |

The `judge/` folders under `lab/hunt/` hold read-only checks that print records for inspection and
scripts that merge and apply rule lists. Every accepted list came from a rule; the rules with code
are in `src/rules.py` (step H's region-conflict drops among them), and `lists.parquet` holds the
lists as applied.

Leaderboard effect of each line of work, as logged at the time:

| Work | Leaderboard |
|---|---|
| stage-4 pair encoder and stacker | 0.974 -> 0.985 |
| dense recall | 0.985 -> 0.986 |
| France sister-organisation filter | 0.986 -> 0.988 |
| test-time adaptation, rounds 1-3, with the list rules | 0.988 -> 0.989667 |
