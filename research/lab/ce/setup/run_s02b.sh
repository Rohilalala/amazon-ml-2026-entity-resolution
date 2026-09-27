#!/bin/sh
# Training-throughput grid, each configuration in its own process; MPS memory capped so a blow-up errors instead of swapping.
cd "$(dirname "$0")"
export PYTORCH_MPS_HIGH_WATERMARK_RATIO=0.5 PYTORCH_MPS_LOW_WATERMARK_RATIO=0.4
for cfg in "64 1" "128 0" "128 1" "256 0" "256 1"; do
  set -- $cfg
  ../.venv/bin/python s02b_train_bench.py $1 $2 || echo "{\"bs\": $1, \"freeze_word_emb\": $2, \"error\": \"failed (see log)\"}" >> out/s02b_train.jsonl
done
