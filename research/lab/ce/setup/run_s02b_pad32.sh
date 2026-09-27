#!/bin/sh
# Training throughput with 4 padded shapes (multiple of 32) and the MPS cache released every 20 steps.
cd "$(dirname "$0")"
export PYTORCH_MPS_HIGH_WATERMARK_RATIO=0.7 PYTORCH_MPS_LOW_WATERMARK_RATIO=0.6 CE_PAD=32 CE_EC=20
for cfg in "128 0" "128 1" "256 1"; do
  set -- $cfg
  ../.venv/bin/python s02b_train_bench.py $1 $2 || echo "{\"bs\": $1, \"freeze_word_emb\": $2, \"pad_multiple\": 32, \"error\": \"failed (see log)\"}" >> out/s02b_train.jsonl
done
