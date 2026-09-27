#!/bin/sh
# Score HOLDOUT, STAGE4-TRAIN, TEST active pairs under the global heavy-job lock. Safe to rerun: finished chunks are skipped.
cd "$(dirname "$0")"
export PYTORCH_MPS_HIGH_WATERMARK_RATIO=0.7 PYTORCH_MPS_LOW_WATERMARK_RATIO=0.6 TOKENIZERS_PARALLELISM=true CE_BS=1024
exec /usr/bin/time -l ../../day1b/heavy.sh ../.venv/bin/python i01_score.py holdout stage4 test
