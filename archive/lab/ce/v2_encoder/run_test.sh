#!/bin/sh
# Only after the holdout gate passed: score TEST active pairs with v2 under the heavy lock -> out/TEST_ce.parquet.
cd "$(dirname "$0")"
export PYTORCH_MPS_HIGH_WATERMARK_RATIO=0.7 PYTORCH_MPS_LOW_WATERMARK_RATIO=0.6 TOKENIZERS_PARALLELISM=true HF_HUB_OFFLINE=1 CE_BS=1024
exec /usr/bin/time -l ../../day1b/heavy.sh ../.venv/bin/python i01_score.py test
