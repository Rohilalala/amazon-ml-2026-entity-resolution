#!/bin/sh
# v2 CE scores for the new pairs, under the machine-wide lock. Resumes from chunks.
cd "$(dirname "$0")"
export PYTORCH_MPS_HIGH_WATERMARK_RATIO=0.4 PYTORCH_MPS_LOW_WATERMARK_RATIO=0.3 HF_HUB_OFFLINE=1 CE_BS=512
exec /usr/bin/time -l ../../day1b/heavy.sh ../../ce/.venv/bin/python c01_score.py head holdout test
