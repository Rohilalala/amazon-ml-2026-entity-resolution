#!/bin/sh
# Adaptation training (resumes from out/ckpt) under the machine-wide heavy lock; MPS watermarks 0.7/0.6 as v3.
cd "$(dirname "$0")"
export PYTORCH_MPS_HIGH_WATERMARK_RATIO=0.7 PYTORCH_MPS_LOW_WATERMARK_RATIO=0.6 TOKENIZERS_PARALLELISM=true HF_HUB_OFFLINE=1
exec /usr/bin/time -l ../../day1b/heavy.sh ../../ce/.venv/bin/python t01_adapt.py
