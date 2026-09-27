#!/bin/sh
# Train then evaluate, one heavy.sh lock for both.
cd "$(dirname "$0")"
export PYTORCH_MPS_HIGH_WATERMARK_RATIO=0.7 PYTORCH_MPS_LOW_WATERMARK_RATIO=0.6 TOKENIZERS_PARALLELISM=true
CE_BUDGET_MIN=75 CE_CKPT_MIN=10 ../.venv/bin/python t01_train.py && ../.venv/bin/python t02_eval.py
