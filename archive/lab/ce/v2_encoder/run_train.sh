#!/bin/sh
# One heavy.sh lock: v2 training, then VAL2 (v1 + v2) and CE-VAL (v2) scoring, then HOLDOUT + STAGE4 active pairs with v2.
cd "$(dirname "$0")"
export PYTORCH_MPS_HIGH_WATERMARK_RATIO=0.7 PYTORCH_MPS_LOW_WATERMARK_RATIO=0.6 TOKENIZERS_PARALLELISM=true HF_HUB_OFFLINE=1
PY=../.venv/bin/python
exec /usr/bin/time -l ../../day1b/heavy.sh sh -c "
CE_BUDGET_MIN=75 CE_CKPT_MIN=10 $PY t01_train.py &&
CE_BS=1024 CE_MODEL=../train/model CE_OUTDIR=out/v1 $PY i01_score.py val2 &&
CE_BS=1024 $PY i01_score.py val2 ceval holdout stage4"
