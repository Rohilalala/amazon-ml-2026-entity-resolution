#!/bin/sh
# Score VAL3 / VAL2 / CE-VAL / held-out pseudo pairs with the adapted model (out/eval), then the quick check.
cd "$(dirname "$0")"
export PYTORCH_MPS_HIGH_WATERMARK_RATIO=0.7 PYTORCH_MPS_LOW_WATERMARK_RATIO=0.6 TOKENIZERS_PARALLELISM=true HF_HUB_OFFLINE=1
PY=../../ce/.venv/bin/python
../../day1b/heavy.sh sh -c "CE_BS=1024 CE_OUTDIR=out/eval $PY i01_score.py val3 val2 ceval pseudo_ho" && $PY m01_check.py
