#!/bin/sh
# One heavy.sh lock: v3 training (one full pass, resumes from out/ckpt), then VAL3 with v2 (reference), VAL3/VAL2/CE-VAL with v3,
# the quick check (m01), then HOLDOUT, STAGE4 and TEST active pairs with v3 (resumable chunks). Re-running resumes each step.
cd "$(dirname "$0")"
export PYTORCH_MPS_HIGH_WATERMARK_RATIO=0.7 PYTORCH_MPS_LOW_WATERMARK_RATIO=0.6 TOKENIZERS_PARALLELISM=true HF_HUB_OFFLINE=1
PY=../.venv/bin/python
exec /usr/bin/time -l ../../day1b/heavy.sh sh -c "
CE_CKPT_MIN=15 $PY t01_train.py && {
CE_BS=1024 CE_MODEL=../v2_encoder/model CE_OUTDIR=out/v2 $PY i01_score.py val3;
CE_BS=1024 $PY i01_score.py val3 val2 ceval;
$PY m01_val.py;
CE_BS=1024 $PY i01_score.py holdout stage4 test; }"
