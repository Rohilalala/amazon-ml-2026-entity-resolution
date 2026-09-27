#!/bin/sh
# Resume v3 training from out/ckpt WITHOUT the global heavy lock (the recall workflow owns it), lower MPS cap,
# then the same scoring steps as run_all.sh. Re-running resumes each step.
cd "$(dirname "$0")"
export PYTORCH_MPS_HIGH_WATERMARK_RATIO=0.7 PYTORCH_MPS_LOW_WATERMARK_RATIO=0.6 TOKENIZERS_PARALLELISM=true HF_HUB_OFFLINE=1
PY=../.venv/bin/python
CE_CKPT_MIN=15 $PY t01_train.py && {
CE_BS=1024 CE_MODEL=../v2_encoder/model CE_OUTDIR=out/v2 $PY i01_score.py val3;
CE_BS=1024 $PY i01_score.py val3 val2 ceval;
$PY m01_val.py;
CE_BS=1024 $PY i01_score.py holdout stage4 test; }
