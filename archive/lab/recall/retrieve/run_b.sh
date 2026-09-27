#!/bin/sh
# Reverse pass: every S2/S3 record -> top-3 S1 among ALL S1 of its country (TEST, then TRAIN). Runs inside heavy.sh.
set -e
cd /Users/adityarohilla/Documents/amazon-ml-2026/v8_lab/recall/retrieve
PY=/Users/adityarohilla/Documents/amazon-ml-2026/v8_lab/ce/.venv/bin/python
export PYTORCH_MPS_HIGH_WATERMARK_RATIO=0.7 PYTORCH_MPS_LOW_WATERMARK_RATIO=0.6
for c in us india france; do for s in 2 3; do [ -f out/knn/test_${c}_s${s}_rev.npz ] || $PY k01_knn.py test $c $s rev; done; done
for c in us india; do for s in 2 3; do [ -f out/knn/train_${c}_s${s}_rev.npz ] || $PY k01_knn.py train $c $s rev; done; done
echo RUN_B_DONE
