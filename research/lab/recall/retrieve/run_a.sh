#!/bin/sh
# Embed TRAIN -> kNN TRAIN forward (holdout+head S1) -> embed TEST -> kNN TEST forward. Runs inside heavy.sh.
set -e
cd /Users/adityarohilla/Documents/amazon-ml-2026/v8_lab/recall/retrieve
PY=/Users/adityarohilla/Documents/amazon-ml-2026/v8_lab/ce/.venv/bin/python
export HF_HUB_OFFLINE=1 PYTORCH_MPS_HIGH_WATERMARK_RATIO=0.7 PYTORCH_MPS_LOW_WATERMARK_RATIO=0.6
$PY e01_embed.py train 512
for c in us india; do for s in 2 3; do [ -f out/knn/train_${c}_s${s}_fwd.npz ] || $PY k01_knn.py train $c $s fwd out/sets/q_train_rows.npy; done; done
$PY e01_embed.py test 512
for c in us india france; do for s in 2 3; do [ -f out/knn/test_${c}_s${s}_fwd.npz ] || $PY k01_knn.py test $c $s fwd; done; done
echo RUN_A_DONE
