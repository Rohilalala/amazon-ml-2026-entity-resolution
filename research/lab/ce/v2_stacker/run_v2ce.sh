#!/bin/zsh
# After the v2 encoder's TEST logits exist: TEST prob for NAME on v2 logits (p03, < 3 GB), then run_build.sh (heavy steps queued).
cd "$(dirname "$0")"; N=$1
until [ -f ../v2_encoder/out/TEST_ce.parquet ]; do sleep 20; done
sleep 30  # let the writer finish
export CE_V2=1 PYTHONDONTWRITEBYTECODE=1 POLARS_MAX_THREADS=6
/usr/bin/time -l /Users/adityarohilla/Documents/amazon-ml-2026/code/business_entity_resolution/.venv/bin/python p03_test.py $N > out_v2ce/p03_$N.log 2>&1 || exit 1
./run_build.sh $N
