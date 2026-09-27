#!/bin/sh
# usage: CE_DIR=<logits dir> CE_TAG=<tag> run_p01.sh  -> out_<tag>/f_{stage4,holdout}.parquet (each < 3.2 GB RSS, guarded)
cd "$(dirname "$0")"; mkdir -p out_$CE_TAG
export PYTHONDONTWRITEBYTECODE=1 POLARS_MAX_THREADS=6
PY=/Users/adityarohilla/Documents/amazon-ml-2026/code/business_entity_resolution/.venv/bin/python
for m in stage4 holdout; do /usr/bin/time -l $PY p01_build.py $m > out_$CE_TAG/p01_$m.log 2>&1 || exit 1; done
