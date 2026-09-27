#!/bin/sh
# usage: CE_DIR=<logits dir> CE_TAG=<tag> run_p03.sh NAME -> out_<tag>/test_NAME_prob.parquet (< 3 GB, guarded)
cd "$(dirname "$0")"
export PYTHONDONTWRITEBYTECODE=1 POLARS_MAX_THREADS=6
/usr/bin/time -l /Users/adityarohilla/Documents/amazon-ml-2026/code/business_entity_resolution/.venv/bin/python p03_test.py "$1" > "out_$CE_TAG/p03_$1.log" 2>&1
