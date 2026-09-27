#!/bin/sh
# usage: CE_DIR=<logits dir> CE_TAG=<tag> run_eval.sh NAME GROUPS [lgb_param=json ...]  (p01 tables must exist in out_<tag>/)
# -> out_<tag>/eval_NAME.{log,json}, model/NAME*.{txt,json}; RSS guarded at 3.2 GB inside p02
cd "$(dirname "$0")"
export PYTHONDONTWRITEBYTECODE=1 POLARS_MAX_THREADS=6
/usr/bin/time -l /Users/adityarohilla/Documents/amazon-ml-2026/code/business_entity_resolution/.venv/bin/python p02_eval.py "$@" > "out_$CE_TAG/eval_$1.log" 2>&1
