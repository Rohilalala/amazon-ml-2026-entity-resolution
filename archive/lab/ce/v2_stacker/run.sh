#!/bin/sh
# usage: run.sh NAME GROUPS [lgb_param=json ...]  -> out/eval_NAME.{log,json}, model/NAME.{txt,json}
cd "$(dirname "$0")"
export PYTHONDONTWRITEBYTECODE=1 POLARS_MAX_THREADS=6
/usr/bin/time -l /Users/adityarohilla/Documents/amazon-ml-2026/code/business_entity_resolution/.venv/bin/python p02_eval.py "$@" > "out/eval_$1.log" 2>&1
