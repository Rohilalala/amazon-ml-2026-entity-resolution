#!/bin/sh
# usage: lane.sh WAIT_JSON "split country src" ...   (runs frames sequentially after WAIT_JSON exists)
cd /Users/adityarohilla/Documents/amazon-ml-2026/v8_lab/hunt/sparse
PY=/Users/adityarohilla/Documents/amazon-ml-2026/code/business_entity_resolution/.venv/bin/python
w=$1; shift
while [ ! -f "$w" ]; do sleep 5; done
for f in "$@"; do set -- $f; [ -f out/eng_$1_$2_s$3.json ] || SC=1000 TH=6 $PY eng.py $1 $2 $3 > logs/eng_$1_$2_s$3.log 2>&1; done
