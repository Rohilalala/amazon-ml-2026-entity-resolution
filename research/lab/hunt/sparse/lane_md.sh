#!/bin/sh
# usage: lane_md.sh WAIT_JSON "country src" ...  (train frames at MAXDF=6000, sequential, after WAIT_JSON exists)
cd /Users/adityarohilla/Documents/amazon-ml-2026/v8_lab/hunt/sparse
PY=/Users/adityarohilla/Documents/amazon-ml-2026/code/business_entity_resolution/.venv/bin/python
w=$1; shift
while [ ! -f "$w" ]; do sleep 5; done
for f in "$@"; do set -- $f; [ -f out/eng_train_$1_s$2_md6000.json ] || MAXDF=6000 SC=${SC:-1000} TH=6 $PY eng.py train $1 $2 > logs/eng_train_$1_s$2_md6000.log 2>&1; done
