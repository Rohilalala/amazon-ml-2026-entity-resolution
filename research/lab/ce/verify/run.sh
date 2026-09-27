#!/bin/sh
# Heavy audit steps (v01 leakage ~3.5 GB, v03 test inputs) under the global heavy.sh lock, one lock wait.
cd "$(dirname "$0")"
PY=/Users/adityarohilla/Documents/amazon-ml-2026/code/business_entity_resolution/.venv/bin/python
export PYTHONDONTWRITEBYTECODE=1 RSS_LIMIT_GB=12
/usr/bin/time -l $PY v01_leak.py > out/v01_leak.log 2>&1
/usr/bin/time -l $PY v03_test.py > out/v03_test.log 2>&1
echo done > out/run.done
