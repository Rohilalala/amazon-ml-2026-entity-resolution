#!/bin/sh
# Stage 4 main (CV, rule, final fit, holdout gate) then the country-transfer proxy. Peak RSS is guarded at 3.2 GB.
cd "$(dirname "$0")"
PY=/Users/adityarohilla/Documents/amazon-ml-2026/code/business_entity_resolution/.venv/bin/python
export PYTHONDONTWRITEBYTECODE=1
/usr/bin/time -l $PY s01_stage4.py main > out/s01_main.log 2>&1 && /usr/bin/time -l $PY s01_stage4.py transfer > out/s01_transfer.log 2>&1
