#!/bin/sh
# Runs the holdout scoring once the dataset builder has finished (it writes holdout_s1.parquet).
cd "$(dirname "$0")"
while pgrep -f "s03_dataset[s].py" > /dev/null; do sleep 15; done
[ -f out/s03.json ] || { echo "s03 failed, s04 not started"; exit 1; }
/usr/bin/time -l ../../day1b/heavy.sh /Users/adityarohilla/Documents/amazon-ml-2026/code/business_entity_resolution/.venv/bin/python s04_holdout.py
