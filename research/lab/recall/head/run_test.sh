#!/bin/sh
# After c01_score.py has written TEST_ce.parquet: TEST head prob + drift check (p01_test.py) under the machine-wide lock.
cd "$(dirname "$0")"
until [ -f out/TEST_ce.parquet ] && ! pgrep -f "c01_score.py" >/dev/null; do sleep 15; done
export PYTHONDONTWRITEBYTECODE=1 POLARS_MAX_THREADS=6
exec /usr/bin/time -l ../../day1b/heavy.sh ../../../code/business_entity_resolution/.venv/bin/python p01_test.py
