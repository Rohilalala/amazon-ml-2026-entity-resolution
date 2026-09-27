#!/bin/zsh
# Verifier runs, in order (repo venv, PYTHONDONTWRITEBYTECODE=1); the heavy ones via day1b/heavy.sh as run on 2026-09-27.
R=/Users/adityarohilla/Documents/amazon-ml-2026; V=$R/v8_lab/ce/v2_verify; H=$R/v8_lab/day1b/heavy.sh
PY=$R/code/business_entity_resolution/.venv/bin/python
export PYTHONDONTWRITEBYTECODE=1 POLARS_MAX_THREADS=6
cd $R
$V/run_val.sh > $V/out/run_val.stdout 2>&1                                                        # validators (default + --check-ids), sha256
RSS_LIMIT_GB=9 $H /usr/bin/time -l $PY $V/v01_leak.py > $V/out/v01_leak.log 2>&1                  # leakage, 5.0 GB
RSS_LIMIT_GB=10 $H /usr/bin/time -l $PY $V/v02_holdout.py > $V/out/v02_holdout.log 2>&1           # holdout gate, 4.6 GB
RSS_LIMIT_GB=10 $H /usr/bin/time -l $PY $V/v03_test_p4.py > $V/out/v03_test_p4.log 2>&1           # TEST 200k p4, 4.2 GB
$H /usr/bin/time -l $PY $V/v04_decide.py > $V/out/v04_decide.log 2>&1                             # TEST decisions + ids, 15.3 GB footprint
$H /usr/bin/time -l $PY $V/v05_france.py > $V/out/v05_france.log 2>&1                             # France profile, 2.7 GB
$H /usr/bin/time -l $PY $V/v06_cv.py > $V/out/v06_cv.log 2>&1                                     # stacker CV + refit, 4.0 GB
