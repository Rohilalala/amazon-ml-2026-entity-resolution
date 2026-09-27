#!/bin/sh
# All verifier steps, each under the machine-wide heavy lock; logs in out/.
R=/Users/adityarohilla/Documents/amazon-ml-2026; D=$R/v8_lab/recall/verify; H=$R/v8_lab/day1b/heavy.sh
PY=$R/code/business_entity_resolution/.venv/bin/python
export PYTHONDONTWRITEBYTECODE=1 POLARS_MAX_THREADS=6
cd $R
for S in v04_leak v03_holdout v01_files v02_replay; do
  $H /usr/bin/time -l $PY $D/$S.py > $D/out/$S.log 2>&1; echo "exit $?" >> $D/out/$S.log
done
for V in ALL USIN; do
  $H /usr/bin/time -l python3 student_resource/utils/validate_submission.py --matching $R/v8_lab/recall/build/$V/matching_results.tsv \
     --candidate $R/v8_lab/recall/build/$V/candidate_pairs.tsv --test-dir student_resource/dataset/test --check-ids > $D/out/validate_$V.log 2>&1
  echo "validator exit $?" >> $D/out/validate_$V.log
done
echo done > $D/out/run.done
