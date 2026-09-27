#!/bin/sh
# b01_build.py (peak a few GB) then the official validator with --check-ids on ALL and USIN, all under the machine-wide lock.
R=/Users/adityarohilla/Documents/amazon-ml-2026; D=$R/v8_lab/recall/build; H=$R/v8_lab/day1b/heavy.sh
PY=$R/code/business_entity_resolution/.venv/bin/python
export PYTHONDONTWRITEBYTECODE=1 POLARS_MAX_THREADS=6
cd $R
mkdir -p $D/out
$H /usr/bin/time -l $PY $D/b01_build.py > $D/out/b01.log 2>&1 || { echo "b01 failed" >> $D/out/b01.log; exit 1; }
for V in ALL USIN; do
  $H /usr/bin/time -l python3 student_resource/utils/validate_submission.py --matching $D/$V/matching_results.tsv \
     --candidate $D/$V/candidate_pairs.tsv --test-dir student_resource/dataset/test --check-ids > $D/$V/validate.log 2>&1
  echo "validator exit $?" >> $D/$V/validate.log
done
echo done > $D/out/build.done
