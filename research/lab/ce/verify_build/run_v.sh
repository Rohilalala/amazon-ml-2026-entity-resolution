#!/bin/zsh
# Verifier: official validator (default and --check-ids) on the three builds, then v02 rows and v01 recompute, all via heavy.sh.
R=/Users/adityarohilla/Documents/amazon-ml-2026
V=$R/v8_lab/ce/verify_build
O=$V/out
PY=$R/code/business_entity_resolution/.venv/bin/python
H=$R/v8_lab/day1b/heavy.sh
cd $R
for k in A_all B_usin C_fr; do
  $H python3 student_resource/utils/validate_submission.py --matching v8_lab/ce/build/$k/matching_results.tsv --candidate output_v7_final/candidate_pairs.tsv --test-dir student_resource/dataset/test > $O/val_$k.log 2>&1
  echo "validator exit $?" >> $O/val_$k.log
  $H python3 student_resource/utils/validate_submission.py --matching v8_lab/ce/build/$k/matching_results.tsv --candidate output_v7_final/candidate_pairs.tsv --test-dir student_resource/dataset/test --check-ids > $O/val_ids_$k.log 2>&1
  echo "validator exit $?" >> $O/val_ids_$k.log
done
PYTHONDONTWRITEBYTECODE=1 $H /usr/bin/time -l $PY $V/v02_rows.py > $O/v02.log 2> $O/v02.err
echo "exit $?" >> $O/v02.log
PYTHONDONTWRITEBYTECODE=1 $H /usr/bin/time -l $PY $V/v01_verify.py > $O/v01.log 2> $O/v01.err
echo "exit $?" >> $O/v01.log
echo done > $O/run_v.done
