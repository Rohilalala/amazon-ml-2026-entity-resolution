#!/bin/zsh
# TEST decision (p04, ~5 GB -> heavy.sh), upload folders (mix.py), official validator (~9 GB -> heavy.sh).
# usage: CE_TAG=<tag> run_build.sh NAME  -> out_<tag>/build_NAME/keep_*.parquet, out_<tag>/upload_NAME_{usinbest,all}/
#   usinbest = US + India from NAME, France rows of A_all_v2 (the LB 0.985 file) unchanged; all = NAME in all three countries.
R=/Users/adityarohilla/Documents/amazon-ml-2026
D=$R/v8_lab/hunt/plan/s4ce; O=$D/out_$CE_TAG; N=$1
H=$R/v8_lab/day1b/heavy.sh
PY=$R/code/business_entity_resolution/.venv/bin/python
WPY=/Users/adityarohilla/Documents/amazon-ml-2026-integrate/code/business_entity_resolution/.venv/bin/python
export PYTHONDONTWRITEBYTECODE=1 CE_TAG
cd $R
V2_HEAVY=1 $H /usr/bin/time -l $WPY $D/p04_build.py $N > $O/p04_$N.log 2>&1 || { echo "p04 failed" >> $O/p04_$N.log; exit 1; }
B=$O/build_$N
$PY v8_lab/audit_a/mix.py $O/upload_${N}_usinbest us=$B/keep_us.parquet india=$B/keep_india.parquet france=v8_lab/ce/v2_build/A_all_v2 > $O/mix_${N}_usinbest.log 2>&1 || exit 1
$PY v8_lab/audit_a/mix.py $O/upload_${N}_all us=$B/keep_us.parquet india=$B/keep_india.parquet france=$B/keep_france.parquet > $O/mix_${N}_all.log 2>&1 || exit 1
for k in usinbest all; do
  $H /usr/bin/time -l python3 student_resource/utils/validate_submission.py --matching $O/upload_${N}_$k/matching_results.tsv \
     --candidate output_v7_final/candidate_pairs.tsv --test-dir student_resource/dataset/test > $O/upload_${N}_$k/validate.log 2>&1
  echo "validator exit $?" >> $O/upload_${N}_$k/validate.log
done
echo done > $O/build_$N.done
