#!/bin/zsh
# TEST decision (p04, peak ~3.3 GB on India -> heavy.sh), both upload files (mix.py), official validator (~6-8 GB -> heavy.sh).
# usage: run_build.sh NAME   -> out/build_NAME/, out/upload_NAME_{usin,all}/{matching_results.tsv,validate.log}
R=/Users/adityarohilla/Documents/amazon-ml-2026
D=$R/v8_lab/ce/v2_stacker; O=$D/out; N=$1
[ -n "$CE_V2" ] && O=$D/out_v2ce  # CE_V2=1: the v2-encoder-logit variant (p03/p04 write there)
H=$R/v8_lab/day1b/heavy.sh
PY=$R/code/business_entity_resolution/.venv/bin/python
WPY=/Users/adityarohilla/Documents/amazon-ml-2026-integrate/code/business_entity_resolution/.venv/bin/python
export PYTHONDONTWRITEBYTECODE=1
cd $R
V2_HEAVY=1 $H /usr/bin/time -l $WPY $D/p04_build.py $N > $O/p04_$N.log 2>&1 || { echo "p04 failed" >> $O/p04_$N.log; exit 1; }
B=$O/build_$N
$PY v8_lab/audit_a/mix.py $O/upload_${N}_usin us=$B/keep_us.parquet india=$B/keep_india.parquet france=v8_lab/ce/build/out/keep_france.parquet > $O/mix_${N}_usin.log 2>&1 || exit 1
$PY v8_lab/audit_a/mix.py $O/upload_${N}_all us=$B/keep_us.parquet india=$B/keep_india.parquet france=$B/keep_france.parquet > $O/mix_${N}_all.log 2>&1 || exit 1
for k in usin all; do
  $H /usr/bin/time -l python3 student_resource/utils/validate_submission.py --matching $O/upload_${N}_$k/matching_results.tsv \
     --candidate output_v7_final/candidate_pairs.tsv --test-dir student_resource/dataset/test > $O/upload_${N}_$k/validate.log 2>&1
  echo "validator exit $?" >> $O/upload_${N}_$k/validate.log
done
echo done > $O/build_$N.done
