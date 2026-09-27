#!/bin/zsh
# Assemble A_all / B_usin / C_fr with mix.py and run the official validator on each (heavy.sh).
R=/Users/adityarohilla/Documents/amazon-ml-2026
D=$R/v8_lab/ce/build
O=$D/out
PY=$R/code/business_entity_resolution/.venv/bin/python
H=$R/v8_lab/day1b/heavy.sh
S=v8_lab/day1b/stack/r4
cd $R
typeset -A SPEC
SPEC[A_all]="us=$O/keep_us.parquet india=$O/keep_india.parquet france=$O/keep_france.parquet"
SPEC[B_usin]="us=$O/keep_us.parquet india=$O/keep_india.parquet france=$S"
SPEC[C_fr]="us=$S india=$S france=$O/keep_france.parquet"
for k in A_all B_usin C_fr; do
  mkdir -p $D/$k
  PYTHONDONTWRITEBYTECODE=1 $H /usr/bin/time -l $PY v8_lab/audit_a/mix.py $D/$k ${=SPEC[$k]} > $D/$k/mix.log 2> $D/$k/mix.err
  echo "mix exit $?" >> $D/$k/mix.log
  $H python3 student_resource/utils/validate_submission.py --matching $D/$k/matching_results.tsv --candidate output_v7_final/candidate_pairs.tsv --test-dir student_resource/dataset/test > $D/$k/validate.log 2>&1
  echo "validator exit $?" >> $D/$k/validate.log
done
PYTHONDONTWRITEBYTECODE=1 $H $PY $D/b03_rows.py > $O/b03.log 2> $O/b03.err
echo "b03 exit $?" >> $O/b03.log
echo done > $O/b02.done
