#!/bin/zsh
# A_all_v2 upload file: the combined stage 4 (ACNRST_3s_v2ce) keeps per country, decided by v2_stacker/p04_build.py
# (b01_stage4_test.py's decision path, asserted to reproduce build/A_all's keeps with 0 differing pairs), assembled
# with mix.py (~2.1 GB), then the official validator (~9 GB -> heavy.sh), then b02_stats.py (guarded < 3.2 GB).
R=/Users/adityarohilla/Documents/amazon-ml-2026
D=$R/v8_lab/ce/v2_build; O=$D/A_all_v2; K=$R/v8_lab/ce/v2_stacker/out_v2ce/build_ACNRST_3s_v2ce
PY=$R/code/business_entity_resolution/.venv/bin/python
H=$R/v8_lab/day1b/heavy.sh
export PYTHONDONTWRITEBYTECODE=1
cd $R
mkdir -p $O
/usr/bin/time -l $PY v8_lab/audit_a/mix.py $O us=$K/keep_us.parquet india=$K/keep_india.parquet france=$K/keep_france.parquet > $O/mix.log 2> $O/mix.err || { echo "mix failed" >> $O/mix.log; exit 1; }
$H /usr/bin/time -l python3 student_resource/utils/validate_submission.py --matching $O/matching_results.tsv \
   --candidate output_v7_final/candidate_pairs.tsv --test-dir student_resource/dataset/test > $O/validate.log 2>&1
echo "validator exit $?" >> $O/validate.log
$D/run_b02.sh  # stats peak ~4 GB -> heavy.sh
echo done > $D/out/build.done
