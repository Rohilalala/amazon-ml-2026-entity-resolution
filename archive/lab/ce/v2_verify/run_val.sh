#!/bin/zsh
# Verifier: official validator (default and --check-ids) on A_all_v2 and the usin variant, each under heavy.sh.
R=/Users/adityarohilla/Documents/amazon-ml-2026
O=$R/v8_lab/ce/v2_verify/out
H=$R/v8_lab/day1b/heavy.sh
cd $R
for k in A_all_v2:v8_lab/ce/v2_build/A_all_v2 usin:v8_lab/ce/v2_stacker/out_v2ce/upload_ACNRST_3s_v2ce_usin; do
  n=${k%%:*}; p=${k#*:}
  shasum -a 256 $p/matching_results.tsv > $O/sha_$n.txt
  $H /usr/bin/time -l python3 student_resource/utils/validate_submission.py --matching $p/matching_results.tsv --candidate output_v7_final/candidate_pairs.tsv --test-dir student_resource/dataset/test > $O/val_$n.log 2>&1
  echo "validator exit $?" >> $O/val_$n.log
  $H /usr/bin/time -l python3 student_resource/utils/validate_submission.py --matching $p/matching_results.tsv --candidate output_v7_final/candidate_pairs.tsv --test-dir student_resource/dataset/test --check-ids > $O/val_ids_$n.log 2>&1
  echo "validator exit $?" >> $O/val_ids_$n.log
done
shasum -a 256 v8_lab/ce/build/A_all/matching_results.tsv > $O/sha_A_all.txt
echo done > $O/run_val.done
