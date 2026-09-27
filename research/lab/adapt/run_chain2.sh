#!/bin/zsh
# Resume of run_chain.sh after the logits: stage-4 eval without the 3.2 GB guard (machine otherwise idle), TEST prob, build,
# recall adds rebased, France sister filter v2, validator.
R=/Users/adityarohilla/Documents/amazon-ml-2026; A=$R/v8_lab/adapt/train; D=$R/v8_lab/hunt/plan; PY=$R/code/business_entity_resolution/.venv/bin/python
export CE_DIR=$A/out/logits CE_TAG=adapt PYTHONDONTWRITEBYTECODE=1 POLARS_MAX_THREADS=6
cd $D/s4ce && /usr/bin/time -l $PY p02_eval_nog.py ACNRST_3s_adapt ACNRST 'seeds=[42,7,99]' > out_adapt/eval_ACNRST_3s_adapt.log 2>&1 || { echo "eval failed"; exit 1; }
python3 -c "import json;d=json.load(open('$D/s4ce/out_adapt/eval_ACNRST_3s_adapt.json'));print('GATE OOF',d['oof_f05'],'holdout',d['holdout_f05']['ACNRST_3s_adapt'],'vs A_all_v2',d['diff_vs_stage4_A_v2'])"
echo "$(date +%T) eval done"
$D/s4ce/run_p03.sh ACNRST_3s_adapt || { echo "p03 failed"; exit 1; }
$D/s4ce/run_build.sh ACNRST_3s_adapt || { echo "build failed"; exit 1; }
cd $R
for V in all usinbest; do
  RECALL_BASE=$D/s4ce/out_adapt/upload_ACNRST_3s_adapt_$V/matching_results.tsv RECALL_OUT=$R/v8_lab/hunt/final_adapt_$V $PY v8_lab/recall/build/b02_rebase.py > $R/v8_lab/hunt/rebase_adapt_$V.log 2>&1 || echo "rebase $V failed"
  $PY v8_lab/hunt/fr_sister/fr_sister_drop.py v8_lab/hunt/final_adapt_$V/ALL/matching_results.tsv v8_lab/hunt/final_adapt_${V}_sister v2 20 2>&1 | grep -v Deprec | cut -c1-200
  echo "$V: $(python3 student_resource/utils/validate_submission.py --matching v8_lab/hunt/final_adapt_${V}_sister/matching_results.tsv --candidate output_v7_final/candidate_pairs.tsv --test-dir student_resource/dataset/test 2>&1 | tail -1)"
done
echo "$(date +%T) CHAIN DONE"
