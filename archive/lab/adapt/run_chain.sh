#!/bin/zsh
# Adapted encoder -> logits (HOLDOUT/STAGE4/TEST) -> ACNRST_3s stage 4 (OOF + holdout eval) -> TEST files (usinbest/all) -> recall adds rebased.
R=/Users/adityarohilla/Documents/amazon-ml-2026; A=$R/v8_lab/adapt/train; D=$R/v8_lab/hunt/plan; PY=$R/code/business_entity_resolution/.venv/bin/python
export PYTORCH_MPS_HIGH_WATERMARK_RATIO=0.7 PYTORCH_MPS_LOW_WATERMARK_RATIO=0.6 TOKENIZERS_PARALLELISM=true HF_HUB_OFFLINE=1 PYTHONDONTWRITEBYTECODE=1
cd $A && $R/v8_lab/day1b/heavy.sh sh -c "CE_BS=1024 CE_OUTDIR=out/logits ../../ce/.venv/bin/python i01_score.py holdout stage4 test" || { echo "scoring failed"; exit 1; }
echo "$(date +%T) logits done"
export CE_DIR=$A/out/logits CE_TAG=adapt
$D/s4ce/run_p01.sh || exit 1
$D/s4ce/run_eval.sh ACNRST_3s_adapt ACNRST 'seeds=[42,7,99]' || exit 1
python3 -c "import json;d=json.load(open('$D/s4ce/out_adapt/eval_ACNRST_3s_adapt.json'));print('GATE OOF',d['oof_f05'],'holdout',d['holdout_f05']['ACNRST_3s_adapt'],'vs A_all_v2',d['diff_vs_stage4_A_v2'])"
echo "$(date +%T) eval done"
$D/s4ce/run_p03.sh ACNRST_3s_adapt || exit 1
$D/s4ce/run_build.sh ACNRST_3s_adapt || exit 1
cd $R
for V in all usinbest; do
  RECALL_BASE=$D/s4ce/out_adapt/upload_ACNRST_3s_adapt_$V/matching_results.tsv RECALL_OUT=$R/v8_lab/hunt/final_adapt_$V POLARS_MAX_THREADS=6 $PY v8_lab/recall/build/b02_rebase.py > $R/v8_lab/hunt/rebase_adapt_$V.log 2>&1
  echo "$V: $(python3 student_resource/utils/validate_submission.py --matching v8_lab/hunt/final_adapt_$V/ALL/matching_results.tsv --candidate output_v7_final/candidate_pairs.tsv --test-dir student_resource/dataset/test 2>&1 | tail -1)"
done
echo "$(date +%T) CHAIN DONE"
