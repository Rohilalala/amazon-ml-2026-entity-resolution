#!/bin/zsh
# Round-2 adaptation: train2 (from round-1 model, sister negatives) -> logits -> stage 4 (CE_TAG=adapt2) -> TEST file 'all'
# -> recall adds -> France sister filter v2 -> validator.
R=/Users/adityarohilla/Documents/amazon-ml-2026; A=$R/v8_lab/adapt/train2; D=$R/v8_lab/hunt/plan; PY=$R/code/business_entity_resolution/.venv/bin/python
export PYTORCH_MPS_HIGH_WATERMARK_RATIO=0.7 PYTORCH_MPS_LOW_WATERMARK_RATIO=0.6 TOKENIZERS_PARALLELISM=true HF_HUB_OFFLINE=1 PYTHONDONTWRITEBYTECODE=1
cd $A && $R/v8_lab/day1b/heavy.sh ../../ce/.venv/bin/python t01_adapt.py > out/t01.stdout 2>&1 || { echo "train failed"; exit 1; }
echo "$(date +%T) trained"
cd $A && $R/v8_lab/day1b/heavy.sh sh -c "CE_BS=1024 CE_OUTDIR=out/logits ../../ce/.venv/bin/python i01_score.py holdout stage4 test" || { echo "scoring failed"; exit 1; }
echo "$(date +%T) logits done"
export CE_DIR=$A/out/logits CE_TAG=adapt2 POLARS_MAX_THREADS=6
mkdir -p $D/s4ce/out_adapt2
cd $D/s4ce && ./run_p01.sh || { echo "p01 failed"; exit 1; }
/usr/bin/time -l $PY p02_eval_nog.py ACNRST_3s_adapt2 ACNRST 'seeds=[42,7,99]' > out_adapt2/eval_ACNRST_3s_adapt2.log 2>&1 || { echo "eval failed"; exit 1; }
python3 -c "import json;d=json.load(open('$D/s4ce/out_adapt2/eval_ACNRST_3s_adapt2.json'));print('GATE OOF',d['oof_f05'],'holdout',d['holdout_f05']['ACNRST_3s_adapt2'],'vs A_all_v2',d['diff_vs_stage4_A_v2'])"
echo "$(date +%T) eval done"
./run_p03.sh ACNRST_3s_adapt2 || { echo "p03 failed"; exit 1; }
./run_build.sh ACNRST_3s_adapt2 || { echo "build failed"; exit 1; }
cd $R
RECALL_BASE=$D/s4ce/out_adapt2/upload_ACNRST_3s_adapt2_all/matching_results.tsv RECALL_OUT=$R/v8_lab/hunt/final_adapt2_all $PY v8_lab/recall/build/b02_rebase.py > $R/v8_lab/hunt/rebase_adapt2_all.log 2>&1 || echo "rebase failed"
$PY v8_lab/hunt/fr_sister/fr_sister_drop.py v8_lab/hunt/final_adapt2_all/ALL/matching_results.tsv v8_lab/hunt/final_adapt2_all_sister v2 20 2>&1 | grep -v Deprec | cut -c1-200
echo "validator: $(python3 student_resource/utils/validate_submission.py --matching v8_lab/hunt/final_adapt2_all_sister/matching_results.tsv --candidate output_v7_final/candidate_pairs.tsv --test-dir student_resource/dataset/test 2>&1 | tail -1)"
echo "$(date +%T) CHAIN DONE"
