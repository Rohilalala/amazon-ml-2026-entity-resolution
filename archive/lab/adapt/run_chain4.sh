#!/bin/zsh
# Round 3: train3 (from round-2 model) -> logits -> stage 4 (CE_TAG=adapt3) -> TEST 'all' -> recall adds -> sister v2 ->
# restore scrambles -> groupe adds + street drops -> validator -> Desktop UPLOAD_17.
R=/Users/adityarohilla/Documents/amazon-ml-2026; A=$R/v8_lab/adapt/train3; D=$R/v8_lab/hunt/plan; PY=$R/code/business_entity_resolution/.venv/bin/python; H=$R/v8_lab/hunt
export PYTORCH_MPS_HIGH_WATERMARK_RATIO=0.7 PYTORCH_MPS_LOW_WATERMARK_RATIO=0.6 TOKENIZERS_PARALLELISM=true HF_HUB_OFFLINE=1 PYTHONDONTWRITEBYTECODE=1
cd $A && ../../ce/.venv/bin/python t01_adapt.py > out/t01.stdout 2>&1 || { echo "train failed"; exit 1; }
echo "$(date +%T) trained"
cd $A && sh -c "CE_BS=1024 CE_OUTDIR=out/logits ../../ce/.venv/bin/python i01_score.py holdout stage4 test" || { echo "scoring failed"; exit 1; }
echo "$(date +%T) logits done"
export CE_DIR=$A/out/logits CE_TAG=adapt3 POLARS_MAX_THREADS=8
mkdir -p $D/s4ce/out_adapt3
cd $D/s4ce && ./run_p01.sh || { echo "p01 failed"; exit 1; }
/usr/bin/time -l $PY p02_eval_nog.py ACNRST_3s_adapt3 ACNRST 'seeds=[42,7,99]' > out_adapt3/eval_ACNRST_3s_adapt3.log 2>&1 || { echo "eval failed"; exit 1; }
python3 -c "import json;d=json.load(open('$D/s4ce/out_adapt3/eval_ACNRST_3s_adapt3.json'));print('GATE OOF',d['oof_f05'],'holdout',d['holdout_f05']['ACNRST_3s_adapt3'],'vs A_all_v2',d['diff_vs_stage4_A_v2'])"
echo "$(date +%T) eval done"
./run_p03.sh ACNRST_3s_adapt3 || { echo "p03 failed"; exit 1; }
./run_build.sh ACNRST_3s_adapt3 || { echo "build failed"; exit 1; }
cd $R
RECALL_BASE=$D/s4ce/out_adapt3/upload_ACNRST_3s_adapt3_all/matching_results.tsv RECALL_OUT=$H/final_adapt3_all $PY v8_lab/recall/build/b02_rebase.py > $H/rebase_adapt3_all.log 2>&1 || echo "rebase failed"
$PY $H/fr_sister/fr_sister_drop.py $H/final_adapt3_all/ALL/matching_results.tsv $H/final_adapt3_sister v2 20 2>&1 | grep -v Deprec | cut -c1-160
$PY $H/restore_scrambles.py $H/final_adapt3_all/ALL/matching_results.tsv $H/final_adapt3_sister/matching_results.tsv $H/final_adapt3_restored 2>&1 | grep -v "Deprec\|def pairs"
(cd $H/deep/judge && $PY apply_combo.py $H/final_adapt3_restored/matching_results.tsv $H/final17 2>&1 | grep -v Deprec)
echo "validator: $(python3 student_resource/utils/validate_submission.py --matching $H/final17/matching_results.tsv --candidate output_v7_final/candidate_pairs.tsv --test-dir student_resource/dataset/test 2>&1 | tail -1)"
mkdir -p ~/Desktop/UPLOAD_17_round3 && cp $H/final17/matching_results.tsv ~/Desktop/UPLOAD_17_round3/ && echo "copied to Desktop"
echo "$(date +%T) CHAIN DONE"
