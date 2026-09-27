#!/bin/zsh
# Stage 4 on the v3 encoder's logits, end to end. NOT RUN by the planner (v3 logits do not exist yet).
# 1) waits for v8_lab/ce/v3_encoder/out/{holdout,stage4}_ce.parquet, checks rows == v2 (c01_rows), builds the feature tables and
#    trains/evaluates ACNRST_3s_v3 (same 57 features, params, seeds 42/7/99 as ACNRST_3s_v2ce) -> s4ce/out_v3/eval_ACNRST_3s_v3.json
#    (gate numbers: diff_vs_stage4_A_v2 = paired bootstrap vs ACNRST_3s_v2ce, the A_all_v2 decision, holdout 0.989291)
# 2) waits for test_ce.parquet, TEST prob (p03), decision + upload folders + validator (run_build.sh; heavy steps via heavy.sh)
# usage: run_v3_stack.sh [eval|test|all]   (default all). Every step < 3.2 GB RSS except p04/validator (heavy.sh).
R=/Users/adityarohilla/Documents/amazon-ml-2026; D=$R/v8_lab/hunt/plan; V3=$R/v8_lab/ce/v3_encoder
export CE_DIR=$V3/out CE_TAG=v3 PYTHONDONTWRITEBYTECODE=1
STEP=${1:-all}
if [[ $STEP == eval || $STEP == all ]]; then
  until [[ -f $V3/out/stage4_ce.parquet && -f $V3/out/holdout_ce.parquet ]]; do sleep 20; done; sleep 10
  (cd $V3 && ../.venv/bin/python c01_rows.py HOLDOUT STAGE4) || exit 1
  $D/s4ce/run_p01.sh || exit 1
  $D/s4ce/run_eval.sh ACNRST_3s_v3 ACNRST 'seeds=[42,7,99]' || exit 1
  python3 -c "import json;d=json.load(open('$D/s4ce/out_v3/eval_ACNRST_3s_v3.json'));print('OOF',d['oof_f05'],'(v2ce 0.98916)','holdout',d['holdout_f05']['ACNRST_3s_v3'],'vs A_all_v2',d['diff_vs_stage4_A_v2'])"
fi
if [[ $STEP == test || $STEP == all ]]; then
  until [[ -f $V3/out/test_ce.parquet ]]; do sleep 20; done; sleep 10
  (cd $V3 && ../.venv/bin/python c01_rows.py TEST) || exit 1   # overall v2/v3 sign agreement -> v3_encoder/out/c01_rows.json (France rows stay A_all_v2 in usinbest)
  $D/s4ce/run_p03.sh ACNRST_3s_v3 || exit 1
  $D/s4ce/run_build.sh ACNRST_3s_v3 || exit 1
  grep -h "validator exit" $D/s4ce/out_v3/upload_ACNRST_3s_v3_*/validate.log
fi
