#!/bin/sh
# Resume v3 cross-encoder training from v8_lab/ce/v3_encoder/out/ckpt (step 8,040, el 45.16 min, 1,029,085 pairs seen),
# then the same scoring chain as v3_encoder/run_all.sh (VAL sets, m01, HOLDOUT/STAGE4/TEST; resumable chunks).
# NOT RUN by the planner. Why a new script: the 13:23 run_nolock.sh attempt died at 13:27 with MPS OOM (51 skips > 50)
# because PYTORCH_MPS_HIGH_WATERMARK_RATIO=0.5 caps MPS at 8.88 GiB while bs128 training peaks at 10.2 GiB driver memory
# (v1/v2/v3 logs); 0.7/0.6 is the setting that ran v1, v2 and v3's first 8,040 steps with 0 OOM.
# usage: [LOCK=1] CE_BUDGET_MIN=<45.16 + training minutes wanted> resume_v3.sh
#   LOCK=1 -> whole chain under day1b/heavy.sh (safe for RAM, but blocks every other heavy job for ~3.5-4 h)
#   LOCK unset -> outside the lock: v3 needs ~16 GB (RSS 5.6 GB + MPS <= 10.2 GiB); a concurrent heavy job (<= 11 GB) can swap.
# The schedule is prog = max(step/39113, el/budget): the budget makes LR reach 0 at the deadline whatever the throughput.
cd /Users/adityarohilla/Documents/amazon-ml-2026/v8_lab/ce/v3_encoder || exit 1
[ -n "$CE_BUDGET_MIN" ] || { echo "set CE_BUDGET_MIN (e.g. 215 = full pass at ~390 pairs/s)"; exit 1; }
export CE_BUDGET_MIN PYTORCH_MPS_HIGH_WATERMARK_RATIO=0.7 PYTORCH_MPS_LOW_WATERMARK_RATIO=0.6 TOKENIZERS_PARALLELISM=true HF_HUB_OFFLINE=1
PY=../.venv/bin/python
CHAIN="CE_CKPT_MIN=15 $PY t01_train.py && { CE_BS=1024 CE_MODEL=../v2_encoder/model CE_OUTDIR=out/v2 $PY i01_score.py val3; CE_BS=1024 $PY i01_score.py val3 val2 ceval; $PY m01_val.py; CE_BS=1024 $PY i01_score.py holdout stage4 test; }"
if [ -n "$LOCK" ]; then exec /usr/bin/time -l ../../day1b/heavy.sh sh -c "$CHAIN" >> out/resume_v3.stdout 2>&1
else exec /usr/bin/time -l sh -c "$CHAIN" >> out/resume_v3.stdout 2>&1; fi
