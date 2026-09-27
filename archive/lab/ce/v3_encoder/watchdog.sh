#!/bin/sh
# Restart run_nolock.sh (every step resumes) if it dies before the TEST logits exist; at most 5 restarts, stop at 22:00.
cd "$(dirname "$0")"
n=0
while [ "$(date +%H%M)" -lt 2200 ]; do
  sleep 120
  [ -f out/TEST_ce.parquet ] && { echo "$(date +%T) done"; exit 0; }
  if ! pgrep -f run_nolock.sh >/dev/null; then
    n=$((n+1)); [ $n -gt 5 ] && { echo "$(date +%T) too many restarts"; exit 1; }
    echo "$(date +%T) restart $n"; mv out/run_nolock.stdout "out/run_nolock_dead$n.stdout" 2>/dev/null
    nohup ./run_nolock.sh > out/run_nolock.stdout 2>&1 &
  fi
done
