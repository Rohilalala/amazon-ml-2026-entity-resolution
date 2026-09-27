#!/bin/sh
# Run a memory-heavy command (>3 GB peak) under one machine-wide lock so parallel jobs never overlap on 24 GB RAM.
# usage: heavy.sh CMD [ARGS...]
exec python3 - "$@" <<'PY'
import fcntl, subprocess, sys, time
f = open("/Users/adityarohilla/Documents/amazon-ml-2026/v8_lab/day1b/.heavy.lock", "w")
t = time.time(); fcntl.flock(f, fcntl.LOCK_EX)
print(f"[heavy] lock acquired after {time.time() - t:.0f}s", file=sys.stderr, flush=True)
sys.exit(subprocess.call(sys.argv[1:]))
PY
