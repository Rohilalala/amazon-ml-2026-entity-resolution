#!/bin/sh
# v3 data build under the heavy lock -> out/d01_*.
cd "$(dirname "$0")"
exec /usr/bin/time -l ../../day1b/heavy.sh ../.venv/bin/python d01_data.py
