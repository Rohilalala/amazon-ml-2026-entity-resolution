#!/bin/zsh
# b02_stats.py peaks above 3.2 GB (relation() on the US / India kept pairs): run it under heavy.sh.
R=/Users/adityarohilla/Documents/amazon-ml-2026; D=$R/v8_lab/ce/v2_build
cd $R
B02_HEAVY=1 PYTHONDONTWRITEBYTECODE=1 POLARS_MAX_THREADS=6 $R/v8_lab/day1b/heavy.sh /usr/bin/time -l $R/code/business_entity_resolution/.venv/bin/python $D/b02_stats.py > $D/out/b02.log 2>&1
echo "b02 exit $?" >> $D/out/b02.log
