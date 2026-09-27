#!/bin/zsh
R=/Users/adityarohilla/Documents/amazon-ml-2026
D=$R/v8_lab/ce/build
cd $R
PYTHONDONTWRITEBYTECODE=1 $R/v8_lab/day1b/heavy.sh /usr/bin/time -l $R/code/business_entity_resolution/.venv/bin/python $D/b04_holdout_hv.py > $D/out/b04.log 2> $D/out/b04.err
echo "exit $?" >> $D/out/b04.log
