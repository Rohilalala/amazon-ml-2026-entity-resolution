# Usage: apply_late.py BASE_TSV OUT_DIR -> OUT_DIR/matching_results.tsv
# Drops apply where present; an add applies only if its S1 row exists and m is kept by no S1 in the base.
import sys, os, polars as pl
from pathlib import Path
H=Path(__file__).parent
base_tsv,out=sys.argv[1],sys.argv[2]
os.makedirs(out,exist_ok=True)
rows=[l.rstrip('\n').split('\t') for l in open(base_tsv,encoding='utf-8')]
head,rows=rows[0],rows[1:]
ms=[[m for m in (r[1].split(',') if len(r)>1 else []) if m] for r in rows]
drops=set(pl.read_parquet(H/'drops.parquet').select('s1','m').iter_rows())
adds=pl.read_parquet(H/'adds.parquet').select('s1','m').iter_rows()
kept={m for L in ms for m in L}
pos={r[0]:k for k,r in enumerate(rows)}
nd=na=0
for k,r in enumerate(rows):
    new=[m for m in ms[k] if (r[0],m) not in drops]; nd+=len(ms[k])-len(new); ms[k]=new
for s1,m in adds:
    if s1 in pos and m not in kept:
        ms[pos[s1]].append(m); kept.add(m); na+=1
allm=[m for L in ms for m in L]
assert len(allm)==len(set(allm)), 'm kept twice'
with open(os.path.join(out,'matching_results.tsv'),'w',encoding='utf-8') as f:
    f.write('\t'.join(head)+'\n')
    for r,L in zip(rows,ms): f.write(r[0]+'\t'+','.join(L)+'\n')
print(f'dropped {nd} added {na} rows {len(rows)}')
