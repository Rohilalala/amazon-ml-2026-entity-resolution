"""India sparse k30 channel ON TOP OF any base file (b03_assemble.py with the base as a parameter and 'j already matched'
as a filter instead of an assert, so it stacks on a base that already carries the dense recall adds).
adds = b02 pairs with CE v2 logit > T, j not matched anywhere in BASE, one S1 per j (highest logit);
India kept = BASE India rows + adds -> OUT/india_kept.parquet; OUT/matching_results.tsv via mix.py (us/france = BASE).
Label evidence (u01 + s20): T>4 true-add precision 484/486 on the train sample, 34.1% of the true adds are NOT in the
dense m1_r1 lists -> incremental India dF ~ +0.0003 over dense; T>6 454/454.
usage (repo venv): u02_union.py BASE_DIR OUT_DIR [T=6] [B02=.../r6_blocking_recall/build/out/b02_ce.parquet] [DRY=1]"""
import os, subprocess, sys
from pathlib import Path
import polars as pl
ROOT = Path("/Users/adityarohilla/Documents/amazon-ml-2026")
BASE, OUT = sys.argv[1], Path(sys.argv[2])
T = float(sys.argv[3]) if len(sys.argv) > 3 else 6.0
src = Path(sys.argv[4]) if len(sys.argv) > 4 else ROOT / "v8_lab/day1b/r6_blocking_recall/build/out/b02_ce.parquet"
DRY = bool(os.environ.get("DRY"))
st = pl.read_csv(ROOT / BASE / "matching_results.tsv", separator="\t", quote_char=None, infer_schema=False)
allm = st.select(pl.col("source1_entity_id").alias("s1"), pl.col("matched_entity_ids").fill_null("").str.split(",").alias("m")).explode("m").filter(pl.col("m") != "")
ce = pl.read_parquet(src)
adds = (ce.filter(pl.col("ce_logit") > T).join(allm.select("m").unique(), on="m", how="anti")
        .sort("ce_logit", descending=True).unique("m", keep="first", maintain_order=True))
s1 = pl.read_csv(ROOT / "student_resource/dataset/test/test_source1.tsv", separator="\t", quote_char=None, infer_schema=False)
india = s1.filter(pl.col("country").str.strip_chars().str.to_lowercase() == "india").select(pl.col("entity_id").str.strip_chars().alias("s1"))
cur = allm.join(india, on="s1", how="semi")
kept = pl.concat([cur, adds.select("s1", "m")]).unique()
n1 = india.height
print(f"T>{T}: candidates {ce.height}, adds {adds.height} ({adds.height / n1:.4f}/S1; b03 train expectation ~0.0095 at T>4 "
      f"before removing dense-covered j), S1 touched {adds['s1'].n_unique()}, by hrel {adds.group_by('hrel').len().sort('hrel').rows()}; "
      f"India kept pairs {cur.height} -> {kept.height}", flush=True)
if DRY:
    sys.exit(0)
if not (0.001 <= adds.height / n1 <= 0.02):
    sys.exit("adds per S1 outside [0.001, 0.02]: inspect before building (orphan-copy decoys?)")
OUT.mkdir(parents=True, exist_ok=True); kept.write_parquet(OUT / "india_kept.parquet")
subprocess.run([str(ROOT / "code/business_entity_resolution/.venv/bin/python"), str(ROOT / "v8_lab/audit_a/mix.py"), str(OUT),
                f"us={BASE}", f"india={OUT / 'india_kept.parquet'}", f"france={BASE}"], check=True)
