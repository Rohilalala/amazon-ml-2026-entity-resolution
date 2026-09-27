"""STAGE4-TRAIN consistency check (nothing fitted): AUC of the CE logit, the v7 OOF production prob and the fixed untuned blend
(logit(prob) + ce_logit + delta) / 2 on the active pairs, per country and by empty candidate address. -> out/i03_stage4_check.json"""
import json
from pathlib import Path
import polars as pl
from sklearn.metrics import roc_auc_score
HERE = Path(__file__).parent; CE = HERE.parent; SET = CE / "setup/out"
DELTA = json.load(open(CE / "train/out/t03_metrics.json"))["prior"]["delta_logit"]
C = pl.read_parquet(HERE / "out/stage4_ce.parquet")
S = pl.scan_parquet(SET / "stage4_train.parquet").filter("active").select("i", "j", "y", "country", "prob").collect().join(C, on=["i", "j"], how="left")
assert S.height == C.height and S["ce_logit"].null_count() == 0
emp = (pl.scan_parquet(SET / "text_train.parquet").filter(pl.col("row").is_in(C["j"].unique().implode()))
       .select(j="row", b_empty=pl.col("text").str.splitn(" | ", 2).struct.field("field_1").fill_null("").str.strip_chars() == "").collect())
q = pl.col("prob").cast(pl.Float64).clip(1e-6, 1 - 1e-6)
S = S.join(emp, on="j", how="left").with_columns(blend=((q / (1 - q)).log() + pl.col("ce_logit") + DELTA) / 2)
auc = lambda d: {"pairs": d.height, "pos_rate": round(float(d["y"].mean()), 4),
                 **{c: round(roc_auc_score(d["y"].to_numpy(), d[c].to_numpy()), 5) for c in ("ce_logit", "prob", "blend")}}
R = {}
for c in ("all", "us", "india"):
    d = S if c == "all" else S.filter(pl.col("country") == c)
    R[c] = {"all": auc(d), "b_empty": auc(d.filter("b_empty")), "b_nonempty": auc(d.filter(~pl.col("b_empty")))}
json.dump(R, open(HERE / "out/i03_stage4_check.json", "w"), indent=1); print(json.dumps(R, indent=1))
