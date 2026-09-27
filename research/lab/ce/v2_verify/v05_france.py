"""Verifier (5) France look (label-free): encoder v1 vs v2 drift by country on TEST active pairs; relation mix and
random examples of the pairs A_all_v2 drops / adds in France vs A_all.
usage (repo venv): v05_france.py -> out/v05_france.json"""
import json, sys
from pathlib import Path
import numpy as np, polars as pl
ROOT = Path("/Users/adityarohilla/Documents/amazon-ml-2026"); CE = ROOT / "v8_lab/ce"; OUT = Path(__file__).parent / "out"
sys.path.insert(0, str(ROOT / "v8_lab/audit_a"))
from common import relation
R = {}
A = pl.scan_parquet(CE / "setup/out/test_pairs.parquet").filter("active").select("i", "j", pl.col("country").cast(pl.String), "prob").collect()
v1, v2 = pl.read_parquet(CE / "infer/out/test_ce.parquet"), pl.read_parquet(CE / "v2_encoder/out/TEST_ce.parquet")
assert (v1["i"] == A["i"]).all() and (v2["j"] == A["j"]).all() and (v1["j"] == v2["j"]).all()
z1, z2, c = v1["ce_logit"].to_numpy().astype(float), v2["ce_logit"].to_numpy().astype(float), A["country"].to_numpy()
R["encoder_drift"] = {k: {"pairs": int((c == k).sum()), "v1_pos": round(float((z1[c == k] > 0).mean()), 4), "v2_pos": round(float((z2[c == k] > 0).mean()), 4),
                          "v1_mean": round(float(z1[c == k].mean()), 3), "v2_mean": round(float(z2[c == k].mean()), 3),
                          "sign_agree": round(float(((z1[c == k] > 0) == (z2[c == k] > 0)).mean()), 4),
                          "pearson": round(float(np.corrcoef(z1[c == k], z2[c == k])[0, 1]), 4),
                          "v2_minus_v1_median": round(float(np.median(z2[c == k] - z1[c == k])), 3)} for k in ("us", "india", "france")}
del v1, v2, A, z1, z2, c
rc = (pl.scan_parquet(ROOT / "cache/3ee42f05d06efbcb/recs.parquet").filter(pl.col("country_key") == "france")
      .select("row", pl.col("entity_id").str.strip_chars(), "core", "addr", "house").collect())
tx = pl.scan_parquet(CE / "setup/out/text_test.parquet").join(rc.select("row").lazy(), on="row", how="semi").collect()
ids = rc.select("entity_id", "row")
kA = pl.read_parquet(CE / "build/out/keep_france.parquet"); k2 = pl.read_parquet(CE / "v2_stacker/out_v2ce/build_ACNRST_3s_v2ce/keep_france.parquet")
def rows(df):
    return (df.join(ids.rename({"entity_id": "s1", "row": "i"}), on="s1", how="left").join(ids.rename({"entity_id": "m", "row": "j"}), on="m", how="left")
            .select(pl.col("i").cast(pl.UInt32), pl.col("j").cast(pl.UInt32)))
drop, add, both = rows(kA.join(k2, on=["s1", "m"], how="anti")), rows(k2.join(kA, on=["s1", "m"], how="anti")), rows(kA.join(k2, on=["s1", "m"], how="semi"))
R["counts"] = {"dropped": drop.height, "added": add.height, "kept_both": both.height}
def mix(p):
    r = relation(p, rc.select("row", "core", "addr", "house"))
    return {"hrel": dict(sorted(r["hrel"].value_counts().iter_rows(), key=lambda x: -x[1])), "nrel": dict(sorted(r["nrel"].value_counts().iter_rows(), key=lambda x: -x[1]))}
R["relation_mix"] = {"dropped": mix(drop), "added": mix(add), "kept_both_sample": mix(both.sample(20000, seed=1))}
t = dict(zip(tx["row"].to_list(), tx["text"].to_list()))
R["examples_dropped"] = [f"{t[a]}  ||  {t[b]}" for a, b in drop.sample(25, seed=7).iter_rows()]
R["examples_added"] = [f"{t[a]}  ||  {t[b]}" for a, b in add.sample(15, seed=7).iter_rows()]
(OUT / "v05_france.json").write_text(json.dumps(R, indent=1, ensure_ascii=False))
print(json.dumps(R, indent=1, ensure_ascii=False))
