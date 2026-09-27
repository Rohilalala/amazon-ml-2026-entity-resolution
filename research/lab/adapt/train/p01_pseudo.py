"""Pseudo-labelled TEST pairs + labelled train anchors for test-time adaptation of the v3 cross-encoder.
Pool: TEST active pairs (test_pairs.parquet, v7 prob >= 0.002) with the v3 stage-4 prob p4 (hunt/plan/s4ce/out_v3/test_ACNRST_3s_v3_prob.parquet)
and the v3 CE logit (ce/v3_encoder/out/TEST_ce.parquet); kept = in the v3 stage-4 decision (out_v3/build_ACNRST_3s_v3/keep_*.parquet).
Classes: pos = kept & p4 >= 0.98; hard = per S1 the top 2 non-kept candidates by CE logit among those with p4 <= 0.05;
easy = other pairs with p4 <= 0.02 (kept or not).
Sample (seed 0): N_PSEUDO pairs, France 35%, US/India share the rest by their active-pair counts; per country 55% pos, 45% neg
split half hard / half easy. 10% of the sampled S1 (all their pairs) are held out (is_ho) for the check, never trained on.
Anchor: N_ANCHOR random pairs of the v3 fit set (same filter as ce/v3_encoder/t01_train.py; its leak asserts passed there).
usage (ce venv): p01_pseudo.py -> out/pseudo.parquet, out/anchor.parquet, out/p01.json"""
import json
from pathlib import Path
import numpy as np, polars as pl

HERE = Path(__file__).parent; LAB = HERE.parent.parent; CE = LAB / "ce"; OUT = HERE / "out"
N_PSEUDO, N_ANCHOR, FR, POS, SEED = 555_000, 500_000, 0.35, 0.55, 0
rng = np.random.default_rng(SEED)

T = pl.scan_parquet(CE / "setup/out/test_pairs.parquet").filter("active").select("i", "j", "country").collect()
p = pl.read_parquet(LAB / "hunt/plan/s4ce/out_v3/test_ACNRST_3s_v3_prob.parquet")
z = pl.read_parquet(CE / "v3_encoder/out/TEST_ce.parquet")
assert T["i"].equals(p["i"]) and T["j"].equals(p["j"]) and T["i"].equals(z["i"]) and T["j"].equals(z["j"])
T = T.with_columns(p4=p["p4"], ce=z["ce_logit"], country=pl.col("country").cast(pl.String)); del p, z
ids = pl.scan_parquet(LAB.parent / "cache/3ee42f05d06efbcb/recs.parquet").select("row", "entity_id").collect()
K = pl.concat([pl.read_parquet(LAB / f"hunt/plan/s4ce/out_v3/build_ACNRST_3s_v3/keep_{c}.parquet") for c in ("us", "india", "france")])
K = (K.join(ids.rename({"entity_id": "s1", "row": "i"}), on="s1").join(ids.rename({"entity_id": "m", "row": "j"}), on="m")
     .select("i", "j", kept=pl.lit(True))); del ids
T = T.join(K, on=["i", "j"], how="left").with_columns(pl.col("kept").fill_null(False))
R = {"test_active_pairs": T.height, "kept_pairs_in_decision": K.height, "kept_active": int(T["kept"].sum())}; del K

hard = (T.filter(~pl.col("kept") & (pl.col("p4") <= 0.05)).with_columns(r=pl.col("ce").rank("ordinal", descending=True).over("i"))
        .filter(pl.col("r") <= 2).select("i", "j", hard=pl.lit(True)))
T = (T.join(hard, on=["i", "j"], how="left").with_columns(pl.col("hard").fill_null(False))
     .with_columns(cls=pl.when(pl.col("kept") & (pl.col("p4") >= 0.98)).then(pl.lit("pos"))
                   .when(pl.col("hard")).then(pl.lit("hard")).when(pl.col("p4") <= 0.02).then(pl.lit("easy")).otherwise(None))
     .drop("hard"))
pool = T.group_by("country", "cls").len().sort("country", "cls")
R["pool_by_country_class"] = {f"{c}/{k}": n for c, k, n in pool.iter_rows()}
act = dict(T.group_by("country").len().iter_rows())
quota = {"france": FR * N_PSEUDO}
for c in ("us", "india"): quota[c] = (1 - FR) * N_PSEUDO * act[c] / (act["us"] + act["india"])
S = []
for c, q in quota.items():
    for k, f in (("pos", POS), ("hard", (1 - POS) / 2), ("easy", (1 - POS) / 2)):
        d = T.filter((pl.col("country") == c) & (pl.col("cls") == k)); n = int(round(q * f))
        assert d.height >= n, (c, k, d.height, n)
        S.append(d.gather(rng.choice(d.height, n, replace=False)))
S = pl.concat(S).with_columns(y=(pl.col("cls") == "pos").cast(pl.Int8))
s1 = S["i"].unique().sort(); ho = s1.gather(rng.choice(len(s1), len(s1) // 10, replace=False))
S = S.with_columns(is_ho=pl.col("i").is_in(ho)).sort("i", "j")
S.write_parquet(OUT / "pseudo.parquet")
q = [0.01, 0.05, 0.25, 0.5, 0.75, 0.95, 0.99]
R["sample"] = {f"{c}/{k}/{'ho' if h else 'train'}": n for c, k, h, n in S.group_by("country", "cls", "is_ho").len().sort("country", "cls", "is_ho").iter_rows()}
R["sample_train_pairs"] = int((~S["is_ho"]).sum()); R["sample_ho_pairs"] = int(S["is_ho"].sum())
R["sample_country_share"] = {c: round(n / S.height, 4) for c, n in S.group_by("country").len().iter_rows()}
R["sample_pos_share"] = round(float(S["y"].mean()), 4)
R["dist"] = {f"{c}/{k}": {"ce_q":
                         [round(float(d["ce"].quantile(v)), 2) for v in q],
                         "p4_q": [round(float(d["p4"].quantile(v)), 5) for v in q]}
             for (c, k), d in S.group_by("country", "cls") }
R["dist_quantiles"] = q
R["pool_dist"] = {f"{c}/{k}": {"ce_q": [round(float(d["ce"].quantile(v)), 2) for v in q]} for (c, k), d in T.filter(pl.col("cls").is_not_null()).group_by("country", "cls")}

P = pl.read_parquet(CE / "v3_encoder/out/d01_pairs.parquet")
fit = P.filter(~pl.col("is_val3") & ~pl.col("j_in_stage4_active") & ~pl.col("j_in_holdout_active") & ((pl.col("rank") <= 6) | (pl.col("y") == 1)))
A = fit.gather(np.sort(rng.choice(fit.height, N_ANCHOR, replace=False))).select("i", "j", "y", "country")
A.write_parquet(OUT / "anchor.parquet")
R["anchor"] = {"pairs": A.height, "pos_share": round(float(A["y"].mean()), 4), "by_country": dict(A.group_by("country").len().iter_rows())}
(OUT / "p01.json").write_text(json.dumps(R, indent=1)); print(json.dumps(R, indent=1))
