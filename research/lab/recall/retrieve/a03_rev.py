"""Forward new-rank rule + reverse pass (each S2/S3 record's top-3 S1 among ALL S1 of the country).
usage: a03_rev.py -> out/a03_rev.json: HOLDOUT/HEAD recovered missed trues and new pairs/S1, TEST new pairs/S1, for
forward 'new_rank <= m' (rank among the S1's non-candidate dense neighbours of that source), reverse 'rev_rank <= r'
(not a sparse candidate), and their union."""
import json, resource, sys
import numpy as np, polars as pl
sys.path.insert(0, __file__.rsplit("/", 1)[0])
from a01_eval import ROOT, D, CACHE, dense, cand_pairs, new_pairs, COUNTRIES, f05


def rev(split, country, rows):
    parts = []
    for s in (2, 3):
        z = np.load(f"{D}/knn/{split}_{country}_s{s}_rev.npz")
        j, i, c = z["q_row"], z["t_row"], z["cos"]
        k = i.shape[1]
        df = pl.DataFrame({"i": i.ravel(), "j": np.repeat(j, k), "rcos": c.ravel().astype(np.float32),
                           "rev_rank": np.tile(np.arange(1, k + 1, dtype=np.uint8), len(j)), "src": np.full(len(j) * k, s, np.uint8)})
        parts.append(df.filter(pl.col("i").is_in(pl.Series(rows).implode())))
    return pl.concat(parts)


def pairs(split, country, rows):
    """Forward (new_rank) and reverse candidates outside the sparse set, joined on (i, j)."""
    cand = cand_pairs(split, rows)
    f = new_pairs(dense(split, country, rows), cand).filter(~pl.col("in_cand")).drop("in_cand")
    f = f.with_columns(pl.col("rank").rank("ordinal").over("i", "src").cast(pl.UInt8).alias("new_rank"))
    r = rev(split, country, rows).join(cand, on=["i", "j"], how="anti")
    return f.join(r, on=["i", "j", "src"], how="full", coalesce=True)


RULES = [("fwd", m, 0) for m in (1, 2, 3)] + [("rev", 0, r) for r in (1, 2, 3)] + \
        [("union", m, r) for m in (1, 2) for r in (1, 2, 3)]


def sel(d, kind, m, r):
    fw = pl.col("new_rank").is_not_null() & (pl.col("new_rank") <= m)
    rv = pl.col("rev_rank").is_not_null() & (pl.col("rev_rank") <= r)
    return d.filter(fw if kind == "fwd" else rv if kind == "rev" else fw | rv)



def main():
    R = {}
    truth = pl.read_parquet(f"{ROOT}/v8_lab/ce/setup/out/truth_train.parquet").select("i", "j", pl.lit(1, pl.Int8).alias("y"))
    keeps = pl.read_parquet(f"{ROOT}/v8_lab/ce/v2_build/out/holdout_keeps.parquet", columns=["i", "j", "ACNRST_3s_v2ce"])
    for name in ("holdout", "head"):
        rows = np.load(f"{D}/sets/{name}_rows.npy")
        d = pl.concat([pairs("train", c, rows) for c in COUNTRIES["train"]])
        d = d.join(truth, on=["i", "j"], how="left").with_columns(pl.col("y").fill_null(0))
        tr = truth.filter(pl.col("i").is_in(pl.Series(rows).implode()))
        agg = pl.DataFrame({"i": rows}).join(tr.group_by("i").len("n_true"), on="i", how="left").fill_null(0)
        if name == "holdout":  # best file's holdout keeps (in-candidate decisions unchanged)
            kc = keeps.join(tr, on=["i", "j"], how="left").with_columns(pl.col("y").fill_null(0))
            agg = agg.join(kc.group_by("i").agg(pl.col("ACNRST_3s_v2ce").sum().alias("np"),
                                                (pl.col("ACNRST_3s_v2ce") & (pl.col("y") == 1)).sum().alias("tp")),
                           on="i", how="left").fill_null(0)
            base = float(f05(agg["np"], agg["n_true"], agg["tp"]).mean())
            print("holdout best file F0.5", round(base, 6), flush=True)
        out = []
        for kind, m, r in RULES:
            s = sel(d, kind, m, r)
            per = s.group_by("i").len()["len"]
            out.append({"rule": f"{kind} m{m} r{r}", "new_per_s1": round(s.height / len(rows), 3), "recovered": int(s["y"].sum()),
                        "max_per_s1": int(per.max()), "p99_per_s1": float(per.quantile(0.99))})
            if name == "holdout":
                a2 = agg.join(s.filter(pl.col("y") == 1).group_by("i").len("r"), on="i", how="left").fill_null(0)
                out[-1]["best_file_gain_ceiling"] = round(float(f05(a2["np"] + a2["r"], a2["n_true"], a2["tp"] + a2["r"]).mean()) - base, 6)
            print(name, out[-1], flush=True)
        R[name] = out
    recs = pl.read_parquet(f"{CACHE['test']}/recs.parquet", columns=["row", "src", "country_key"])
    T = {}
    for c in COUNTRIES["test"]:
        rows = recs.filter((pl.col("country_key") == c) & (pl.col("src") == 1))["row"].to_numpy()
        d = pairs("test", c, rows)
        T[c] = {"n_s1": len(rows)}
        for kind, m, r in RULES:
            s = sel(d, kind, m, r)
            per = s.group_by("i").len()["len"]
            T[c][f"{kind} m{m} r{r}"] = {"new_pairs": s.height, "new_per_s1": round(s.height / len(rows), 3),
                                          "max_per_s1": int(per.max()), "p99_per_s1": float(per.quantile(0.99))}
        print("test", c, json.dumps(T[c]), flush=True)
        del d
    T["all"] = {k: round(sum(T[c][k]["new_pairs"] for c in COUNTRIES["test"]) / sum(T[c]["n_s1"] for c in COUNTRIES["test"]), 3)
                for k in T["us"] if k != "n_s1"}
    print("test all new_per_s1", T["all"], flush=True)
    R["test"] = T
    R["peak_rss_gb"] = round(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 2**30, 2)
    json.dump(R, open(f"{D}/a03_rev.json", "w"), indent=1)


if __name__ == "__main__":
    main()
