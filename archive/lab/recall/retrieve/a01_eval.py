"""Dense top-k vs the sparse candidate set.
usage: a01_eval.py train  -> out/a01_train.json (HOLDOUT + HEAD: missed-true recovery, new pairs/S1, F0.5 ceilings, grid)
       a01_eval.py test   -> out/a01_test.json  (new pairs/S1 per country on the same grid)
       a01_eval.py write K FLOOR -> out/pairs/{holdout,head,test}_k{K}_f{FLOOR}.parquet (new pairs only)
Sparse candidate set = the split's cache pairs (train cache 58f0... for HOLDOUT/HEAD, test cache 3ee4... for TEST).
dense_rank = 1-based rank of j in the S1's dense list for that source (computed before removing sparse candidates)."""
import json, os, resource, sys
import numpy as np, polars as pl
ROOT = "/Users/adityarohilla/Documents/amazon-ml-2026"
D = os.path.join(os.path.dirname(os.path.abspath(__file__)), "out")
CACHE = {"train": f"{ROOT}/cache/58f0a60c5c264fdd", "test": f"{ROOT}/cache/3ee42f05d06efbcb"}
KS = [1, 2, 3, 5, 10, 20]
FLOORS = [0.0, 0.80, 0.85, 0.88, 0.90, 0.92, 0.94, 0.96]
COUNTRIES = {"train": ["us", "india"], "test": ["us", "india", "france"]}


def f05(n_pred, n_true, tp):
    n_pred, n_true, tp = (np.asarray(x, np.float64) for x in (n_pred, n_true, tp))
    with np.errstate(divide="ignore", invalid="ignore"):
        p, r = tp / n_pred, tp / n_true
        f = np.where(tp > 0, 1.25 * p * r / (0.25 * p + r), 0.0)
    return np.where(n_true == 0, (n_pred == 0).astype(np.float64), f)


def dense(split, country, rows=None):
    parts = []
    for s in (2, 3):
        z = np.load(f"{D}/knn/{split}_{country}_s{s}_fwd.npz")
        q, t, c = z["q_row"], z["t_row"], z["cos"]
        if rows is not None:
            m = np.isin(q, rows); q, t, c = q[m], t[m], c[m]
        k = t.shape[1]
        parts.append(pl.DataFrame({"i": np.repeat(q, k), "j": t.ravel(), "cos": c.ravel().astype(np.float32),
                                   "rank": np.tile(np.arange(1, k + 1, dtype=np.uint8), len(q)),
                                   "src": np.full(len(q) * k, s, np.uint8)}))
    return pl.concat(parts)


def cand_pairs(split, rows):
    return (pl.scan_parquet(f"{CACHE[split]}/pairs.parquet").select("i", "j")
            .filter(pl.col("i").is_in(pl.Series(rows))).collect())


def new_pairs(d, cand):
    return d.join(cand.with_columns(pl.lit(True).alias("in_cand")), on=["i", "j"], how="left").with_columns(
        pl.col("in_cand").fill_null(False))


def train_eval():
    truth = pl.read_parquet(f"{ROOT}/v8_lab/ce/setup/out/truth_train.parquet").select("i", "j")
    recs = pl.read_parquet(f"{CACHE['train']}/recs.parquet", columns=["row", "country_key"])
    keeps = pl.read_parquet(f"{ROOT}/v8_lab/ce/v2_build/out/holdout_keeps.parquet", columns=["i", "j", "ACNRST_3s_v2ce"])
    ho_nt = pl.read_parquet(f"{ROOT}/v8_lab/ce/setup/out/holdout_s1.parquet").select("i", pl.col("n_true").alias("nt_ref"))
    R = {}
    for name in ("holdout", "head"):
        rows = np.load(f"{D}/sets/{name}_rows.npy")
        s1 = pl.DataFrame({"i": rows}).join(recs.rename({"row": "i", "country_key": "country"}), on="i")
        tr = truth.filter(pl.col("i").is_in(pl.Series(rows)))
        cand = cand_pairs("train", rows)
        tr = tr.join(cand.with_columns(pl.lit(True).alias("in_cand")), on=["i", "j"], how="left").with_columns(
            pl.col("in_cand").fill_null(False))
        per = s1.join(tr.group_by("i").agg(pl.len().alias("n_true"), pl.col("in_cand").sum().alias("found")),
                      on="i", how="left").fill_null(0)
        if name == "holdout":
            chk = per.join(ho_nt, on="i")
            assert chk.height == 100_000 and (chk["n_true"] == chk["nt_ref"]).all()
        d = pl.concat([dense("train", c, rows) for c in COUNTRIES["train"]])
        d = new_pairs(d, cand).join(tr.select("i", "j", pl.lit(1, pl.Int8).alias("y")), on=["i", "j"], how="left").with_columns(
            pl.col("y").fill_null(0))
        # sanity: dense lists found for every S1 of the set, 20 per source
        assert d.group_by("i").len()["len"].min() == 40 and d["i"].n_unique() == len(rows)
        missed = tr.filter(~pl.col("in_cand"))
        dm = missed.join(d.select("i", "j", "rank", "cos", "src"), on=["i", "j"], how="left")
        out = {"n_s1": len(rows), "n_true": tr.height, "n_missed": missed.height,
               "missed_rate": round(missed.height / tr.height, 5),
               "dense_cand_overlap_top20": round(float(d["in_cand"].mean()), 4),
               "sparse_true_recall_of_dense_top20": round(d.filter(pl.col("y") == 1)["in_cand"].mean(), 4)}
        by_c = {}
        for c in ["all"] + COUNTRIES["train"]:
            sc = s1 if c == "all" else s1.filter(pl.col("country") == c)
            ids = sc["i"]
            pc = per.filter(pl.col("i").is_in(ids)); dc = d.filter(pl.col("i").is_in(ids))
            mc = dm.filter(pl.col("i").is_in(ids))
            nt, fo = pc["n_true"].to_numpy(), pc["found"].to_numpy()
            base_perfect = f05(fo, nt, fo)
            row = {"n_s1": sc.height, "n_true": int(nt.sum()), "n_missed": mc.height,
                   "missed_rate": round(mc.height / max(1, int(nt.sum())), 5),
                   "perfect_matcher_f05_sparse": round(float(base_perfect.mean()), 6)}
            if name == "holdout":
                kc = keeps.filter(pl.col("i").is_in(ids)).join(tr.select("i", "j", pl.lit(True).alias("t")), on=["i", "j"], how="left")
                kc = kc.with_columns(pl.col("t").fill_null(False))
                agg = pc.select("i", "n_true").join(kc.group_by("i").agg(pl.col("ACNRST_3s_v2ce").sum().alias("np"),
                                                                        (pl.col("ACNRST_3s_v2ce") & pl.col("t")).sum().alias("tp")),
                                                    on="i", how="left").fill_null(0)
                row["best_file_f05"] = round(float(f05(agg["np"], agg["n_true"], agg["tp"]).mean()), 6)
            ks = {}
            for k in KS:
                sel = dc.filter((pl.col("rank") <= k) & ~pl.col("in_cand"))
                rec = sel.filter(pl.col("y") == 1).group_by("i").len("r")
                p2 = pc.join(rec, on="i", how="left").fill_null(0)
                fn = p2["found"].to_numpy() + p2["r"].to_numpy()
                e = {"new_pairs": sel.height, "new_per_s1": round(sel.height / sc.height, 3),
                     "recovered": int(p2["r"].sum()), "recovered_share_of_missed": round(int(p2["r"].sum()) / max(1, mc.height), 4),
                     "new_true_rate": round(int(p2["r"].sum()) / max(1, sel.height), 5),
                     "perfect_matcher_gain": round(float((f05(fn, nt, fn) - base_perfect).mean()), 6)}
                if name == "holdout":
                    a2 = agg.join(rec, on="i", how="left").fill_null(0)
                    e["best_file_f05_plus_recovered"] = round(float(f05(a2["np"] + a2["r"], a2["n_true"], a2["tp"] + a2["r"]).mean()), 6)
                    e["best_file_gain_ceiling"] = round(e["best_file_f05_plus_recovered"] - row["best_file_f05"], 6)
                ks[str(k)] = e
            row["by_k"] = ks
            grid = []
            for k in KS:
                for fl in FLOORS:
                    sel = dc.filter((pl.col("rank") <= k) & ~pl.col("in_cand") & (pl.col("cos") >= fl))
                    r = int(sel["y"].sum())
                    grid.append({"k": k, "floor": fl, "new_per_s1": round(sel.height / sc.height, 3), "recovered": r,
                                 "recovered_share": round(r / max(1, mc.height), 4)})
            row["grid"] = grid
            # where the missed trues sit in the dense lists (per source list of the true's own source)
            row["missed_dense_rank_hist"] = {str(k): int((mc["rank"] <= k).sum()) for k in KS}
            row["missed_not_in_top20"] = int(mc["rank"].is_null().sum())
            mcos = mc.drop_nulls("cos")["cos"]
            row["missed_in_top20_cos_p10_p50"] = [round(float(mcos.quantile(q)), 4) for q in (0.1, 0.5)] if len(mcos) else []
            by_c[c] = row
        out["by_country"] = by_c
        R[name] = out
        print(name, json.dumps({c: {k: v for k, v in r.items() if k not in ("grid", "by_k")} for c, r in by_c.items()}), flush=True)
    R["peak_rss_gb"] = round(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 2**30, 2)
    json.dump(R, open(f"{D}/a01_train.json", "w"), indent=1)


def test_eval():
    recs = pl.read_parquet(f"{CACHE['test']}/recs.parquet", columns=["row", "src", "country_key"])
    R = {}
    for c in COUNTRIES["test"]:
        rows = recs.filter((pl.col("country_key") == c) & (pl.col("src") == 1))["row"].to_numpy()
        d = new_pairs(dense("test", c), cand_pairs("test", rows))
        assert d["i"].n_unique() == len(rows)
        grid = []
        for k in KS:
            for fl in FLOORS:
                n = d.filter((pl.col("rank") <= k) & ~pl.col("in_cand") & (pl.col("cos") >= fl)).height
                grid.append({"k": k, "floor": fl, "new_per_s1": round(n / len(rows), 3), "new_pairs": n})
        R[c] = {"n_s1": len(rows), "dense_cand_overlap_top20": round(float(d["in_cand"].mean()), 4), "grid": grid}
        print(c, R[c]["n_s1"], R[c]["dense_cand_overlap_top20"], flush=True)
        del d
    R["peak_rss_gb"] = round(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 2**30, 2)
    json.dump(R, open(f"{D}/a01_test.json", "w"), indent=1)


def write(k, fl):
    os.makedirs(f"{D}/pairs", exist_ok=True)
    truth = pl.read_parquet(f"{ROOT}/v8_lab/ce/setup/out/truth_train.parquet").select("i", "j")
    recs = {s: pl.read_parquet(f"{CACHE[s]}/recs.parquet", columns=["row", "src", "country_key"]) for s in CACHE}
    W = {}
    for name in ("holdout", "head", "test"):
        split = "test" if name == "test" else "train"
        parts = []
        for c in COUNTRIES[split]:
            if name == "test":
                rows = recs["test"].filter((pl.col("country_key") == c) & (pl.col("src") == 1))["row"].to_numpy()
            else:
                allr = np.load(f"{D}/sets/{name}_rows.npy")
                cr = recs["train"].filter(pl.col("country_key") == c)["row"].to_numpy()
                rows = allr[np.isin(allr, cr)]
            d = new_pairs(dense(split, c, rows), cand_pairs(split, rows))
            d = d.filter((pl.col("rank") <= k) & ~pl.col("in_cand") & (pl.col("cos") >= fl)).drop("in_cand")
            parts.append(d.with_columns(pl.lit(c).alias("country")))
        d = pl.concat(parts).select(pl.col("i").alias("s1_row"), pl.col("j").alias("j_row"), pl.col("cos").alias("dense_cos"),
                                    pl.col("rank").alias("dense_rank"), pl.col("src").alias("source"), "country")
        if name != "test":
            d = d.join(truth.select(pl.col("i").alias("s1_row"), pl.col("j").alias("j_row"), pl.lit(1, pl.Int8).alias("y")),
                       on=["s1_row", "j_row"], how="left").with_columns(pl.col("y").fill_null(0))
        assert d.select(pl.struct("s1_row", "j_row").n_unique()).item() == d.height
        p = f"{D}/pairs/{name}_k{k}_f{fl:.2f}.parquet"
        d.sort("s1_row", "source", "dense_rank").write_parquet(p)
        W[name] = {"path": p, "pairs": d.height, "s1": d["s1_row"].n_unique()} | ({"trues": int(d["y"].sum())} if name != "test" else {})
        print(name, W[name], flush=True)
    W["peak_rss_gb"] = round(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 2**30, 2)
    json.dump(W, open(f"{D}/pairs/w_k{k}_f{fl:.2f}.json", "w"), indent=1)


if __name__ == "__main__":
    if sys.argv[1] == "train": train_eval()
    elif sys.argv[1] == "test": test_eval()
    else: write(int(sys.argv[2]), float(sys.argv[3]))
