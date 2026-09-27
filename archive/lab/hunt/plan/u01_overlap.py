"""Overlap of the India sparse k30 add channel (r6_blocking_recall s19/s20: forward ranks 11..30, house-compatible,
j not blocked, CE v2 logit > T, one S1 per j) with the dense new-pair lists the recall head scores
(recall/retrieve w02 rule m1_r1_cap10: forward new_rank <= 1 per source OR reverse rev_rank <= 1).
For every TRUE sparse add (the pairs that carry s20's +0.00089 India dF) this recomputes, on the TRAIN pool:
  new_rank = 1 + #(records of j's source with cos > cos(i,j)) - #(i's sparse cache candidates of that source with cos > cos(i,j))
  rev1     = the reverse kNN top-1 S1 of j is i (recall/retrieve/out/knn/train_india_s{src}_rev.npz)
Embeddings: recall/retrieve/out/emb/train_india_s{1,2,3}.npy (fp16, unit norm, rows ascending); cosine in fp32 on CPU.
usage (repo venv): u01_overlap.py -> out/u01_overlap.json   (RSS watchdog aborts at 4.5 GB)"""
import json, os, resource, sys, threading, time
from pathlib import Path
import numpy as np, polars as pl

ROOT = Path("/Users/adityarohilla/Documents/amazon-ml-2026"); E = ROOT / "v8_lab/recall/retrieve/out"
R6 = ROOT / "v8_lab/day1b/r6_blocking_recall/out"; OUT = Path(__file__).parent / "out"; OUT.mkdir(exist_ok=True)


def guard(lim=4.5e9):
    def run():
        while True:
            if resource.getrusage(resource.RUSAGE_SELF).ru_maxrss > lim:
                print("RSS guard abort", flush=True); os._exit(3)
            time.sleep(0.5)
    threading.Thread(target=run, daemon=True).start()


guard(); T0 = time.time(); Res = {}
ce = pl.read_parquet(R6 / "s19_ce.parquet")
src = pl.read_parquet(R6 / "s18_india_top30.parquet", columns=["i", "j", "src"])
for T in (4.0, 6.0):
    a = (ce.filter(~pl.col("blocked") & (pl.col("ce_logit") > T)).sort("ce_logit", descending=True).unique("j", keep="first")
         .join(src, on=["i", "j"], how="left"))
    assert a["src"].null_count() == 0
    Res[f"T{T:g}"] = {"adds": a.height, "true": int(a["y"].sum())}
    if T == 4.0:
        A = a.filter(pl.col("y") == 1)
cand = (pl.scan_parquet(ROOT / "cache/58f0a60c5c264fdd/pairs.parquet").select("i", "j")
        .join(A.lazy().select("i").unique(), on="i", how="semi").collect())
emb = {s: (np.load(E / f"emb/train_india_s{s}_rows.npy"), np.load(E / f"emb/train_india_s{s}.npy", mmap_mode="r")) for s in (1, 2, 3)}


def vec(s, rows):
    r, e = emb[s]; k = np.searchsorted(r, rows); assert (r[k] == rows).all(); return np.asarray(e[k], np.float32), k


out = []
for s in (2, 3):
    a = A.filter(pl.col("src") == s)
    if a.height == 0:
        continue
    qi, qj = a["i"].to_numpy(), a["j"].to_numpy()
    Q, _ = vec(1, qi); J, kj = vec(s, qj)
    c_ij = (Q * J).sum(1)
    # all records of source s with cos > cos(i, j), blocked matmul
    r, e = emb[s]; greater = np.zeros(len(qi), np.int64)
    for b in range(0, len(r), 200_000):
        B = np.asarray(e[b:b + 200_000], np.float32)
        greater += (B @ Q.T > c_ij[None, :]).sum(0)
    # minus i's sparse candidates of source s with cos > cos(i, j)
    cs = cand.filter(pl.col("i").is_in(pl.Series(qi).implode()) & pl.col("j").is_in(pl.Series(r).implode()))
    cs = cs.join(pl.DataFrame({"i": qi, "c_ij": c_ij}), on="i")
    ci = cs["i"].to_numpy(); CI, _ = vec(1, ci); CJ, _ = vec(s, cs["j"].to_numpy())
    cg = pl.DataFrame({"i": ci, "g": ((CI * CJ).sum(1) > cs["c_ij"].to_numpy()).astype(np.int64)}).group_by("i").agg(pl.col("g").sum())
    new_rank = 1 + greater - pl.DataFrame({"i": qi}).join(cg, on="i", how="left").fill_null(0)["g"].to_numpy()
    z = np.load(E / f"knn/train_india_s{s}_rev.npz"); assert (z["q_row"] == r).all()
    rev1 = z["t_row"][kj, 0] == qi
    out.append(pl.DataFrame({"i": qi, "j": qj, "src": s, "cos": c_ij, "new_rank": new_rank, "rev1": rev1}))
    print(f"src {s}: {len(qi)} true adds, {time.time() - T0:.0f}s", flush=True)
O = pl.concat(out)
n = O.height
for m in (1, 2, 3, 5, 10):
    Res[f"dense_m{m}_r1_covers"] = int(((O["new_rank"] <= m) | O["rev1"]).sum())
Res.update(true_adds_T4=n, fwd_m1=int((O["new_rank"] <= 1).sum()), rev1=int(O["rev1"].sum()),
           new_rank_quantiles={q: float(np.quantile(O["new_rank"].to_numpy(), q / 100)) for q in (10, 25, 50, 75, 90)},
           share_not_in_m1_r1=round(1 - Res["dense_m1_r1_covers"] / n, 4), share_not_in_m2_r1=round(1 - Res["dense_m2_r1_covers"] / n, 4),
           s20_india_dF_T4=0.00089, incremental_india_dF_over_dense_m1_r1_est=round(0.00089 * (1 - Res["dense_m1_r1_covers"] / n), 6),
           seconds=round(time.time() - T0), peak_rss_gb=round(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1e9, 2))
O.write_parquet(OUT / "u01_true_adds_dense.parquet")
(OUT / "u01_overlap.json").write_text(json.dumps(Res, indent=1)); print(json.dumps(Res, indent=1))
