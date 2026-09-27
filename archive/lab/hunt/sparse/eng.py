"""Wide sparse blocking search for one frame (split, country, source), rebuilt from src/candidates.py key code and
bounded in memory. Same keys (_keys_chunk with the frame's address-word df), same IDF weights (df 2..max_df,
log N - log df, float32), same L2 normalisation (scipy, column-sorted rows), same cosine.
Memory: keys go to disk partitioned by hash bucket (df per bucket), weighted entries go to disk partitioned by row
chunk, then target chunks are searched one at a time against the S1 matrix.
Outputs (global row ids) under out/:
  fwd_{tag}.parquet  i, j, cos, rank          forward top-KF targets of every query S1 (train: holdout S1; test: all S1)
  rev_{tag}.parquet  i, j, cos, rev_rank      reverse top-KR S1 (among ALL S1 of the frame) of every target
  pc_{tag}.parquet   i, j, cos[, y]           cosine of listed pairs (train: holdout truth + cached pairs;
                                               test: cached pairs of a 20k-S1 sample) - reproduction check
usage: eng.py SPLIT COUNTRY SRC"""
import json, os, shutil, subprocess, sys, threading, time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
import numpy as np, polars as pl, scipy.sparse as sp

ROOT = Path("/Users/adityarohilla/Documents/amazon-ml-2026")
sys.path.insert(0, str(ROOT / "code/business_entity_resolution"))
from src import candidates  # noqa: E402

CACHE = {"train": ROOT / "cache/58f0a60c5c264fdd", "test": ROOT / "cache/3ee42f05d06efbcb"}
HERE = Path(__file__).parent
split, country, s = sys.argv[1], sys.argv[2], int(sys.argv[3])
KF, KR, NB, KCH, RC = 30, 10, 16, 150_000, 400_000
MAXDF = int(os.environ.get("MAXDF", 3000))
tag = f"{split}_{country}_s{s}" + ("" if MAXDF == 3000 else f"_md{MAXDF}")
SC, TH = int(os.environ.get("SC", 2000)), int(os.environ.get("TH", 7))
CAP_GB = 5.0
OUT, TMP = HERE / "out", HERE / "tmp" / tag
OUT.mkdir(exist_ok=True)
shutil.rmtree(TMP, ignore_errors=True)
t0 = time.time()
peak = [0.0]


def rss_gb():
    return int(subprocess.run(["ps", "-o", "rss=", "-p", str(os.getpid())], capture_output=True, text=True).stdout) / 2**20


def watchdog():
    while True:
        g = rss_gb()
        peak[0] = max(peak[0], g)
        if g > CAP_GB:
            print(f"WATCHDOG: rss {g:.2f} GB > {CAP_GB} GB, abort", flush=True)
            os._exit(99)
        time.sleep(1)


threading.Thread(target=watchdog, daemon=True).start()
log = lambda *a: print(tag, *a, f"{time.time() - t0:.0f}s rss {rss_gb():.2f} peak {peak[0]:.2f}", flush=True)

# ---- frame: S1 rows first, then source-s rows (as candidates.generate)
lf = pl.scan_parquet(CACHE[split] / "recs.parquet").select("row", "src", "country_key", "core", "addr") \
    .filter(pl.col("country_key") == country)
q_rows = lf.filter(pl.col("src") == 1).select("row", "core", "addr").collect()
t_rows = lf.filter(pl.col("src") == s).select("row", "core", "addr").collect()
nq, N = q_rows.height, q_rows.height + t_rows.height
glob = np.concatenate([q_rows["row"].to_numpy(), t_rows["row"].to_numpy()]).astype(np.uint32)
local = pl.concat([q_rows, t_rows]).with_columns(pl.int_range(N, dtype=pl.UInt32).alias("row"))
del q_rows, t_rows
log("frame", nq, "+", N - nq)

# ---- A: keys, partitioned by hash bucket
wdf = pl.concat([candidates._address_words(candidates._tokens(candidates.tnorm(local.slice(o, KCH)), "addr")).unique()
                 for o in range(0, N, KCH)]).group_by("h").agg(pl.len().alias("wdf"))
n_keys = 0
for ci, o in enumerate(range(0, N, KCH)):
    k = candidates._keys_chunk(local.slice(o, KCH), wdf).with_columns((pl.col("h") % NB).alias("b"))
    n_keys += k.height
    for (b,), part in k.partition_by("b", as_dict=True).items():
        d = TMP / "keys" / f"b{b:02d}"; d.mkdir(parents=True, exist_ok=True)
        part.drop("b").write_parquet(d / f"c{ci:03d}.parquet")
    del k
del local, wdf
log("keys", n_keys)

# ---- B: df per bucket -> weights -> entries partitioned by row chunk
vocab = []
for b in range(NB):
    k = pl.read_parquet(TMP / "keys" / f"b{b:02d}")
    v = (k.group_by("h").agg(pl.len().alias("df")).filter((pl.col("df") >= 2) & (pl.col("df") <= MAXDF))
         .with_columns((np.log(N) - pl.col("df").log()).cast(pl.Float32).alias("w")))
    vocab.append(v["h"].to_numpy())
    e = k.join(v.select("h", "w"), on="h").with_columns((pl.col("row") // RC).alias("rc"))
    for (rc,), part in e.partition_by("rc", as_dict=True).items():
        d = TMP / "ent" / f"r{rc:03d}"; d.mkdir(parents=True, exist_ok=True)
        part.drop("rc").write_parquet(d / f"b{b:02d}.parquet")
    del k, v, e
shutil.rmtree(TMP / "keys")
vocab = np.sort(np.concatenate(vocab))
V = len(vocab)
log("vocab", V)


def chunk_matrix(rc):
    """L2-normalised rows [rc*RC, ...) exactly as key_matrix: entries sorted by (row, col), scipy norms."""
    e = pl.read_parquet(TMP / "ent" / f"r{rc:03d}")
    col = np.searchsorted(vocab, e["h"].to_numpy()).astype(np.int32)
    lo, hi = rc * RC, min((rc + 1) * RC, N)
    r = e["row"].to_numpy().astype(np.int64) - lo
    o = np.lexsort((col, r))
    m = sp.csr_matrix((e["w"].to_numpy()[o], (r[o], col[o])), shape=(hi - lo, V), dtype=np.float32)
    del e, col, r, o
    norms = np.sqrt(np.asarray(m.multiply(m).sum(axis=1)).ravel())
    norms[norms == 0] = 1.0
    return sp.diags((1.0 / norms).astype(np.float32)).dot(m).tocsr()


n_rc = (N + RC - 1) // RC
first_t = nq // RC
S = sp.vstack([chunk_matrix(rc)[: max(0, min(RC, nq - rc * RC))] for rc in range(first_t + 1)]).tocsr()
assert S.shape[0] == nq
St = S.T.tocsr()
log("S1 matrix nnz", S.nnz)

# ---- queries and scored pairs
if split == "train":
    hold = pl.read_parquet(ROOT / "v8_lab/ce/setup/out/holdout_s1.parquet").filter(pl.col("country") == country)["i"].to_numpy()
    qg = np.intersect1d(glob[:nq], hold)
else:
    qg = glob[:nq]
qidx = np.searchsorted(glob[:nq], qg)  # S1 rows are sorted by global row within the frame
assert (glob[:nq][qidx] == qg).all() and (np.diff(glob[:nq].astype(np.int64)) > 0).all()
Q = S if split == "test" else S[qidx]
nQ = len(qidx)
tglob = glob[nq:]
tord = np.argsort(tglob)
lpc = pl.scan_parquet(CACHE[split] / "pairs.parquet").filter(pl.col("jsrc") == s).select("i", "j", "score")
if split == "train":
    truth = pl.read_parquet(ROOT / "v8_lab/ce/setup/out/truth_train.parquet").with_columns(pl.lit(1, pl.Int8).alias("y"))
    cached = lpc.filter(pl.col("i").is_in(pl.Series(qg).implode())).collect()
    PC = cached.join(truth, on=["i", "j"], how="full", coalesce=True).filter(pl.col("i").is_in(pl.Series(qg).implode()))
    PC = PC.with_columns(pl.col("y").fill_null(0))
else:
    samp = np.random.default_rng(0).choice(qg, min(20_000, nQ), replace=False)
    PC = lpc.filter(pl.col("i").is_in(pl.Series(samp).implode())).collect()
pj = np.searchsorted(tglob[tord], PC["j"].to_numpy())
pj = np.minimum(pj, len(tglob) - 1)
in_frame = tglob[tord][pj] == PC["j"].to_numpy()
PC = PC.filter(pl.Series(in_frame))
pc_t = tord[pj[in_frame]]  # local target index (0-based over targets)
pc_q = np.searchsorted(glob[:nq], PC["i"].to_numpy())
pc_cos = np.zeros(PC.height, dtype=np.float32)
log("queries", nQ, "scored pairs", PC.height)

# ---- C: search target chunks
topv = np.full((nQ, KF), -1.0, dtype=np.float32)
topj = np.zeros((nQ, KF), dtype=np.int64)
rev = []


def ranks(r):
    starts = np.r_[0, np.flatnonzero(np.diff(r)) + 1]
    return np.arange(len(r)) - np.repeat(starts, np.diff(np.r_[starts, len(r)]))


for rc in range(first_t, n_rc):
    lo = max(rc * RC, nq)
    T = chunk_matrix(rc)[lo - rc * RC:]
    off = lo - nq
    tt = T.T.tocsr()

    def fwd(a):
        b = min(a + SC, nQ)
        r, c, v = candidates._top_k((Q[a:b] @ tt).tocsr(), KF)
        nv = np.full((b - a, KF), -1.0, dtype=np.float32); nj = np.zeros((b - a, KF), dtype=np.int64)
        rk = ranks(r); nv[r, rk] = v; nj[r, rk] = c + off
        cv = np.concatenate([topv[a:b], nv], 1); cj = np.concatenate([topj[a:b], nj], 1)
        o = np.argsort(-cv, axis=1, kind="stable")[:, :KF]
        topv[a:b] = np.take_along_axis(cv, o, 1); topj[a:b] = np.take_along_axis(cj, o, 1)

    def bwd(a):
        r, c, v = candidates._top_k((T[a:a + SC] @ St).tocsr(), KR)
        return (r + a + off).astype(np.uint32), c.astype(np.uint32), v, (ranks(r) + 1).astype(np.uint8)

    with ThreadPoolExecutor(TH) as ex:
        list(ex.map(fwd, range(0, nQ, SC)))
        rev += list(ex.map(bwd, range(0, T.shape[0], SC)))
    m = (pc_t >= off) & (pc_t < off + T.shape[0])
    if m.any():
        pc_cos[m] = np.asarray(S[pc_q[m]].multiply(T[pc_t[m] - off]).sum(axis=1)).ravel()
    del T, tt
    log("target chunk", rc, "done")
shutil.rmtree(TMP)

# ---- write
ok = topv > 0
rk = np.broadcast_to(np.arange(1, KF + 1, dtype=np.uint8), topv.shape)
pl.DataFrame({"i": np.repeat(qg, KF)[ok.ravel()], "j": tglob[topj[ok]], "cos": topv[ok], "rank": rk[ok]}) \
    .write_parquet(OUT / f"fwd_{tag}.parquet")
rj, ri, rv, rr = (np.concatenate(x) for x in zip(*rev))
R = pl.DataFrame({"i": glob[ri], "j": tglob[rj], "cos": rv, "rev_rank": rr})
R.write_parquet(OUT / f"rev_{tag}.parquet")
PC.with_columns(pl.Series("cos", pc_cos)).write_parquet(OUT / f"pc_{tag}.parquet")
has, sc = PC["score"].is_not_null().to_numpy(), PC["score"].fill_null(0).to_numpy()
res = {"tag": tag, "nq_frame": nq, "targets": N - nq, "queries": nQ, "keys": n_keys, "vocab": V, "S_nnz": S.nnz,
       "fwd_rows": int(ok.sum()), "rev_rows": R.height, "pc_pairs": PC.height,
       "pc_cached_max_abs_diff": float(np.abs(pc_cos[has] - sc[has]).max()) if has.any() else None,
       "seconds": round(time.time() - t0), "peak_rss_gb": round(peak[0], 2)}
json.dump(res, open(OUT / f"eng_{tag}.json", "w"), indent=1)
log(json.dumps(res))
