"""Score pairs with a cross-encoder (default: v3, model/) -> {CE_OUTDIR}/{DS}_ce.parquet (i, j, ce_logit); v2's i01_score.py
with one more dataset: val3 (VAL3 pairs of this d01); val2 = v2's VAL2 pairs, ceval = v1's CE-VAL pairs. Rows keep the source order, so HOLDOUT/STAGE4/TEST rows equal v1's infer/out files.
ACTIVE = v7 production prob >= 0.002 (HOLDOUT: that rule in either the uncapped or the capped regime).
The work list out/{ds}/pairs.parquet (active pairs in source order) is written once; scores go to out/{ds}/chunk_NNNN.parquet,
CH pairs each, written atomically, so a crash loses at most one chunk and a restart skips finished chunks.
Inside a chunk: pairs sorted by token length, batches of BS, padded to a multiple of 32, bf16 autocast, no grad.
The next chunk is tokenised in a helper thread while the current one runs on the GPU.
usage: i01_score.py [holdout stage4 test]   env CE_LIMIT=N scores only the first N pairs of each dataset into out/smoke/."""
import itertools, json, math, os, resource, sys, time
os.environ["HF_HUB_OFFLINE"] = "1"
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
import numpy as np, polars as pl, torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer

HERE = Path(__file__).parent; CE = HERE.parent; SET = CE / "setup/out"
SRC = {"holdout": (SET / "holdout_pairs.parquet", "text_train", pl.col("active") | pl.col("active_cap")),
       "stage4": (SET / "stage4_train.parquet", "text_train", pl.col("active")),
       "test": (SET / "test_pairs.parquet", "text_test", pl.col("active")),
       "val3": (HERE / "out/d01_pairs.parquet", "text_train", pl.col("is_val3")),
       "val2": (CE / "v2_encoder/out/d01_pairs.parquet", "text_train", pl.col("is_val2")),
       "ceval": (SET / "ce_train.parquet", "text_train", pl.col("is_val"))}
MODEL = Path(os.environ.get("CE_MODEL", HERE / "model"))
LIMIT = int(os.environ.get("CE_LIMIT", 0))
OUT = Path(os.environ.get("CE_OUTDIR", HERE / "out")) / ("smoke" if LIMIT else ""); OUT.mkdir(parents=True, exist_ok=True)
CH, BS = int(os.environ.get("CE_CHUNK", 200_000)), int(os.environ.get("CE_BS", 512))
dev = torch.device("mps")
tok = AutoTokenizer.from_pretrained(MODEL)
model = AutoModelForSequenceClassification.from_pretrained(MODEL).to(dev).eval()
LOG = open(OUT / "i01_score.log", "a")


def emit(**r):
    r = {"t": time.strftime("%H:%M:%S")} | r
    print(json.dumps(r), flush=True); LOG.write(json.dumps(r) + "\n"); LOG.flush()


def prep(txt, w):
    t = time.time()
    ids = tok(txt.gather(w["i"]).to_list(), txt.gather(w["j"]).to_list(), truncation="longest_first", max_length=128)["input_ids"]
    return ids, np.fromiter(map(len, ids), np.int64, len(ids)), time.time() - t


def run(ids, lens):
    o, z = np.argsort(lens, kind="stable"), np.empty(len(ids), np.float32)
    flat, start = np.fromiter(itertools.chain.from_iterable(ids), np.int64, int(lens.sum())), np.cumsum(lens) - lens
    with torch.inference_mode():
        for b, s in enumerate(range(0, len(o), BS)):
            rows = o[s:s + BS]; L = int(math.ceil(lens[rows].max() / 32) * 32)
            x = np.full((len(rows), L), tok.pad_token_id, np.int64); m = np.zeros((len(rows), L), np.int64)
            ln = lens[rows]; r = np.repeat(np.arange(len(rows)), ln); c = np.arange(ln.sum()) - np.repeat(np.cumsum(ln) - ln, ln)
            x[r, c] = flat[np.repeat(start[rows], ln) + c]; m[r, c] = 1
            with torch.autocast("mps", dtype=torch.bfloat16):
                z[rows] = model(input_ids=torch.from_numpy(x).to(dev), attention_mask=torch.from_numpy(m).to(dev)).logits.squeeze(-1).float().cpu().numpy()
            if b % 20 == 19: torch.mps.empty_cache()
    return z


for ds in sys.argv[1:] or ["holdout", "stage4", "test"]:
    src, tname, act = SRC[ds]
    d = OUT / ds; d.mkdir(exist_ok=True)
    if not (d / "pairs.parquet").exists():
        w = pl.scan_parquet(src).filter(act).select("i", "j").collect()
        w = w.head(LIMIT) if LIMIT else w
        w.write_parquet(d / "pairs.tmp"); os.replace(d / "pairs.tmp", d / "pairs.parquet")
    W = pl.read_parquet(d / "pairs.parquet"); n = W.height; nch = (n + CH - 1) // CH
    todo = [k for k in range(nch) if not (d / f"chunk_{k:04d}.parquet").exists()]
    emit(ev="start", ds=ds, pairs=n, chunks=nch, todo=len(todo))
    if todo:
        T = pl.read_parquet(SET / f"{tname}.parquet")
        assert (T["row"].to_numpy() == np.arange(T.height)).all()
        txt = T["text"]; del T
        pool = ThreadPoolExecutor(1)
        nxt = pool.submit(prep, txt, W.slice(todo[0] * CH, CH))
        for a, k in enumerate(todo):
            t0 = time.time()
            w = W.slice(k * CH, CH); ids, lens, tok_s = nxt.result()
            if a + 1 < len(todo): nxt = pool.submit(prep, txt, W.slice(todo[a + 1] * CH, CH))
            wait_s = time.time() - t0
            z = run(ids, lens)
            w.with_columns(ce_logit=pl.Series(z)).write_parquet(d / "chunk.tmp"); os.replace(d / "chunk.tmp", d / f"chunk_{k:04d}.parquet")
            secs = time.time() - t0
            emit(ev="chunk", ds=ds, k=k, of=nch, pairs=w.height, secs=round(secs, 1), pairs_per_s=round(w.height / secs, 1),
                 tok_s=round(tok_s, 1), wait_tok_s=round(wait_s, 1), mean_len=round(float(lens.mean()), 1),
                 mps_driver_gb=round(torch.mps.driver_allocated_memory() / 2**30, 2), maxrss_gb=round(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 2**30, 2))
        pool.shutdown(); del txt
    R = pl.concat([pl.read_parquet(d / f"chunk_{k:04d}.parquet") for k in range(nch)])
    assert R.height == n and R["i"].equals(W["i"]) and R["j"].equals(W["j"]) and R["ce_logit"].is_finite().all()
    R.write_parquet(OUT / f"{ds.upper()}_ce.parquet")
    ev = {}
    for line in open(OUT / "i01_score.log"):
        e = json.loads(line)
        if e.get("ev") == "chunk" and e["ds"] == ds and e["of"] == nch: ev[e["k"]] = e
    secs = sum(e["secs"] for e in ev.values())
    s = {"ds": ds, "model": str(MODEL), "pairs": n, "chunks": nch, "secs_sum_chunks": round(secs, 1), "pairs_per_s": round(n / secs, 1) if secs else None,
         "logit_mean": round(float(R["ce_logit"].mean()), 4), "frac_logit_gt0": round(float((R["ce_logit"] > 0).mean()), 4)}
    emit(ev="done", **s); json.dump(s, open(OUT / f"i01_{ds}.json", "w"))
