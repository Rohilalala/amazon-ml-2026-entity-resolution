"""Embed records with the pretrained e5-small bi-encoder ('query: ' + lower(name) | lower(address), mean pool, L2 norm).
usage: e01_embed.py SPLIT [BS]  -> out/emb/{split}_{country}_s{src}.npy (float16 memmap, rows ascending) + _rows.npy.
TRAIN: every S2/S3 record plus every S1 (superset of holdout + head S1; the reverse pass needs all S1). TEST: all.
Chunks of 200k texts, length-sorted inside the chunk; resumes from the per-group progress file."""
import json, os, sys, time
from concurrent.futures import ThreadPoolExecutor
os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("PYTORCH_MPS_HIGH_WATERMARK_RATIO", "0.7")
os.environ.setdefault("PYTORCH_MPS_LOW_WATERMARK_RATIO", "0.6")
import numpy as np, polars as pl, torch
from transformers import AutoTokenizer, AutoModel
ROOT = "/Users/adityarohilla/Documents/amazon-ml-2026"
M = f"{ROOT}/v8_lab/ce/models/multilingual-e5-small"
CACHE = {"train": f"{ROOT}/cache/58f0a60c5c264fdd", "test": f"{ROOT}/cache/3ee42f05d06efbcb"}
EMB = os.path.join(os.path.dirname(os.path.abspath(__file__)), "out", "emb")
CHUNK = 200_000
split = sys.argv[1]; BS = int(sys.argv[2]) if len(sys.argv) > 2 else 512
os.makedirs(EMB, exist_ok=True)
dev = torch.device("mps")
recs = pl.read_parquet(f"{CACHE[split]}/recs.parquet", columns=["row", "src", "country_key"])
text = pl.read_parquet(f"{ROOT}/v8_lab/ce/setup/out/text_{split}.parquet")
assert text.height == recs.height and (text["row"] == recs["row"]).all()
recs = recs.with_columns(text["text"])
del text
tok = AutoTokenizer.from_pretrained(M)
model = AutoModel.from_pretrained(M, dtype=torch.bfloat16).to(dev).eval()  # bf16 weights: 9.9k vs 7.9k texts/s autocast, min cos 0.99971 (e00)


def tokenize(texts):
    ids = tok(["query: " + t for t in texts], truncation=True, max_length=128)["input_ids"]
    L = np.fromiter((len(x) for x in ids), np.int64, len(ids))
    return ids, L


def encode(ids, L, out):
    order = np.argsort(L, kind="stable")
    with torch.inference_mode():
        for b, s in enumerate(range(0, len(order), BS)):
            idx = order[s:s + BS]; ml = int(L[idx].max()); ml = -(-ml // 8) * 8
            a = np.zeros((len(idx), ml), np.int64); m = np.zeros((len(idx), ml), np.int64)
            for r, k in enumerate(idx):
                a[r, :L[k]] = ids[k]; m[r, :L[k]] = 1
            a, m = torch.from_numpy(a).to(dev), torch.from_numpy(m).to(dev)
            h = model(input_ids=a, attention_mask=m).last_hidden_state
            mf = m.unsqueeze(-1).float()
            e = torch.nn.functional.normalize((h.float() * mf).sum(1) / mf.sum(1), dim=-1)
            out[idx] = e.half().cpu().numpy()
            if b % 20 == 19:
                torch.mps.empty_cache()


pool = ThreadPoolExecutor(1)
T0 = time.time(); done_texts = 0
for (country, src), g in sorted(recs.group_by(["country_key", "src"]), key=lambda x: (x[0][0], x[0][1])):
    g = g.sort("row"); n = g.height
    base = f"{EMB}/{split}_{country}_s{src}"
    if os.path.exists(base + "_rows.npy"):
        print(f"skip {base} (done)", flush=True); continue
    prog = base + ".progress"
    start = int(open(prog).read()) if os.path.exists(prog) else 0
    mm = np.lib.format.open_memmap(base + ".npy", mode="r+" if start else "w+", dtype=np.float16, shape=(n, 384))
    texts = g["text"].to_list()
    starts = list(range(start, n, CHUNK))
    fut = pool.submit(tokenize, texts[starts[0]:starts[0] + CHUNK]) if starts else None
    for ci, s in enumerate(starts):
        t = time.time()
        ids, L = fut.result()
        if ci + 1 < len(starts):
            fut = pool.submit(tokenize, texts[starts[ci + 1]:starts[ci + 1] + CHUNK])
        buf = np.empty((len(ids), 384), np.float16)
        encode(ids, L, buf)
        mm[s:s + len(ids)] = buf; mm.flush()
        with open(prog + ".tmp", "w") as f: f.write(str(s + len(ids)))
        os.replace(prog + ".tmp", prog)
        done_texts += len(ids)
        print(json.dumps({"group": f"{country}_s{src}", "done": s + len(ids), "n": n, "chunk_s": round(time.time() - t, 1),
                          "texts_per_s": round(len(ids) / (time.time() - t)), "elapsed_min": round((time.time() - T0) / 60, 2),
                          "mean_len": round(float(L.mean()), 1)}), flush=True)
    del mm
    np.save(base + "_rows.npy", g["row"].to_numpy().astype(np.uint32))
    torch.mps.empty_cache()
print(json.dumps({"split": split, "texts": done_texts, "min": round((time.time() - T0) / 60, 2)}), flush=True)
