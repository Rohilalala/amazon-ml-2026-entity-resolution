"""Dense kNN per country and source: blocked fp16 matmul on MPS over unit vectors (cosine).
usage: k01_knn.py SPLIT COUNTRY SRC MODE [QROWS]
  MODE fwd: S1 queries -> top-20 records of source SRC (QROWS: .npy of S1 rows to restrict the queries)
  MODE rev: every SRC record -> top-3 S1 among ALL S1 of the country (the sparse blocker's reverse pass)
-> out/knn/{split}_{country}_s{src}_{mode}.npz: q_row[n], t_row[n,k], cos[n,k] (fp16).
Top-k per row is exact via group maxima: the top-k entries lie in the k groups (of G columns) with the largest maxima;
torch.topk over the full 131k-wide rows was 25x slower than the matmul on MPS (bench e00)."""
import os, sys, time, json
os.environ.setdefault("PYTORCH_MPS_HIGH_WATERMARK_RATIO", "0.7")
os.environ.setdefault("PYTORCH_MPS_LOW_WATERMARK_RATIO", "0.6")
import numpy as np, torch
D = os.path.join(os.path.dirname(os.path.abspath(__file__)), "out")
QB, TC, G = 2048, 131072, 128
split, country, src, mode = sys.argv[1], sys.argv[2], int(sys.argv[3]), sys.argv[4]
K = 20 if mode == "fwd" else 3
os.makedirs(f"{D}/knn", exist_ok=True)
tag = f"{split}_{country}_s{src}_{mode}"
dev = torch.device("mps")


def load(s):
    return np.load(f"{D}/emb/{split}_{country}_s{s}_rows.npy"), np.load(f"{D}/emb/{split}_{country}_s{s}.npy", mmap_mode="r")


(s1r, s1e), (pr, pe) = load(1), load(src)
if mode == "fwd":
    qr, qe, tr, te = s1r, s1e, pr, pe
    if len(sys.argv) > 5:
        sel = np.isin(s1r, np.load(sys.argv[5])); qr, qe = s1r[sel], s1e[sel]
else:
    qr, qe, tr, te = pr, pe, s1r, s1e
nq, nt = len(qr), len(tr)
ntp = -(-nt // G) * G
T = torch.zeros((ntp, 384), dtype=torch.float16, device=dev)
T[:nt] = torch.from_numpy(np.ascontiguousarray(te)).to(dev)
Q = torch.from_numpy(np.ascontiguousarray(qe)).to(dev)
ar = torch.arange(G, device=dev)


def topk_rows(s, k):
    b, n = s.shape
    kg = min(k, n // G)
    gi = s.view(b, n // G, G).amax(-1).topk(kg, dim=1).indices
    cols = (gi.unsqueeze(-1) * G + ar).view(b, kg * G)
    v, i = s.gather(1, cols).topk(k, dim=1)
    return v, cols.gather(1, i)


out_v = np.empty((nq, K), np.float16); out_i = np.empty((nq, K), np.int64)
t0 = time.time()
with torch.inference_mode():
    for bi, qs in enumerate(range(0, nq, QB)):
        q = Q[qs:qs + QB]
        cv, ci = [], []
        for ts in range(0, ntp, TC):
            s = q @ T[ts:ts + TC].T
            if ts + s.shape[1] > nt:
                s[:, nt - ts:] = -2.0
            v, i = topk_rows(s, K)
            if bi == 0 and ts == 0:  # exactness self-check against torch.topk
                rv = s.topk(K, dim=1).values
                assert torch.equal(rv, v), "grouped top-k differs from torch.topk"
            cv.append(v); ci.append(i + ts)
        cv, ci = torch.cat(cv, 1), torch.cat(ci, 1)
        v, i = cv.topk(K, dim=1)
        out_v[qs:qs + QB] = v.cpu().numpy(); out_i[qs:qs + QB] = ci.gather(1, i).cpu().numpy()
        if bi % 100 == 0:
            el = time.time() - t0; d = qs + len(q)
            print(json.dumps({"tag": tag, "q_done": d, "nq": nq, "elapsed_s": round(el, 1),
                              "eta_min": round(el / d * (nq - d) / 60, 1)}), flush=True)
        if bi % 20 == 19:
            torch.mps.empty_cache()
assert out_i.max() < nt and (out_v > -1.5).all()
np.savez(f"{D}/knn/{tag}.npz", q_row=qr, t_row=tr[out_i].astype(np.uint32), cos=out_v)
print(json.dumps({"tag": tag, "nq": nq, "nt": nt, "k": K, "total_s": round(time.time() - t0, 1),
                  "pairs_per_s": round(nq * nt / (time.time() - t0))}), flush=True)
