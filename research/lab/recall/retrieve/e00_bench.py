"""Throughput bench: pretrained e5-small bi-encoder on 50k random train texts; blocked matmul + top-k on MPS."""
import json, os, time
os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("PYTORCH_MPS_HIGH_WATERMARK_RATIO", "0.7")
os.environ.setdefault("PYTORCH_MPS_LOW_WATERMARK_RATIO", "0.6")
import numpy as np, polars as pl, torch
from transformers import AutoTokenizer, AutoModel
M = "/Users/adityarohilla/Documents/amazon-ml-2026/v8_lab/ce/models/multilingual-e5-small"
T = "/Users/adityarohilla/Documents/amazon-ml-2026/v8_lab/ce/setup/out/text_train.parquet"
OUT = os.path.join(os.path.dirname(__file__), "out")
dev = torch.device("mps")
texts = pl.read_parquet(T).sample(50_000, seed=1)["text"].to_list()
tok = AutoTokenizer.from_pretrained(M)
R = {}
t = time.time(); enc = tok(["query: " + x for x in texts], truncation=True, max_length=128)["input_ids"]; R["tok_s"] = time.time() - t
L = np.array([len(x) for x in enc]); R["len_p50_p99_max"] = [int(np.percentile(L, 50)), int(np.percentile(L, 99)), int(L.max())]
order = np.argsort(L, kind="stable")

def run(model, bs, dtype_mode):
    t = time.time(); out = np.empty((len(enc), 384), np.float16)
    with torch.inference_mode():
        for b, s in enumerate(range(0, len(order), bs)):
            idx = order[s:s + bs]; ml = int(L[idx].max())
            ids = np.zeros((len(idx), ml), np.int64); am = np.zeros((len(idx), ml), np.int64)
            for r, k in enumerate(idx): ids[r, :L[k]] = enc[k]; am[r, :L[k]] = 1
            ids_t, am_t = torch.from_numpy(ids).to(dev), torch.from_numpy(am).to(dev)
            if dtype_mode == "autocast":
                with torch.autocast("mps", dtype=torch.bfloat16): h = model(input_ids=ids_t, attention_mask=am_t).last_hidden_state
            else:
                h = model(input_ids=ids_t, attention_mask=am_t).last_hidden_state
            m = am_t.unsqueeze(-1).to(h.dtype)
            e = (h * m).sum(1).float() / am_t.sum(1, keepdim=True).float()
            e = torch.nn.functional.normalize(e, dim=-1)
            out[idx] = e.half().cpu().numpy()
            if b % 20 == 0: torch.mps.empty_cache()
    torch.mps.synchronize()
    return len(enc) / (time.time() - t), out

model = AutoModel.from_pretrained(M).to(dev).eval()
run(model, 512, "autocast")  # warm-up
for bs in (512, 1024):
    R[f"autocast_bs{bs}_texts_per_s"], ref = run(model, bs, "autocast"); print(R, flush=True)
mb = AutoModel.from_pretrained(M, dtype=torch.bfloat16).to(dev).eval()
for bs in (512,):
    R[f"bf16w_bs{bs}_texts_per_s"], o2 = run(mb, bs, "bf16w"); print(R, flush=True)
R["bf16w_vs_autocast_min_cos"] = float((ref.astype(np.float32) * o2.astype(np.float32)).sum(1).min())
mf = AutoModel.from_pretrained(M).to(dev).eval()
_, o32 = run(mf, 512, "fp32")
R["fp32_vs_autocast_min_cos"] = float((ref.astype(np.float32) * o32.astype(np.float32)).sum(1).min())
print(R, flush=True)
del model, mb, mf; torch.mps.empty_cache()
# matmul + topk bench: 2048 queries x 131072 targets fp16
q = torch.randn(2048, 384, device=dev, dtype=torch.float16); tt = torch.randn(131072, 384, device=dev, dtype=torch.float16)
for _ in range(2): s = q @ tt.T; v, i = s.topk(20, dim=1); v2, i2 = s.topk(3, dim=0)
torch.mps.synchronize(); t = time.time()
for _ in range(5): s = q @ tt.T; v, i = s.topk(20, dim=1); v2, i2 = s.topk(3, dim=0)
torch.mps.synchronize(); dt = (time.time() - t) / 5
R["block_2048x131072_matmul_topk20_rev3_s"] = dt
t = time.time()
for _ in range(5): s = q @ tt.T
torch.mps.synchronize(); R["block_matmul_only_s"] = (time.time() - t) / 5
R["tflops_matmul"] = 2 * 2048 * 131072 * 384 / R["block_matmul_only_s"] / 1e12
print(json.dumps(R, indent=1)); json.dump(R, open(os.path.join(OUT, "e00_bench.json"), "w"), indent=1)
