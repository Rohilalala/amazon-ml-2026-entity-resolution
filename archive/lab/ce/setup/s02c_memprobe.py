"""MPS memory per training step (bs 64, bf16 autocast), to find what grows. Stops a run above 7 GB allocated.
usage: s02c_memprobe.py ATTN(sdpa|eager) AUTOCAST(1|0)"""
import os, sys, time
os.environ["HF_HUB_OFFLINE"] = "1"
from pathlib import Path
import numpy as np, polars as pl, torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer

HERE = Path(__file__).parent
ATTN, AC = sys.argv[1], sys.argv[2] == "1"
dev = torch.device("mps")
tok = AutoTokenizer.from_pretrained(HERE.parent / "models/multilingual-e5-small")
model = AutoModelForSequenceClassification.from_pretrained(HERE.parent / "models/multilingual-e5-small", num_labels=1,
                                                           attn_implementation=ATTN).to(dev)
txt = pl.read_parquet(HERE / "out/text_train.parquet")["text"]
P = pl.read_parquet(HERE / "out/ce_train.parquet", columns=["i", "j", "y"]).head(20_000)
A, B, Y = txt.gather(P["i"]).to_list(), txt.gather(P["j"]).to_list(), torch.tensor(P["y"].to_numpy(), dtype=torch.float32)
del txt
opt = torch.optim.AdamW(model.parameters(), lr=2e-5)
model.train()
GB = 2**30
print("after load", round(torch.mps.current_allocated_memory() / GB, 2), flush=True)
t0 = time.time()
for s in range(25):
    sl = slice(s * 64, (s + 1) * 64)
    x = tok(A[sl], B[sl], truncation="longest_first", max_length=128, padding=True, pad_to_multiple_of=16, return_tensors="pt").to(dev)
    with torch.autocast("mps", dtype=torch.bfloat16, enabled=AC):
        logit = model(**x).logits.squeeze(-1)
    loss = torch.nn.functional.binary_cross_entropy_with_logits(logit.float(), Y[sl].to(dev))
    loss.backward(); opt.step(); opt.zero_grad(set_to_none=True)
    a = torch.mps.current_allocated_memory() / GB
    print(ATTN, "ac" if AC else "fp32", "step", s, "L", x["input_ids"].shape[1], "alloc GB", round(a, 2), "driver GB", round(torch.mps.driver_allocated_memory() / GB, 2), flush=True)
    if a > 7:
        print("STOP: above 7 GB"); break
torch.mps.synchronize()
print("pairs/s", round(64 * (s + 1) / (time.time() - t0), 1))
