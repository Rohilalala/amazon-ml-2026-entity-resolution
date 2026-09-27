"""Training throughput, one configuration per process (fresh MPS allocator): fwd + bwd + AdamW, bf16 autocast,
random batches of CE-TRAIN pairs, max_length 128, dynamic padding to a multiple of 16, torch.mps.empty_cache() every step.
usage: s02b_train_bench.py BS FREEZE_WORD_EMB(0|1) -> one JSON line appended to out/s02b_train.jsonl"""
import json, os, sys, time
os.environ["HF_HUB_OFFLINE"] = "1"
from pathlib import Path
import numpy as np, polars as pl, torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer

HERE = Path(__file__).parent
BS, FREEZE = int(sys.argv[1]), sys.argv[2] == "1"
PAD, EC = int(os.environ.get("CE_PAD", 16)), int(os.environ.get("CE_EC", 1))  # pad multiple, empty_cache every EC steps
dev = torch.device("mps")
tok = AutoTokenizer.from_pretrained(HERE.parent / "models/multilingual-e5-small")
model = AutoModelForSequenceClassification.from_pretrained(HERE.parent / "models/multilingual-e5-small", num_labels=1).to(dev)
if FREEZE:
    model.bert.embeddings.word_embeddings.weight.requires_grad_(False)
txt = pl.read_parquet(HERE / "out/text_train.parquet")["text"]
P = pl.read_parquet(HERE / "out/ce_train.parquet", columns=["i", "j", "y"]).head(200_000)
P = P.gather(np.random.default_rng(0).permutation(P.height)[:40 * BS])
A, B, Y = txt.gather(P["i"]).to_list(), txt.gather(P["j"]).to_list(), torch.tensor(P["y"].to_numpy(), dtype=torch.float32)
del txt
opt = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=2e-5, weight_decay=0.01)
model.train()
n, peak = 0, 0
for s in range(40):
    sl = slice(s * BS, (s + 1) * BS)
    x = tok(A[sl], B[sl], truncation="longest_first", max_length=128, padding=True, pad_to_multiple_of=PAD, return_tensors="pt").to(dev)
    if s == 5:
        torch.mps.synchronize(); t0 = time.time(); n = 0
    with torch.autocast("mps", dtype=torch.bfloat16):
        logit = model(**x).logits.squeeze(-1)
    loss = torch.nn.functional.binary_cross_entropy_with_logits(logit.float(), Y[sl].to(dev))
    loss.backward()
    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
    opt.step(); opt.zero_grad(set_to_none=True)
    n += BS
    peak = max(peak, torch.mps.driver_allocated_memory())
    if (s + 1) % EC == 0:
        torch.mps.empty_cache()  # the MPS allocator otherwise keeps freed blocks per shape: driver memory grew to 11 GB at bs 64 (s02c)
torch.mps.synchronize()
r = {"bs": BS, "freeze_word_emb": FREEZE, "pad_multiple": PAD, "empty_cache_every": EC, "pairs_per_s": round(n / (time.time() - t0), 1), "peak_driver_MB": round(peak / 2**20),
     "trainable_M": round(sum(p.numel() for p in model.parameters() if p.requires_grad) / 1e6, 2), "last_loss": round(loss.item(), 4)}
print(json.dumps(r), flush=True)
with open(HERE / "out/s02b_train.jsonl", "a") as f:
    f.write(json.dumps(r) + "\n")
