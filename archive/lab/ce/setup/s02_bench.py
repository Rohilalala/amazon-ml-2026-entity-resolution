"""Cross-encoder setup checks on MPS: load multilingual-e5-small as a 1-logit sequence classifier (pretrained encoder,
new head), autocast (bf16 / fp16) against fp32, pair token lengths on 20k train candidate pairs, and throughput
(inference batch 256/512/1024, training fwd+bwd+AdamW batch 64/128/256). Offline: weights from ../models only.
usage: s02_bench.py -> out/s02_bench.json"""
import json, os, time
os.environ["HF_HUB_OFFLINE"] = "1"; os.environ["TRANSFORMERS_OFFLINE"] = "1"
from pathlib import Path
import numpy as np, polars as pl, torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer

HERE = Path(__file__).parent
MODEL = HERE.parent / "models/multilingual-e5-small"
TRAIN_CACHE = Path("/Users/adityarohilla/Documents/amazon-ml-2026/cache/58f0a60c5c264fdd")
dev = torch.device("mps")
torch.manual_seed(0)
res = {}

tok = AutoTokenizer.from_pretrained(MODEL)
model, info = AutoModelForSequenceClassification.from_pretrained(MODEL, num_labels=1, output_loading_info=True)
model.to(dev)
res["params_M"] = round(sum(p.numel() for p in model.parameters()) / 1e6, 2)
res["loading_info"] = {k: sorted(v) for k, v in info.items() if v}
print("params", res["params_M"], "M; new/missing:", res["loading_info"], flush=True)

# 20k random train candidate pairs -> texts
txt = pl.read_parquet(HERE / "out/text_train.parquet")["text"]
P = pl.read_parquet(TRAIN_CACHE / "pairs.parquet", columns=["i", "j"])
P = P.gather(np.random.default_rng(0).choice(P.height, 20_000, replace=False))
A, B = txt.gather(P["i"]).to_list(), txt.gather(P["j"]).to_list()
del txt
lens = np.array([len(x) for x in tok(A, B)["input_ids"]])
q = [50, 90, 95, 99, 99.5, 99.9]
res["pair_tokens"] = {f"p{k}": float(np.percentile(lens, k)) for k in q} | {"max": int(lens.max()), "mean": float(lens.mean())}
for L in (64, 96, 128):
    res["pair_tokens"][f"share_over_{L}"] = float((lens > L).mean())
print("pair tokens", res["pair_tokens"], flush=True)
MAX_LEN = 128 if res["pair_tokens"]["p99.9"] > 96 else 96
res["max_length"] = MAX_LEN


def enc(a, b):
    return tok(a, b, truncation="longest_first", max_length=MAX_LEN, padding=True, pad_to_multiple_of=16,
               return_tensors="pt").to(dev)


# autocast check: same batch in fp32 / bf16 / fp16, forward + backward
x = enc(A[:256], B[:256])
model.eval()
with torch.no_grad():
    ref = model(**x).logits.float().cpu()
res["autocast"] = {}
for name, dt in (("bf16", torch.bfloat16), ("fp16", torch.float16)):
    try:
        with torch.no_grad(), torch.autocast("mps", dtype=dt):
            out = model(**x).logits.float().cpu()
        model.train()
        with torch.autocast("mps", dtype=dt):
            loss = torch.nn.functional.binary_cross_entropy_with_logits(model(**x).logits.squeeze(-1), torch.ones(256, device=dev))
        loss.backward()
        gfin = all(torch.isfinite(p.grad).all().item() for p in model.parameters() if p.grad is not None)
        model.zero_grad(set_to_none=True); model.eval()
        res["autocast"][name] = {"ok": bool(torch.isfinite(out).all()) and gfin,
                                 "max_abs_logit_diff_vs_fp32": float((out - ref).abs().max()), "grads_finite": gfin}
    except Exception as e:  # noqa: BLE001 - report and fall back
        res["autocast"][name] = {"ok": False, "error": repr(e)[:300]}
print("autocast", res["autocast"], flush=True)
DT = next((d for n, d in (("bf16", torch.bfloat16), ("fp16", torch.float16)) if res["autocast"][n]["ok"]), None)
res["dtype"] = {torch.bfloat16: "bf16", torch.float16: "fp16", None: "fp32"}[DT]


def ctx():
    return torch.autocast("mps", dtype=DT) if DT else torch.autocast("mps", enabled=False)


# inference throughput: length-sorted batches (as production scoring will run)
order = np.argsort(lens)
res["infer_pairs_per_s"] = {}
model.eval()
for bs in (256, 512, 1024):
    for dt_name, c in ((res["dtype"], ctx), ("fp32", lambda: torch.autocast("mps", enabled=False))):
        n = 0
        with torch.no_grad():
            for w in range(2):  # warm-up on two batches
                idx = order[w * bs:(w + 1) * bs]
                with c():
                    model(**enc([A[k] for k in idx], [B[k] for k in idx])).logits.cpu()
            torch.mps.synchronize(); t0 = time.time()
            for s in range(0, len(order), bs):
                idx = order[s:s + bs]
                with c():
                    model(**enc([A[k] for k in idx], [B[k] for k in idx])).logits.float().cpu()
                n += len(idx)
            torch.mps.synchronize()
        res["infer_pairs_per_s"][f"bs{bs}_{dt_name}"] = round(n / (time.time() - t0), 1)
        print("infer", bs, dt_name, res["infer_pairs_per_s"][f"bs{bs}_{dt_name}"], flush=True)
        torch.mps.empty_cache()
        if dt_name == "fp32" and res["dtype"] == "fp32":
            break

# training throughput: random batches, fwd + bwd + AdamW step
opt = torch.optim.AdamW(model.parameters(), lr=2e-5, weight_decay=0.01)
yv = torch.from_numpy((np.random.default_rng(1).random(len(A)) < 0.3).astype(np.float32))
res["train_pairs_per_s"] = {}
rng = np.random.default_rng(2)
model.train()
for bs in (64, 128, 256):
    for dt_name, c in ((res["dtype"], ctx), ("fp32", lambda: torch.autocast("mps", enabled=False))):
        steps, n = 30, 0
        for s in range(steps + 3):
            idx = rng.choice(len(A), bs, replace=False)
            xb = enc([A[k] for k in idx], [B[k] for k in idx])
            if s == 3:
                torch.mps.synchronize(); t0 = time.time(); n = 0
            with c():
                logit = model(**xb).logits.squeeze(-1)
            loss = torch.nn.functional.binary_cross_entropy_with_logits(logit.float(), yv[idx].to(dev))
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step(); opt.zero_grad(set_to_none=True)
            n += bs
        torch.mps.synchronize()
        res["train_pairs_per_s"][f"bs{bs}_{dt_name}"] = round(n / (time.time() - t0), 1)
        print("train", bs, dt_name, res["train_pairs_per_s"][f"bs{bs}_{dt_name}"], "loss", round(loss.item(), 4), flush=True)
        torch.mps.empty_cache()
        if dt_name == "fp32" and res["dtype"] == "fp32":
            break
res["peak_mps_driver_MB"] = round(torch.mps.driver_allocated_memory() / 2**20)
res["torch"], res["mps_available"] = torch.__version__, torch.backends.mps.is_available()
(HERE / "out/s02_bench.json").write_text(json.dumps(res, indent=1))
print(json.dumps(res, indent=1))
