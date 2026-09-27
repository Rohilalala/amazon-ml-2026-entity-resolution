"""Score CE-VAL (20,000 held-out S1, all their CE-TRAIN pairs) with the fine-tuned cross-encoder -> out/val_scores.parquet.
usage: t02_eval.py [MODEL_DIR] (default model/). Metrics: t03_metrics.py."""
import json, math, os, sys, time
os.environ["HF_HUB_OFFLINE"] = "1"
from pathlib import Path
import numpy as np, polars as pl, torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer

HERE = Path(__file__).parent; CE = HERE.parent
MODEL = Path(sys.argv[1]) if len(sys.argv) > 1 else HERE / "model"
dev = torch.device("mps")
tok = AutoTokenizer.from_pretrained(MODEL)
model = AutoModelForSequenceClassification.from_pretrained(MODEL).to(dev).eval()
val = pl.read_parquet(CE / "setup/out/ce_train.parquet").filter(pl.col("is_val"))
txt = pl.read_parquet(CE / "setup/out/text_train.parquet")["text"]
A, B = txt.gather(val["i"]).to_list(), txt.gather(val["j"]).to_list()
del txt
ids = tok(A, B, truncation="longest_first", max_length=128)["input_ids"]
lens = np.array([len(x) for x in ids]); o = np.argsort(lens, kind="stable")
t0, z = time.time(), np.empty(len(ids), np.float32)
with torch.no_grad():
    for s in range(0, len(o), 512):
        rows = o[s:s + 512]; L = int(math.ceil(lens[rows].max() / 32) * 32)
        x = np.full((len(rows), L), tok.pad_token_id, np.int64); m = np.zeros((len(rows), L), np.int64)
        for r, k in enumerate(rows):
            x[r, :lens[k]] = ids[k]; m[r, :lens[k]] = 1
        with torch.autocast("mps", dtype=torch.bfloat16):
            z[rows] = model(input_ids=torch.from_numpy(x).to(dev), attention_mask=torch.from_numpy(m).to(dev)).logits.squeeze(-1).float().cpu().numpy()
        if (s // 512) % 20 == 0: torch.mps.empty_cache()
secs = time.time() - t0
val.with_columns(logit=pl.Series(z), p=pl.Series(1 / (1 + np.exp(-z.astype(np.float64)))), a=pl.Series(A), b=pl.Series(B)).write_parquet(HERE / "out/val_scores.parquet")
r = {"model": str(MODEL), "pairs": val.height, "infer_s": round(secs, 1), "pairs_per_s": round(val.height / secs, 1)}
print(json.dumps(r)); json.dump(r, open(HERE / "out/t02_eval.json", "w"))
