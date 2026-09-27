"""Pair encoder for stage 4: a multilingual transformer fine-tuned as a cross-encoder on
the raw text of candidate pairs (Source 1 text, candidate text) -> one match logit.

Text is the raw record, lowercased: 'business_name | business_address' (accents, punctuation
and legal forms kept; no normalisation, so an unseen country's text reaches the model as is).
Inference reads only the weights in the bundle's encoder/ folder, offline.

Every torch step runs in a fresh process: on macOS torch and LightGBM each load their own
OpenMP runtime, and torch's CPU kernels hang in a process where LightGBM has already run
(loading torch first makes LightGBM segfault). So torch and transformers are imported only
inside the functions that run in that process; importing this module never loads them.
"""
import logging
import math
import multiprocessing
import os
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

# The settings the v9 run was verified with (the spawned process inherits them). The MPS
# watermarks cap the Apple GPU allocator cache, so a blow-up raises instead of swapping.
for _name, _value in (("HF_HUB_OFFLINE", "1"), ("TOKENIZERS_PARALLELISM", "true"),
                      ("PYTORCH_MPS_HIGH_WATERMARK_RATIO", "0.7"), ("PYTORCH_MPS_LOW_WATERMARK_RATIO", "0.6")):
    os.environ.setdefault(_name, _value)

import numpy as np
import polars as pl

log = logging.getLogger(__name__)

ENCODER_DIR = "encoder"
CHUNK = 200_000  # pairs tokenised at a time
PAD = 32  # batches are padded to a multiple of this (few distinct shapes, bounded MPS cache)


def texts(raw: pl.DataFrame) -> pl.Series:
    """Pair text of every record, row-aligned with `raw` (data.load_split)."""
    return raw.select(pl.concat_str([pl.col("business_name").fill_null("").str.strip_chars().str.to_lowercase(),
                                     pl.col("business_address").fill_null("").str.strip_chars().str.to_lowercase()],
                                    separator=" | ").alias("text"))["text"]


def empty_address(text: pl.Series) -> np.ndarray:
    """Records whose address is empty, read from the pair text by splitting at the first ' | '.
    A name that itself contains ' | ' reads as non-empty; the stage-4 model was trained on
    exactly this definition, so it must not change on one side only."""
    return (text.str.splitn(" | ", 2).struct.field("field_1").fill_null("").str.strip_chars() == "").to_numpy()


def device(choice: str = "auto") -> "tuple[torch.device, torch.dtype | None]":
    """Apple GPU, else CUDA, else CPU ("auto"), or "cpu"; bfloat16 autocast where supported."""
    import torch
    if choice == "cpu":
        return torch.device("cpu"), None
    if torch.backends.mps.is_available():
        return torch.device("mps"), torch.bfloat16
    if torch.cuda.is_available():
        return torch.device("cuda"), torch.bfloat16 if torch.cuda.is_bf16_supported() else None
    return torch.device("cpu"), None


def _empty_cache(dev) -> None:
    import torch
    if dev.type == "mps":
        torch.mps.empty_cache()
    elif dev.type == "cuda":
        torch.cuda.empty_cache()


def _tokenize(tok, text: pl.Series, i: np.ndarray, j: np.ndarray, max_length: int) -> list[list[int]]:
    return tok(text.gather(i).to_list(), text.gather(j).to_list(), truncation="longest_first",
               max_length=max_length)["input_ids"]


def _batch(ids: list[list[int]], rows: np.ndarray, pad_id: int, dev) -> dict:
    """Padded input ids and attention mask for `rows` of `ids`."""
    import torch
    lens = np.fromiter((len(ids[k]) for k in rows), np.int64, len(rows))
    width = int(math.ceil(lens.max() / PAD) * PAD)
    x = np.full((len(rows), width), pad_id, np.int64)
    m = np.zeros((len(rows), width), np.int64)
    for r, k in enumerate(rows):
        x[r, :lens[r]] = ids[k]
        m[r, :lens[r]] = 1
    return {"input_ids": torch.from_numpy(x).to(dev), "attention_mask": torch.from_numpy(m).to(dev)}


def _logits(model, ids: list[list[int]], pad_id: int, batch_size: int, dev, amp) -> np.ndarray:
    """Logits for tokenised pairs, run in length-sorted batches (stable sort, so deterministic)."""
    import torch
    order = np.argsort(np.fromiter(map(len, ids), np.int64, len(ids)), kind="stable")
    out = np.empty(len(ids), np.float32)
    for b, s in enumerate(range(0, len(order), batch_size)):
        rows = order[s:s + batch_size]
        with torch.inference_mode(), torch.autocast(dev.type, dtype=amp, enabled=amp is not None):
            out[rows] = model(**_batch(ids, rows, pad_id, dev)).logits.squeeze(-1).float().cpu().numpy()
        if b % 20 == 19:
            _empty_cache(dev)
    return out


def _child_logging() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")


def _isolated(fn, *args):
    """fn(*args) in a new process (spawned, so it loads torch but not LightGBM)."""
    with ProcessPoolExecutor(1, mp_context=multiprocessing.get_context("spawn"), initializer=_child_logging) as pool:
        return pool.submit(fn, *args).result()


def score(model_dir: Path, text: pl.Series, i: np.ndarray, j: np.ndarray, params: dict) -> np.ndarray:
    """Encoder logit of every pair (i, j) of record rows, from the weights in `model_dir`."""
    return _isolated(_score, model_dir, text, i, j, params)


def _score(model_dir: Path, text: pl.Series, i: np.ndarray, j: np.ndarray, params: dict) -> np.ndarray:
    from transformers import AutoModelForSequenceClassification, AutoTokenizer
    dev, amp = device(params.get("device", "auto"))
    tok = AutoTokenizer.from_pretrained(model_dir)
    model = AutoModelForSequenceClassification.from_pretrained(model_dir).to(dev).eval()
    log.info("pair encoder on %s (%s autocast): %d pairs", dev, amp or "no", len(i))
    out, t0 = np.empty(len(i), np.float32), time.time()
    for s in range(0, len(i), CHUNK):
        ids = _tokenize(tok, text, i[s:s + CHUNK], j[s:s + CHUNK], params["max_length"])
        out[s:s + len(ids)] = _logits(model, ids, tok.pad_token_id, params["infer_batch"], dev, amp)
        done = s + len(ids)
        log.info("pair encoder: %d/%d pairs, %.0f pairs/s", done, len(i), done / (time.time() - t0))
    del model
    _empty_cache(dev)
    return out


def train(base: Path, out_dir: Path, text: pl.Series, fit: pl.DataFrame, val: pl.DataFrame, params: dict) -> dict:
    """Fine-tune `base` for one pass over `fit` (columns i, j, y) and save model + tokenizer to `out_dir`.

    Binary cross-entropy on one logit, AdamW, linear warmup then linear decay to 0, gradient
    clip 1.0, word embeddings frozen (speed and memory), bfloat16 autocast where supported.
    Batches: pairs shuffled, sorted by token length inside chunks of 64 batches, batch order
    shuffled (similar lengths per batch, about twice as fast as random batches).
    Returns a report with the validation AUC on `val`."""
    report, z = _isolated(_train, base, out_dir, text, fit, val, params)
    if val.height:
        from sklearn.metrics import roc_auc_score
        report["val_pairs"], report["val_auc"] = val.height, round(float(roc_auc_score(val["y"].to_numpy(), z)), 5)
    log.info("pair encoder: %s", report)
    return report


def _train(base: Path, out_dir: Path, text: pl.Series, fit: pl.DataFrame, val: pl.DataFrame, params: dict):
    import torch
    from transformers import AutoModelForSequenceClassification, AutoTokenizer
    dev, amp = device(params.get("device", "auto"))
    torch.manual_seed(params["seed"])
    tok = AutoTokenizer.from_pretrained(base)
    model = AutoModelForSequenceClassification.from_pretrained(base, num_labels=1).to(dev)
    model.get_input_embeddings().weight.requires_grad_(False)
    opt = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=params["lr"],
                            weight_decay=params["weight_decay"])
    rng = np.random.default_rng(params["seed"])  # one stream: pair order, then batch order
    fit = fit.gather(rng.permutation(fit.height))
    ids = []
    for s in range(0, fit.height, CHUNK):
        ids += _tokenize(tok, text, fit["i"][s:s + CHUNK].to_numpy(), fit["j"][s:s + CHUNK].to_numpy(),
                         params["max_length"])
    lens, y = np.fromiter(map(len, ids), np.int64, len(ids)), torch.tensor(fit["y"].to_numpy(), dtype=torch.float32)
    bs = params["batch_size"]
    batches = []
    for s in range(0, len(ids), bs * 64):
        c = np.arange(s, min(s + bs * 64, len(ids)))
        c = c[np.argsort(lens[c], kind="stable")]
        batches += [c[k:k + bs] for k in range(0, len(c), bs)]
    batches = [batches[k] for k in rng.permutation(len(batches))]
    log.info("pair encoder training on %s: %d pairs (%d positives), %d steps", dev, len(ids), int(y.sum()), len(batches))
    model.train()
    t0, warm = time.time(), params["warmup"]
    for step, rows in enumerate(batches):
        prog = step / len(batches)
        lr = params["lr"] * (prog / warm if prog < warm else (1 - prog) / (1 - warm))
        for g in opt.param_groups:
            g["lr"] = lr
        with torch.autocast(dev.type, dtype=amp, enabled=amp is not None):
            logit = model(**_batch(ids, rows, tok.pad_token_id, dev)).logits.squeeze(-1)
        loss = torch.nn.functional.binary_cross_entropy_with_logits(logit.float(), y[rows].to(dev))
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        opt.zero_grad(set_to_none=True)
        if step % 10 == 9:
            _empty_cache(dev)
        if step % 1000 == 999:
            log.info("step %d/%d loss %.4f, %.0f pairs/s", step + 1, len(batches), loss.item(),
                     (step + 1) * bs / (time.time() - t0))
    out_dir.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(out_dir)
    tok.save_pretrained(out_dir)
    del model, opt
    _empty_cache(dev)
    report = {"fit_pairs": len(ids), "fit_positives": int(y.sum()), "steps": len(batches),
              "train_minutes": round((time.time() - t0) / 60, 1), "device": str(dev)}
    return report, _score(out_dir, text, val["i"].to_numpy(), val["j"].to_numpy(), params) if val.height else None
