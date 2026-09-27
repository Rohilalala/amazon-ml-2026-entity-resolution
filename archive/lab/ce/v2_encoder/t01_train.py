"""Cross-encoder v2: continue fine-tuning the v1 cross-encoder (ce/train/model) on fresh pairs (1 pass, time-boxed).
Data: out/d01_pairs.parquet (S1 outside v7 sample, holdout and v1 CE-TRAIN) rows with is_val2 False, j not in
STAGE4/HOLDOUT active pairs (j-disjoint), rank<=6 or y==1. Val check = 8,192 random VAL2 pairs.
Everything else is v1's t01_train.py; only the init, LR (2e-5) and data paths differ.
Input: A = S1 text, B = candidate text (lower(name) | lower(address)), max_length 128, longest_first truncation.
Batches: shuffled pairs, sorted by length inside chunks of 64 batches, batch order shuffled; padded to a multiple of 32.
Schedule: linear warmup 5% then linear decay to 0 over progress = max(step/steps_in_epoch, elapsed/budget);
stops at the end of the pass or when the time budget is used. Word embeddings frozen (speed, memory).
Checkpoints (weights + optimizer + step + a val-subset check) every CKPT_MIN minutes to out/ckpt; a restart resumes
from out/ckpt (same seeded batch order, same time-based schedule). Final model to model/."""
import json, math, os, shutil, sys, time
os.environ["HF_HUB_OFFLINE"] = "1"
from pathlib import Path
import numpy as np, polars as pl, torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer

HERE = Path(__file__).parent; CE = HERE.parent; OUT = HERE / "out"; INIT = CE / "train/model"
BUDGET = float(os.environ.get("CE_BUDGET_MIN", 72)) * 60
CKPT = float(os.environ.get("CE_CKPT_MIN", 20)) * 60
MAX_STEPS = int(os.environ.get("CE_MAX_STEPS", 0))  # smoke test only
BS, LR, WARM, PAD, EC, SEED = 128, 2e-5, 0.05, 32, 10, 0
dev = torch.device("mps")
log = open(OUT / "t01_train.log", "a")
def say(**k):
    k["t"] = round(time.time() - T0, 1); s = json.dumps(k); print(s, flush=True); log.write(s + "\n"); log.flush()
T0 = time.time()

tok = AutoTokenizer.from_pretrained(INIT)
PADID = tok.pad_token_id
P = pl.read_parquet(OUT / "d01_pairs.parquet")
fit = P.filter(~pl.col("is_val2") & ~pl.col("j_in_stage4_active") & ~pl.col("j_in_holdout_active") & ((pl.col("rank") <= 6) | (pl.col("y") == 1)))
fit = fit.gather(np.random.default_rng(SEED).permutation(fit.height))
val = P.filter(pl.col("is_val2"))
val = val.gather(np.random.default_rng(1).permutation(val.height)[:8192])
txt = pl.read_parquet(CE / "setup/out/text_train.parquet")["text"]
assert (pl.read_parquet(CE / "setup/out/text_train.parquet", columns=["row"])["row"].to_numpy() == np.arange(len(txt))).all()

def encode(df):
    """Flat int32 token ids + offsets for pairs (i, j)."""
    ids, lens = [], []
    for s in range(0, df.height, 200_000):
        d = df.slice(s, 200_000)
        e = tok(txt.gather(d["i"]).to_list(), txt.gather(d["j"]).to_list(), truncation="longest_first", max_length=128)["input_ids"]
        lens.append(np.fromiter((len(x) for x in e), np.int32, len(e)))
        ids.append(np.fromiter((t for x in e for t in x), np.int32, int(lens[-1].sum())))
    lens = np.concatenate(lens)
    return np.concatenate(ids), np.concatenate([[0], np.cumsum(lens)]).astype(np.int64), lens

def batch(flat, off, lens, rows):
    L = int(math.ceil(lens[rows].max() / PAD) * PAD)
    x = np.full((len(rows), L), PADID, np.int64); m = np.zeros((len(rows), L), np.int64)
    for r, k in enumerate(rows):
        n = lens[k]; x[r, :n] = flat[off[k]:off[k] + n]; m[r, :n] = 1
    return {"input_ids": torch.from_numpy(x).to(dev), "attention_mask": torch.from_numpy(m).to(dev)}

flat, off, lens = encode(fit)
vflat, voff, vlens = encode(val)
Y = torch.tensor(fit["y"].to_numpy(), dtype=torch.float32)
say(ev="data", fit_pairs=fit.height, fit_pos=int(fit["y"].sum()), fit_s1=fit["i"].n_unique(), by_country=dict(fit.group_by("country").len().iter_rows()),
    tok_mean=float(lens.mean()), tok_p99=float(np.percentile(lens, 99)), n_trunc=int((lens >= 128).sum()), val_check_pairs=val.height)
# ce_train S1 are outside v7's sample and holdout by construction (setup s03 asserts it); re-check against v7 oof S1 set here.
oof_i = pl.read_parquet(CE.parent.parent / "code/business_entity_resolution/artifacts/v7/oof.parquet", columns=["i"])["i"].unique()
hold_i = pl.read_parquet(CE / "setup/out/holdout_s1.parquet", columns=["i"])["i"]
assert not fit["i"].is_in(oof_i).any() and not fit["i"].is_in(hold_i).any()
assert not fit["i"].is_in(val["i"].unique()).any()
v1_i = pl.read_parquet(CE / "setup/out/ce_s1.parquet", columns=["i"])["i"]
assert not fit["i"].is_in(v1_i).any() and not val["i"].is_in(v1_i).any()
s4j = pl.scan_parquet(CE / "setup/out/stage4_train.parquet").filter("active").select("j").unique().collect()["j"]
hoj = pl.scan_parquet(CE / "setup/out/holdout_pairs.parquet").filter(pl.col("active") | pl.col("active_cap")).select("j").unique().collect()["j"]
assert not fit["j"].is_in(s4j).any() and not fit["j"].is_in(hoj).any()
say(ev="leak_check", fit_s1_in_v7_sample=0, fit_s1_in_holdout=0, fit_s1_in_val2=0, s1_in_v1_ce_train=0, fit_j_in_stage4_or_holdout_active=0)
del s4j, hoj, v1_i
del P, txt, oof_i

# length-bucketed batch order
rng = np.random.default_rng(SEED)
order = np.arange(fit.height)
batches = []
for s in range(0, len(order), BS * 64):
    c = order[s:s + BS * 64]; c = c[np.argsort(lens[c], kind="stable")]
    batches += [c[k:k + BS] for k in range(0, len(c), BS)]
batches = [batches[k] for k in rng.permutation(len(batches))]
NSTEP = len(batches) if not MAX_STEPS else min(MAX_STEPS, len(batches))

RESUME = (OUT / "ckpt/state.pt").exists() and not MAX_STEPS
model = AutoModelForSequenceClassification.from_pretrained(OUT / "ckpt" if RESUME else INIT, num_labels=1).to(dev)
model.bert.embeddings.word_embeddings.weight.requires_grad_(False)
opt = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=LR, weight_decay=0.01)
ST = {"step": 0, "el": 0.0, "seen": 0}
if RESUME:
    ST = torch.load(OUT / "ckpt/state.pt", weights_only=False); opt.load_state_dict(ST.pop("opt"))

@torch.no_grad()
def val_check():
    model.eval(); out = []
    o = np.argsort(vlens)
    for s in range(0, len(o), 512):
        with torch.autocast("mps", dtype=torch.bfloat16):
            out.append(model(**batch(vflat, voff, vlens, o[s:s + 512])).logits.squeeze(-1).float().cpu())
    model.train(); torch.mps.empty_cache()
    z = np.empty(len(o), np.float32); z[o] = torch.cat(out).numpy()
    y = val["y"].to_numpy(); p = 1 / (1 + np.exp(-z.astype(np.float64)))
    from sklearn.metrics import roc_auc_score, log_loss
    return {"val_auc": round(roc_auc_score(y, z), 5), "val_logloss": round(log_loss(y, np.clip(p, 1e-7, 1 - 1e-7)), 5), "val_acc": round(float(((p >= 0.5) == y).mean()), 5)}

def save(dst, state=None):
    tmp = Path(str(dst) + ".tmp"); shutil.rmtree(tmp, ignore_errors=True)
    model.save_pretrained(tmp); tok.save_pretrained(tmp)
    if state: torch.save(state, tmp / "state.pt")
    shutil.rmtree(dst, ignore_errors=True); tmp.rename(dst)

say(ev="resume" if RESUME else "start", from_step=ST["step"], steps_in_epoch=len(batches), steps_cap=NSTEP, budget_min=BUDGET / 60, **val_check())
model.train()
t_train, last_ck, n_oom, seen, done, run_loss = time.time() - ST["el"], time.time(), 0, ST["seen"], ST["step"], [float("nan")]
for step in range(ST["step"], NSTEP):
    el = time.time() - t_train
    prog = max(step / NSTEP, el / BUDGET)
    if prog >= 1: break
    lr = LR * (prog / WARM if prog < WARM else (1 - prog) / (1 - WARM))
    for g in opt.param_groups: g["lr"] = lr
    rows = batches[step]
    try:
        x = batch(flat, off, lens, rows)
        with torch.autocast("mps", dtype=torch.bfloat16):
            logit = model(**x).logits.squeeze(-1)
        loss = torch.nn.functional.binary_cross_entropy_with_logits(logit.float(), Y[rows].to(dev))
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
    except RuntimeError as e:  # MPS OOM on a rare long batch: skip it
        if "out of memory" not in str(e): raise
        n_oom += 1; say(ev="oom_skip", step=step, L=int(lens[rows].max()))
        if n_oom > 50: raise
    opt.zero_grad(set_to_none=True)
    seen += len(rows); done += 1
    if (step + 1) % EC == 0:
        torch.mps.empty_cache()
    if (step + 1) % 50 == 0:
        run_loss.append(loss.item())
    if (step + 1) % 200 == 0:
        say(ev="step", step=step + 1, prog=round(prog, 4), lr=f"{lr:.2e}", loss_last50avg=round(float(np.mean(run_loss[-4:])), 4),
            pairs_per_s=round((seen - ST["seen"]) / (time.time() - t_train - ST["el"]), 1), driver_GB=round(torch.mps.driver_allocated_memory() / 2**30, 2))
    if time.time() - last_ck >= CKPT:
        save(OUT / "ckpt", {"opt": opt.state_dict(), "step": step + 1, "el": time.time() - t_train, "seen": seen}); say(ev="ckpt", step=step + 1, pairs_seen=seen, **val_check()); last_ck = time.time()

save(HERE / "model")
say(ev="done", steps=done, pairs_seen=seen, epoch_frac=round(seen / fit.height, 4), n_oom=n_oom,
    train_min=round((time.time() - t_train) / 60, 2), **val_check())
