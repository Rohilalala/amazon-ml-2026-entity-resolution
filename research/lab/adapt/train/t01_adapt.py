"""Test-time adaptation of the v3 cross-encoder: 1 pass from ce/v3_encoder/model over out/anchor.parquet (labelled train pairs,
text_train) + the non-held-out rows of out/pseudo.parquet (pseudo-labelled TEST pairs, text_test), shuffled together.
Recipe = ce/v3_encoder/t01_train.py (bs 128, bf16 autocast, frozen word embeddings, pad 32, length-bucketed batches, empty_cache
every 10 steps, clip 1.0, AdamW wd 0.01, 5% warmup then linear decay) with lr 1e-5. Checkpoints every CE_CKPT_MIN (default 15) min
to out/ckpt (a restart resumes), with a check on 8,192 VAL3 pairs (v3's subset) and 8,192 held-out pseudo pairs. Final model to model/."""
import json, math, os, shutil, time
os.environ["HF_HUB_OFFLINE"] = "1"
from pathlib import Path
import numpy as np, polars as pl, torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer
from sklearn.metrics import roc_auc_score, log_loss

HERE = Path(__file__).parent; CE = HERE.parent.parent / "ce"; OUT = HERE / "out"; INIT = CE / "v3_encoder/model"; SET = CE / "setup/out"
CKPT = float(os.environ.get("CE_CKPT_MIN", 15)) * 60
MAX_STEPS = int(os.environ.get("CE_MAX_STEPS", 0))  # smoke test only
BS, LR, WARM, PAD, EC, SEED = 128, 1e-5, 0.05, 32, 10, 0
dev = torch.device("mps")
log = open(OUT / "t01_train.log", "a")
def say(**k):
    k["t"] = round(time.time() - T0, 1); s = json.dumps(k); print(s, flush=True); log.write(s + "\n"); log.flush()
T0 = time.time()

tok = AutoTokenizer.from_pretrained(INIT); PADID = tok.pad_token_id
A = pl.read_parquet(OUT / "anchor.parquet")
S = pl.read_parquet(OUT / "pseudo.parquet")
St = S.filter(~pl.col("is_ho")); Sh = S.filter("is_ho"); Sh = Sh.gather(np.random.default_rng(1).permutation(Sh.height)[:8192])
P = pl.read_parquet(CE / "v3_encoder/out/d01_pairs.parquet").filter("is_val3")
val = P.gather(np.random.default_rng(1).permutation(P.height)[:8192]); del P

def encode(df, txt):
    ids, lens = [], []
    for s in range(0, df.height, 200_000):
        d = df.slice(s, 200_000)
        e = tok(txt.gather(d["i"]).to_list(), txt.gather(d["j"]).to_list(), truncation="longest_first", max_length=128)["input_ids"]
        lens.append(np.fromiter((len(x) for x in e), np.int32, len(e)))
        ids.append(np.fromiter((t for x in e for t in x), np.int32, int(lens[-1].sum())))
    return np.concatenate(ids), np.concatenate(lens)

def table(name):
    T = pl.read_parquet(SET / f"{name}.parquet"); assert (T["row"].to_numpy() == np.arange(T.height)).all(); return T["text"]

tr = table("text_train"); fa, la = encode(A, tr); vflat, vlens = encode(val, tr); del tr
te = table("text_test"); fp, lp = encode(St, te); hflat, hlens = encode(Sh, te); del te
flat, lens = np.concatenate([fa, fp]), np.concatenate([la, lp]); del fa, fp
off = np.concatenate([[0], np.cumsum(lens)]).astype(np.int64)
voff = np.concatenate([[0], np.cumsum(vlens)]).astype(np.int64); hoff = np.concatenate([[0], np.cumsum(hlens)]).astype(np.int64)
Y = torch.tensor(np.concatenate([A["y"].to_numpy(), St["y"].to_numpy()]), dtype=torch.float32)
N = len(lens)
say(ev="data", pairs=N, anchor=A.height, pseudo=St.height, pos=int(Y.sum()), tok_mean=float(lens.mean()), tok_p99=float(np.percentile(lens, 99)),
    n_trunc=int((lens >= 128).sum()), val3_check=val.height, pseudo_ho_check=Sh.height)

def batch(flat, off, lens, rows):
    L = int(math.ceil(lens[rows].max() / PAD) * PAD)
    x = np.full((len(rows), L), PADID, np.int64); m = np.zeros((len(rows), L), np.int64)
    for r, k in enumerate(rows):
        n = lens[k]; x[r, :n] = flat[off[k]:off[k] + n]; m[r, :n] = 1
    return {"input_ids": torch.from_numpy(x).to(dev), "attention_mask": torch.from_numpy(m).to(dev)}

rng = np.random.default_rng(SEED)
order = rng.permutation(N); batches = []
for s in range(0, N, BS * 64):
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
def check1(fl, of, ln, y, tag):
    model.eval(); out = []; o = np.argsort(ln)
    for s in range(0, len(o), 512):
        with torch.autocast("mps", dtype=torch.bfloat16):
            out.append(model(**batch(fl, of, ln, o[s:s + 512])).logits.squeeze(-1).float().cpu())
    model.train(); torch.mps.empty_cache()
    z = np.empty(len(o), np.float32); z[o] = torch.cat(out).numpy(); p = np.clip(1 / (1 + np.exp(-z.astype(np.float64))), 1e-7, 1 - 1e-7)
    return {f"{tag}_auc": round(roc_auc_score(y, z), 5), f"{tag}_logloss": round(log_loss(y, p), 5)}

def checks():
    return check1(vflat, voff, vlens, val["y"].to_numpy(), "val3") | check1(hflat, hoff, hlens, Sh["y"].to_numpy(), "pseudo_ho")

def save(dst, state=None):
    tmp = Path(str(dst) + ".tmp"); shutil.rmtree(tmp, ignore_errors=True)
    model.save_pretrained(tmp); tok.save_pretrained(tmp)
    if state: torch.save(state, tmp / "state.pt")
    shutil.rmtree(dst, ignore_errors=True); tmp.rename(dst)

say(ev="resume" if RESUME else "start", from_step=ST["step"], steps=NSTEP, **checks())
model.train()
t_train, last_ck, n_oom, seen, run_loss = time.time() - ST["el"], time.time(), 0, ST["seen"], [float("nan")]
PG = open(OUT / "train_progress.log", "a"); t_go, last_pg = time.time(), 0.0
for step in range(ST["step"], NSTEP):
    prog = step / NSTEP
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
    seen += len(rows)
    if (step + 1) % EC == 0: torch.mps.empty_cache()
    if (step + 1) % 50 == 0: run_loss.append(loss.item())
    if (step + 1) % 200 == 0:
        say(ev="step", step=step + 1, prog=round(prog, 4), lr=f"{lr:.2e}", loss_last50avg=round(float(np.mean(run_loss[-4:])), 4),
            pairs_per_s=round((seen - ST["seen"]) / (time.time() - t_train - ST["el"]), 1), driver_GB=round(torch.mps.driver_allocated_memory() / 2**30, 2))
    if time.time() - last_pg >= 300 or step + 1 == NSTEP:
        now = time.time(); rate = (seen - ST["seen"]) / (now - t_go); eta = (NSTEP - step - 1) * BS / rate
        PG.write(json.dumps({"clock": time.strftime("%H:%M:%S"), "step": step + 1, "of": NSTEP, "pairs_per_s": round(rate, 1), "lr": f"{lr:.2e}",
                             "eta_clock": time.strftime("%H:%M", time.localtime(now + eta))}) + "\n"); PG.flush(); last_pg = now
    if time.time() - last_ck >= CKPT:
        save(OUT / "ckpt", {"opt": opt.state_dict(), "step": step + 1, "el": time.time() - t_train, "seen": seen}); say(ev="ckpt", step=step + 1, **checks()); last_ck = time.time()

if not MAX_STEPS: save(HERE / "model")
say(ev="done", steps=NSTEP, pairs_seen=seen, n_oom=n_oom, train_min=round((time.time() - t_train) / 60, 2), **checks())
