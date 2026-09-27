"""Dense recall: true pairs the sparse blocker missed, found with a pretrained text embedding.

1. Embed every record with the pretrained bi-encoder intfloat/multilingual-e5-small (MIT licence,
   not fine-tuned): 'query: ' + pair text, mean pooling, L2 norm, float16.
2. Per country and source, exact cosine kNN: forward, each Source 1 record's top 20 Source 2 /
   Source 3 records; reverse, each Source 2/3 record's top 3 Source 1 records.
3. New pairs (outside the sparse candidate set), rule m1_r1_cap10: the Source 1 record's best
   dense neighbour per source that is not already a candidate (forward new rank 1), or a
   reverse rank-1 pair; reverse-only pairs are capped at 10 per Source 1 record, by cosine.
4. Score each new pair with a pair encoder (the v2 encoder) and a small LightGBM head (19 features).
5. Decision (step B of the leaderboard chain): add a new pair when its head probability is at
   least t_keep (Source 1 record already keeps a pair) or t_none (keeps none), its candidate is
   kept by no Source 1 record, and it is the most likely added pair of that candidate.

torch runs in a spawned process (pair_encoder._isolated), as for the pair encoder.
"""
import json
import logging
import time
from pathlib import Path

import numpy as np
import polars as pl
from rapidfuzz import fuzz
from rapidfuzz.process import cpdist

from . import pair_encoder, postprocess
from .stage4 import best_other

log = logging.getLogger(__name__)

RETRIEVER_DIR, ENCODER_DIR, HEAD_FILE, HEAD_META = "retriever", "encoder", "head.txt", "head.json"
EMBED_CHUNK, EMBED_BATCH = 200_000, 512
K_FORWARD, K_REVERSE = 20, 3
FORWARD_NEW_RANK, REVERSE_RANK, REVERSE_CAP = 1, 1, 10
KNN_QUERY_BLOCK, KNN_TARGET_BLOCK, KNN_GROUP = 2048, 131072, 128
HEAD_FEATURES = ["ce", "ce_rank", "ce_gap", "cos", "cos_gap", "drank", "nrank", "rrank", "rev_only", "src", "n_new",
                 "e_name_i", "e_addr_i", "e_name_j", "e_addr_j", "name_ratio", "name_tset", "addr_ratio", "addr_tset"]
HEAD_PARAMS = {"objective": "binary", "learning_rate": 0.05, "num_leaves": 31, "min_data_in_leaf": 100,
               "feature_fraction": 0.8, "bagging_fraction": 0.8, "bagging_freq": 1, "lambda_l2": 1.0,
               "deterministic": True, "force_row_wise": True, "verbose": -1, "seed": 42, "num_threads": 6}
# Threshold grid for (t_keep, t_none); 1.01 adds nothing.
HEAD_GRID = [round(x, 2) for x in np.arange(0.20, 0.96, 0.05)] + [0.97, 0.98, 0.99, 1.01]


# ---------------------------------------------------------------- 1-2. embedding and kNN

def embed(model_dir: Path, records: pl.DataFrame, text: pl.Series, path: Path) -> np.ndarray:
    """float16 unit vectors of every record (row-aligned with `text`), from the pretrained
    retriever, written to the .npy file `path` and returned memory-mapped (11.7M test records
    take 9 GB). Records are embedded per country and source, in row order, EMBED_CHUNK at a
    time (length-sorted batches inside a chunk), so the batches, and the bits, do not depend on
    what else is in the pool."""
    groups = [g["row"].sort().to_numpy() for _, g in
              sorted(records.group_by(["country_key", "src"]), key=lambda x: (x[0][0], x[0][1]))]
    pair_encoder._isolated(_embed, model_dir, [text.gather(r) for r in groups], groups, path, len(text))
    return np.load(path, mmap_mode="r")


def _embed(model_dir: Path, texts: list, groups: list, path: Path, n: int) -> None:
    import torch
    from transformers import AutoModel, AutoTokenizer
    dev, _ = pair_encoder.device("auto")
    tok = AutoTokenizer.from_pretrained(model_dir)
    model = AutoModel.from_pretrained(model_dir, dtype=torch.bfloat16 if dev.type != "cpu" else torch.float32)
    model = model.to(dev).eval()
    mm = np.lib.format.open_memmap(path, mode="w+", dtype=np.float16, shape=(n, model.config.hidden_size))
    t0, done = time.time(), 0
    for text, rows in zip(texts, groups):
        out, texts_ = np.empty((len(text), model.config.hidden_size), np.float16), text.to_list()
        for s in range(0, len(texts_), EMBED_CHUNK):
            ids = tok(["query: " + t for t in texts_[s:s + EMBED_CHUNK]], truncation=True, max_length=128)["input_ids"]
            lens = np.fromiter(map(len, ids), np.int64, len(ids))
            order = np.argsort(lens, kind="stable")
            with torch.inference_mode():
                for b, q in enumerate(range(0, len(order), EMBED_BATCH)):
                    idx = order[q:q + EMBED_BATCH]
                    width = -(-int(lens[idx].max()) // 8) * 8
                    a, m = np.zeros((len(idx), width), np.int64), np.zeros((len(idx), width), np.int64)
                    for r, k in enumerate(idx):
                        a[r, :lens[k]], m[r, :lens[k]] = ids[k], 1
                    a, m = torch.from_numpy(a).to(dev), torch.from_numpy(m).to(dev)
                    h = model(input_ids=a, attention_mask=m).last_hidden_state
                    mf = m.unsqueeze(-1).float()
                    e = torch.nn.functional.normalize((h.float() * mf).sum(1) / mf.sum(1), dim=-1)
                    out[s + idx] = e.half().cpu().numpy()
                    if b % 20 == 19:
                        pair_encoder._empty_cache(dev)
            done += len(ids)
            log.info("retriever: %d/%d texts, %.0f texts/s", done, n, done / (time.time() - t0))
        mm[rows] = out
        pair_encoder._empty_cache(dev)
    mm.flush()


def knn(queries: np.ndarray, targets: np.ndarray, k: int) -> tuple[np.ndarray, np.ndarray]:
    """Top-k targets by cosine for every query (unit float16 vectors): (index [n, k], cosine [n, k])."""
    return pair_encoder._isolated(_knn, queries, targets, k)


def _knn(queries: np.ndarray, targets: np.ndarray, k: int) -> tuple[np.ndarray, np.ndarray]:
    """Blocked float16 matmul. The top-k of a row lies in the k column groups (of KNN_GROUP) with
    the largest maxima, which is exact and much faster than a top-k over the full row on MPS."""
    import torch
    dev, _ = pair_encoder.device("auto")
    G, nt = KNN_GROUP, len(targets)
    ntp = -(-nt // G) * G
    T = torch.zeros((ntp, targets.shape[1]), dtype=torch.float16, device=dev)
    T[:nt] = torch.from_numpy(np.ascontiguousarray(targets)).to(dev)
    Q = torch.from_numpy(np.ascontiguousarray(queries)).to(dev)
    ar = torch.arange(G, device=dev)

    def topk_rows(s, k):
        b, n = s.shape
        kg = min(k, n // G)
        gi = s.view(b, n // G, G).amax(-1).topk(kg, dim=1).indices
        cols = (gi.unsqueeze(-1) * G + ar).view(b, kg * G)
        v, i = s.gather(1, cols).topk(k, dim=1)
        return v, cols.gather(1, i)
    out_v, out_i = np.empty((len(queries), k), np.float16), np.empty((len(queries), k), np.int64)
    with torch.inference_mode():
        for bi, qs in enumerate(range(0, len(queries), KNN_QUERY_BLOCK)):
            q = Q[qs:qs + KNN_QUERY_BLOCK]
            cv, ci = [], []
            for ts in range(0, ntp, KNN_TARGET_BLOCK):
                s = q @ T[ts:ts + KNN_TARGET_BLOCK].T
                if ts + s.shape[1] > nt:
                    s[:, nt - ts:] = -2.0
                v, i = topk_rows(s, k)
                cv.append(v)
                ci.append(i + ts)
            cv, ci = torch.cat(cv, 1), torch.cat(ci, 1)
            v, i = cv.topk(k, dim=1)
            out_v[qs:qs + len(q)], out_i[qs:qs + len(q)] = v.cpu().numpy(), ci.gather(1, i).cpu().numpy()
            if bi % 20 == 19:
                pair_encoder._empty_cache(dev)
    return out_i, out_v


# ---------------------------------------------------------------- 3. new pairs

def neighbours(records: pl.DataFrame, emb: np.ndarray, country: str) -> dict:
    """Forward and reverse kNN lists of one country: {(src, "fwd"|"rev"): (query rows, target rows, cos)}."""
    rows = {s: records.filter((pl.col("country_key") == country) & (pl.col("src") == s))["row"].sort().to_numpy()
            for s in (1, 2, 3)}
    out = {}
    for s in (2, 3):
        if not len(rows[s]) or not len(rows[1]):
            continue
        idx, cos = knn(emb[rows[1]], emb[rows[s]], min(K_FORWARD, len(rows[s])))
        out[(s, "fwd")] = (rows[1], rows[s][idx].astype(np.uint32), cos)
        idx, cos = knn(emb[rows[s]], emb[rows[1]], min(K_REVERSE, len(rows[1])))
        out[(s, "rev")] = (rows[s], rows[1][idx].astype(np.uint32), cos)
    return out


def new_pairs(nb: dict, cand: pl.DataFrame, country: str) -> pl.DataFrame:
    """Rule m1_r1_cap10 over one country's kNN lists, outside the sparse candidates `cand` (i, j).
    Columns s1_row, j_row, dense_cos, dense_rank (forward rank, null = reverse only), new_rank,
    rev_rank, source, country."""
    fwd, rev = [], []
    for (s, kind), (q, t, c) in nb.items():
        k = t.shape[1]
        if kind == "fwd":
            fwd.append(pl.DataFrame({"i": np.repeat(q, k), "j": t.ravel(), "cos": c.ravel().astype(np.float32),
                                     "rank": np.tile(np.arange(1, k + 1, dtype=np.uint8), len(q)),
                                     "src": np.full(len(q) * k, s, np.uint8)}))
        else:
            rev.append(pl.DataFrame({"i": t.ravel(), "j": np.repeat(q, k), "rcos": c.ravel().astype(np.float32),
                                     "rev_rank": np.tile(np.arange(1, k + 1, dtype=np.uint8), len(q)),
                                     "src": np.full(len(q) * k, s, np.uint8)}))
    f = (pl.concat(fwd).join(cand.with_columns(pl.lit(True).alias("in_cand")), on=["i", "j"], how="left")
         .with_columns(pl.col("in_cand").fill_null(False)).filter(~pl.col("in_cand")).drop("in_cand"))
    f = f.with_columns(pl.col("rank").rank("ordinal").over("i", "src").cast(pl.UInt8).alias("new_rank"))
    r = pl.concat(rev).join(cand, on=["i", "j"], how="anti")
    d = (f.join(r, on=["i", "j", "src"], how="full", coalesce=True)
         .with_columns(pl.coalesce("cos", "rcos").alias("dense_cos"), pl.lit(country).alias("country")))
    fw = pl.col("new_rank").is_not_null() & (pl.col("new_rank") <= FORWARD_NEW_RANK)
    rv = pl.col("rev_rank").is_not_null() & (pl.col("rev_rank") <= REVERSE_RANK)
    # Ties in cosine (float16) are broken by source and candidate row, so the cap is deterministic.
    ro = (d.filter(rv & ~fw).sort("i", "dense_cos", "src", "j", descending=[False, True, False, False])
          .with_columns(pl.col("dense_cos").rank("ordinal", descending=True).over("i").alias("_rr")))
    return (pl.concat([d.filter(fw), ro.filter(pl.col("_rr") <= REVERSE_CAP).drop("_rr")])
            .select(pl.col("i").alias("s1_row"), pl.col("j").alias("j_row"), "dense_cos", pl.col("rank").alias("dense_rank"),
                    "new_rank", "rev_rank", pl.col("src").alias("source"), "country"))


def sort_pairs(p: pl.DataFrame) -> pl.DataFrame:
    return p.sort("s1_row", "source", "dense_cos", "j_row", descending=[False, False, True, False])


# ---------------------------------------------------------------- 4. head

def _split_text(col):
    s = pl.col(col).fill_null("").str.splitn(" | ", 2)
    return s.struct.field("field_0").fill_null("").str.strip_chars(), s.struct.field("field_1").fill_null("").str.strip_chars()


def head_features(df: pl.DataFrame, text: pl.Series) -> pl.DataFrame:
    """df: new pairs (whole Source 1 lists) with ce_logit; `text`: pair text of every record (row-aligned)."""
    df = df.with_columns(ti=text.gather(df["s1_row"]), tj=text.gather(df["j_row"]))
    name_i, addr_i = _split_text("ti")
    name_j, addr_j = _split_text("tj")
    df = df.with_columns(name_i=name_i, addr_i=addr_i, name_j=name_j, addr_j=addr_j).drop("ti", "tj")
    sim = {k: cpdist(df[a].to_list(), df[b].to_list(), scorer=sc, workers=6).astype(np.float32)
           for k, a, b, sc in (("name_ratio", "name_i", "name_j", fuzz.ratio), ("name_tset", "name_i", "name_j", fuzz.token_set_ratio),
                               ("addr_ratio", "addr_i", "addr_j", fuzz.ratio), ("addr_tset", "addr_i", "addr_j", fuzz.token_set_ratio))}
    df = df.with_columns(ce=pl.col("ce_logit").cast(pl.Float64), cos=pl.col("dense_cos").cast(pl.Float64),
                         drank=pl.col("dense_rank"), nrank=pl.col("new_rank"), rrank=pl.col("rev_rank"),
                         rev_only=pl.col("dense_rank").is_null().cast(pl.Int8), src=pl.col("source"),
                         e_name_i=(pl.col("name_i") == "").cast(pl.Int8), e_addr_i=(pl.col("addr_i") == "").cast(pl.Int8),
                         e_name_j=(pl.col("name_j") == "").cast(pl.Int8), e_addr_j=(pl.col("addr_j") == "").cast(pl.Int8),
                         **{k: pl.Series(v) for k, v in sim.items()}).drop("name_i", "addr_i", "name_j", "addr_j")
    return df.with_columns(ce_rank=pl.col("ce").rank("min", descending=True).over("s1_row"),
                           ce_gap=pl.col("ce") - best_other("ce", "s1_row"),
                           cos_gap=pl.col("cos") - best_other("cos", "s1_row"), n_new=pl.len().over("s1_row"))


def head_matrix(df: pl.DataFrame) -> np.ndarray:
    return df.select(HEAD_FEATURES).cast(pl.Float32).to_numpy()


def add_new(code: np.ndarray, j: np.ndarray, p: np.ndarray, thr_by_s1: np.ndarray, j_taken: np.ndarray) -> np.ndarray:
    """Added new pairs: p >= the Source 1 record's threshold, candidate kept by nobody, most
    likely pair of its candidate (ties: first row)."""
    cand = (p >= thr_by_s1[code]) & ~j_taken
    idx = np.flatnonzero(cand)
    idx = idx[np.lexsort((idx, -p[idx]))]
    _, first = np.unique(j[idx], return_index=True)
    m = np.zeros(len(p), bool)
    m[idx[first]] = True
    return m


def train_head(F: pl.DataFrame, n_true: np.ndarray, kept0: np.ndarray, code: np.ndarray, j_taken: np.ndarray,
               y: np.ndarray, out_dir: Path) -> dict:
    """Head on labelled new pairs of training Source 1 records outside the matcher's sample and
    holdout: 5-fold CV grouped by record, refit at the mean best round, then (t_keep, t_none) on
    the grid by the macro F0.5 of the combined decision, where a record's existing keeps are its
    true pairs inside the sparse candidates (`kept0`, a perfect in-candidate matcher)."""
    import lightgbm as lgb
    from sklearn.model_selection import GroupKFold
    from .metric import f05_counts
    x, g = head_matrix(F), F["s1_row"].to_numpy()
    folds = np.empty(len(y), np.int8)
    for n, (_, va) in enumerate(GroupKFold(n_splits=5, shuffle=True, random_state=42).split(x, groups=g)):
        folds[va] = n
    ds = lgb.Dataset(x, y, feature_name=HEAD_FEATURES, params=HEAD_PARAMS, free_raw_data=False).construct()
    oof, rounds = np.zeros(len(y)), []
    for n in range(5):
        tr, va = np.flatnonzero(folds != n), np.flatnonzero(folds == n)
        b = lgb.train(HEAD_PARAMS, ds.subset(tr), 3000, valid_sets=[ds.subset(va)],
                      callbacks=[lgb.early_stopping(100, verbose=False)])
        oof[va] = b.predict(x[va], num_iteration=b.best_iteration)
        rounds.append(b.best_iteration)
    has_keep, n = kept0 > 0, len(n_true)
    best = (-1.0, None)
    for tk in HEAD_GRID:
        for tn in HEAD_GRID:
            add = add_new(code, F["j_row"].to_numpy(), oof, np.where(has_keep, tk, tn), j_taken)
            f = f05_counts(kept0 + np.bincount(code, weights=add, minlength=n), n_true,
                           kept0 + np.bincount(code, weights=add & (y == 1), minlength=n)).mean()
            if f > best[0]:
                best = (f, (tk, tn))
    nr = int(round(np.mean(rounds)))
    out_dir.mkdir(parents=True, exist_ok=True)
    lgb.train(HEAD_PARAMS, ds, nr).save_model(str(out_dir / HEAD_FILE))
    meta = {"features": HEAD_FEATURES, "rounds": nr, "params": HEAD_PARAMS, "t_keep": best[1][0], "t_none": best[1][1],
            "fold_rounds": rounds, "oof_f05": round(float(best[0]), 6)}
    (out_dir / HEAD_META).write_text(json.dumps(meta, indent=1))
    return meta


# ---------------------------------------------------------------- 5. decision (step B)

def _append(base: pl.DataFrame, col: str, extra: pl.DataFrame) -> pl.DataFrame:
    e = extra.group_by("s1").agg(pl.col("jid").sort().str.join(",").alias("x"))
    return (base.join(e, on="s1", how="left", maintain_order="left")
            .with_columns(pl.when(pl.col("x").is_null()).then(pl.col(col))
                          .when(pl.col(col) == "").then(pl.col("x"))
                          .otherwise(pl.col(col) + "," + pl.col("x")).alias(col)).drop("x"))


def apply(sub: pl.DataFrame, cands: pl.DataFrame, P: pl.DataFrame, records: pl.DataFrame, t_keep: float,
          t_none: float) -> tuple[pl.DataFrame, pl.DataFrame, dict]:
    """Step B. `sub` / `cands`: submission frames of matches and candidates (postprocess.S1, M);
    `P`: new pairs (s1_row, j_row, country, prob). Added ids are appended (sorted) to the match
    lists; every new pair is appended to the candidate lists."""
    S1, M = postprocess.S1, postprocess.M
    ids = records.select("row", pl.col("entity_id"), "country_key", "src")
    best = sub.select(pl.col(S1).alias("s1"), pl.col(M).fill_null("").alias("m"))
    best = best.join(ids.select(pl.col("entity_id").alias("s1"), pl.col("row").alias("s1_row"), pl.col("country_key").alias("c"),
                                "src"), on="s1", how="left", maintain_order="left")
    assert best["s1_row"].null_count() == 0 and (best["src"] == 1).all()
    best = best.drop("src").with_row_index("code")
    kept_ids = best.select("s1", pl.col("m").str.split(",")).explode("m").filter(pl.col("m") != "")["m"].unique()
    P = (P.join(best.select("s1_row", "code", "s1", has_keep=pl.col("m") != ""), on="s1_row", how="left", maintain_order="left")
         .join(ids.select(pl.col("row").alias("j_row"), pl.col("entity_id").alias("jid")), on="j_row", how="left",
               maintain_order="left"))
    assert P["code"].null_count() == 0 and P["jid"].null_count() == 0
    taken = P["jid"].is_in(kept_ids.implode()).to_numpy()
    thr_by_s1 = np.where(best["m"].to_numpy() != "", t_keep, t_none)
    add = add_new(P["code"].to_numpy(), P["j_row"].to_numpy(), P["prob"].to_numpy(), thr_by_s1, taken)
    A = P.filter(pl.Series(add))
    assert A["jid"].n_unique() == A.height
    res = _append(best.select("s1", "m"), "m", A)
    cres = _append(cands.select(pl.col(S1).alias("s1"), pl.col(M).fill_null("").alias("m")), "m", P)
    by_country = dict(A.group_by("country").len().iter_rows())
    return (res.rename({"s1": S1, "m": M}), cres.rename({"s1": S1, "m": M}),
            {"new_pairs": P.height, "added": A.height, "added_by_country": by_country, "t_keep": t_keep, "t_none": t_none})
