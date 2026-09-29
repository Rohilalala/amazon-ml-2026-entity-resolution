"""The leaderboard run (0.989667): stages 1-4 decision, then the post-decision chain.

  A  pipeline.predict with stage4_scope "all" and the ACNRST stacker over the adapted encoder's
     logits: exclusive owner rule, expected-F0.5 rule, unseen-country calibration and unseen_rules.
  B  dense recall adds (recall.py).
  C  France sister-organisation filter (postprocess.sister_filter).
  D  restore generator scrambles among C's drops (postprocess.restore_scrambles).
  E-H pair lists of the later rules, in the bundle's lists.parquet (postprocess.apply_lists,
     apply_late; rules.py derives each list).

Bundle layout (artifacts/final): the v9 bundle files (stages 1-3, config, vocabulary tables)
with encoder/ = the round-3 adapted pair encoder and stage4_m*.txt = the stacker's seed members;
recall/ = retriever/ (pretrained e5-small), encoder/ (v2 pair encoder), head.txt, head.json and
new_pairs_leaderboard.parquet (the leaderboard run's saved new-pair list, for --recall-pairs);
lists.parquet. Intermediate test results are saved under <bundle>/test_scores/ and reused.
"""
import hashlib
import json
import logging
import time
from pathlib import Path

import lightgbm as lgb
import numpy as np
import polars as pl

from . import data, pair_encoder, pipeline, postprocess, recall, rules, text as text_norm

log = logging.getLogger(__name__)

RECALL_DIR, LISTS_FILE = "recall", "lists.parquet"
RECALL_BATCH = 512  # pair-encoder batch the recall head's logits were computed with
RECALL_MAX_LENGTH = 128


def load_lists(bundle_dir: Path) -> dict:
    """{(step, action): frame of (s1, m[, src])} in list order."""
    L = pl.read_parquet(bundle_dir / LISTS_FILE)
    return {(s, a): L.filter((pl.col("step") == s) & (pl.col("action") == a)) for s in "EFGH" for a in ("add", "drop")}


def post_chain(sub: pl.DataFrame, cands: pl.DataFrame, records: pl.DataFrame, raw: pl.DataFrame, P: pl.DataFrame,
               head_meta: dict, lists: dict, derive: dict | None = None) -> tuple[pl.DataFrame, pl.DataFrame, dict, dict]:
    """Steps B-H on the stage-4 decision `sub` (postprocess frame) and its candidate frame.
    Returns the final frame, the candidate frame, a report and the frames after B and C (the
    next adaptation round's negatives come from them).
    `records`: row, entity_id, country_key, src of every test record; `raw`: data.load_split('test');
    `P`: recall new pairs with the head probability (s1_row, j_row, country, prob).
    `derive` ({"prep": prepared records, "cand": candidate pairs, "active": active pairs}): compute
    the lists of steps E-H with rules.derive on the current frame instead of replaying `lists`."""
    rep, t0 = {}, time.time()
    recs = postprocess.TestRecords(raw)
    b, cands, rep["B_recall"] = recall.apply(sub, cands, P, records, head_meta["t_keep"], head_meta["t_none"])
    c, rep["C_sister"] = postprocess.sister_filter(b, recs)
    d, rep["D_restore"] = postprocess.restore_scrambles(b, c, recs)
    def step(name, frame):
        """(drops, adds, India copy sources) of step `name`: the shipped lists, or rules.derive on `frame`."""
        if derive is None:
            src = lists[name, "add"].filter(pl.col("src").is_not_null()).select("s1", "m", "src") if name == "F" else None
            return lists[name, "drop"], lists[name, "add"], src
        return rules.derive(name, frame, recs, derive["prep"], derive["cand"], derive["active"], P, lists)
    frame = d
    for name in "EFG":
        drops, adds, src = step(name, frame)
        frame, rep[f"{name}_lists"] = postprocess.apply_lists(frame, drops, adds, sources=src, unique_adds=name == "G")
    drops, adds, _ = step("H", frame)
    h, rep["H_lists"] = postprocess.apply_late(frame, drops, adds)
    log.info("post-decision chain in %.0fs: %s", time.time() - t0, rep)
    return h, cands, rep, {"after_recall": b, "after_sister": c}


def with_matches(cands: pl.DataFrame, sub: pl.DataFrame) -> pl.DataFrame:
    """Candidate frame plus every matched id it lacks (appended), so matches stay a subset."""
    S1, M = postprocess.S1, postprocess.M
    extra = postprocess.pairs(sub).join(postprocess.pairs(cands), on=["s1", "m"], how="anti")
    if not extra.height:
        return cands
    return postprocess.append_sorted(cands.rename({S1: "s1"}), M, extra, "m").rename({"s1": S1})


def retrieve(bundle_dir: Path, records: pl.DataFrame, pairs: pl.DataFrame, text: pl.Series) -> pl.DataFrame:
    """Recall new pairs (recall.new_pairs) of every country, from the pretrained retriever."""
    t0 = time.time()
    emb = recall.embed(bundle_dir / RECALL_DIR / recall.RETRIEVER_DIR, records, text,
                       bundle_dir / "test_scores" / "embeddings.npy")
    log.info("retriever: %d records in %.0fs", len(emb), time.time() - t0)
    parts = []
    for country in sorted(records["country_key"].unique().to_list()):
        rows = records.filter((pl.col("country_key") == country) & (pl.col("src") == 1))["row"]
        cand = pairs.select("i", "j").filter(pl.col("i").is_in(rows.implode()))
        nb = recall.neighbours(records, emb, country)
        if nb:  # empty when the country lacks Source 1 or Source 2/3 records
            parts.append(recall.new_pairs(nb, cand, country))
    del emb
    (bundle_dir / "test_scores" / "embeddings.npy").unlink()
    return recall.sort_pairs(pl.concat(parts))


def recall_pairs(bundle_dir: Path, records: pl.DataFrame, pairs: pl.DataFrame, text: pl.Series,
                 fixed: Path | None = None, inputs: dict | None = None) -> pl.DataFrame:
    """Recall new pairs with head probabilities, cached in test_scores/recall_pairs_<key>.parquet,
    the key hashing the test files (`inputs`: name -> sha256), the recall model files, the recall
    settings, `fixed` (a saved new-pair list to score instead of retrieving; see README, recall
    pairs) and, when retrieving, the sparse candidate pairs the new pairs must lie outside."""
    rdir = bundle_dir / RECALL_DIR
    files = [rdir / recall.HEAD_FILE, rdir / recall.ENCODER_DIR / "model.safetensors",
             fixed or rdir / recall.RETRIEVER_DIR / "model.safetensors"]
    parts = {"inputs": inputs or {}, "files": [pipeline._sha256(f) for f in files], "fixed": bool(fixed),
             "settings": [recall.K_FORWARD, recall.K_REVERSE, recall.FORWARD_NEW_RANK, recall.REVERSE_RANK,
                          recall.REVERSE_CAP, RECALL_BATCH, RECALL_MAX_LENGTH]}
    if not fixed:
        h = hashlib.sha256()
        for c in ("i", "j"):
            h.update(pairs[c].to_numpy().tobytes())
        parts["candidates"] = h.hexdigest()
    cache = bundle_dir / "test_scores" / f"recall_pairs_{pipeline._dict_hash(parts)}.parquet"
    if cache.is_file():
        return pl.read_parquet(cache)
    t0 = time.time()
    if fixed:
        P = pl.read_parquet(fixed)
        missing = set(recall.NEW_PAIR_COLUMNS) - set(P.columns)
        if missing:
            raise data.DataError(f"{fixed}: recall pair list lacks columns {sorted(missing)}")
        if P.height and max(P["s1_row"].max(), P["j_row"].max()) >= len(text):
            raise data.DataError(f"{fixed}: record rows beyond this test split; the list belongs to other data")
    else:
        P = retrieve(bundle_dir, records, pairs, text)
    ce = pair_encoder.score(rdir / recall.ENCODER_DIR, text, P["s1_row"].to_numpy(), P["j_row"].to_numpy(),
                            {"max_length": RECALL_MAX_LENGTH, "infer_batch": RECALL_BATCH, "device": "auto"})
    F = recall.head_features(P.with_columns(ce_logit=pl.Series(ce)), text)
    P = P.with_columns(prob=pl.Series(lgb.Booster(model_file=str(rdir / recall.HEAD_FILE)).predict(recall.head_matrix(F))))
    cache.parent.mkdir(parents=True, exist_ok=True)
    P.write_parquet(cache)
    log.info("recall: %d new pairs in %.0fs", P.height, time.time() - t0)
    return P


def predict(data_dir: Path, bundle_dir: Path, cache: Path | None = None, fixed_recall: Path | None = None,
            derive_lists: bool = False) -> tuple[pl.DataFrame, pl.DataFrame, dict]:
    """Final match and candidate frames for the test split."""
    bundle = pipeline.Bundle.load(bundle_dir)
    if bundle.config.stage4_scope != "all" or bundle.stage4 is None:
        raise data.DataError(f"{bundle_dir}: the final run needs stage 4 with stage4_scope 'all'")
    ids, matches, cands, report = pipeline.predict(data_dir, bundle, cache=cache, bundle_dir=bundle_dir)
    records, pairs, probs = pipeline.load_scores(bundle_dir)
    derive = None
    if derive_lists:
        prob = pipeline.select_probability(probs, bundle, records["country_key"].to_numpy()[pairs["i"].to_numpy()])
        derive = {"prep": records, "cand": pairs.select("i", "j"),
                  "active": pairs.select("i", "j").filter(pl.Series(prob >= bundle.config.stage4_active))}
    del probs
    raw = data.load_split(data_dir, "test")
    text = pair_encoder.texts(raw)
    P = recall_pairs(bundle_dir, records.select("row", "entity_id", "src", "country_key"), pairs, text, fixed_recall,
                     report["provenance"]["inputs"])
    if derive is None:
        del pairs
    sub = postprocess.frame(matches["s1"], matches["id"])
    cf = postprocess.frame(cands["s1"], cands["id"])
    head_meta = json.loads((bundle_dir / RECALL_DIR / recall.HEAD_META).read_text())
    sub, cf, report["post"], steps = post_chain(sub, cf, records, raw.select("entity_id", "src", "country", "business_name",
                                                                            "business_address"), P, head_meta, load_lists(bundle_dir),
                                                derive)
    for name, frame in steps.items():  # inputs of the next adaptation round (adapt-labels --negatives-from)
        data.write_id_lists(bundle_dir / "test_scores" / f"{name}.tsv", data.MATCH_COLUMN,
                            frame.rename({postprocess.S1: "s1", postprocess.M: "id"}))
    cf = with_matches(cf, sub)
    m = sub.rename({postprocess.S1: "s1", postprocess.M: "id"})
    c = cf.rename({postprocess.S1: "s1", postprocess.M: "id"})
    data.check_submission(ids, m, c, set(raw.filter(pl.col("src") != 1)["entity_id"].to_list()))
    report["final_matched_pairs"] = postprocess.pairs(sub).height
    rdir = bundle_dir / RECALL_DIR
    report["final"] = {"recall_pairs": {"file": fixed_recall.name, "sha256": pipeline._sha256(fixed_recall)}
                       if fixed_recall else "retrieved", "derive_lists": derive_lists,
                       "lists_sha256": pipeline._sha256(bundle_dir / LISTS_FILE),
                       "head_sha256": pipeline._sha256(rdir / recall.HEAD_FILE)}
    return m, c, report


def train_recall_head(data_dir: Path, bundle_dir: Path, n_head: int = 200_000, seed: int = 2026,
                      cache: Path | None = None) -> dict:
    """Recall head from labelled training pairs: `n_head` training Source 1 records outside the
    matcher's sample and holdout, their recall new pairs (same retrieval and rule as the test),
    the recall encoder's logits and the head features; labels from the ground truth. Writes
    recall/head.txt and head.json into the bundle (recall.train_head)."""
    bundle = pipeline.Bundle.load(bundle_dir)
    cfg, rdir = bundle.config, bundle_dir / RECALL_DIR
    raw = data.load_split(data_dir, "train")
    truth = data.load_truth(data_dir / "train" / "train_ground_truth.tsv", raw)
    prov = pipeline.provenance(data_dir, "train")
    native = bundle.native_words or {}
    text_norm.set_native_words(native)
    text = pair_encoder.texts(raw)
    recs, pairs = pipeline.candidate_table(raw, cfg, {**prov["inputs"], "native_words": pipeline._dict_hash(native)}, cache)
    del raw
    records = recs.select("row", "src", "country_key")
    del recs
    s1 = records.filter(pl.col("src") == 1)
    n_true_all = s1.join(truth.group_by("i").len("n_true"), left_on="row", right_on="i", how="left") \
        .with_columns(pl.col("n_true").fill_null(0))
    sample, holdout = pipeline.sample_and_holdout(n_true_all, cfg)
    used = pl.concat([sample["row"], holdout["row"]])
    head = n_true_all.filter(~pl.col("row").is_in(used.implode())).sample(min(n_head, s1.height - used.len()), seed=seed).sort("row")
    rows = head["row"].to_numpy()
    emb_path = bundle_dir / "recall_train_embeddings.npy"
    emb = recall.embed(rdir / recall.RETRIEVER_DIR, records, text, emb_path)
    cand = pairs.select("i", "j").filter(pl.col("i").is_in(head["row"].implode()))
    parts = []
    for c in sorted(s1["country_key"].unique().to_list()):
        nb = recall.neighbours(records, emb, c)
        if nb:  # empty when the country has no Source 2/3 records
            parts.append(recall.new_pairs(nb, cand, c).filter(pl.col("s1_row").is_in(head["row"].implode())))
    del emb
    emb_path.unlink()
    P = recall.sort_pairs(pl.concat(parts))
    ce = pair_encoder.score(rdir / recall.ENCODER_DIR, text, P["s1_row"].to_numpy(), P["j_row"].to_numpy(),
                            {"max_length": RECALL_MAX_LENGTH, "infer_batch": RECALL_BATCH, "device": "auto"})
    F = recall.head_features(P.with_columns(ce_logit=pl.Series(ce)), text)
    th = (truth.filter(pl.col("i").is_in(head["row"].implode()))
          .join(cand.with_columns(inc=pl.lit(True)), on=["i", "j"], how="left").with_columns(pl.col("inc").fill_null(False)))
    y = F.select("s1_row", "j_row").join(th.select(pl.col("i").alias("s1_row"), pl.col("j").alias("j_row"), y=pl.lit(1, pl.Int8)),
                                         on=["s1_row", "j_row"], how="left", maintain_order="left")["y"].fill_null(0).to_numpy()
    tcode = np.searchsorted(rows, th["i"].to_numpy())
    n_true = np.bincount(tcode, minlength=len(rows)).astype(float)
    kept0 = np.bincount(tcode, weights=th["inc"].to_numpy(), minlength=len(rows))
    taken = F["j_row"].is_in(th.filter("inc")["j"].implode()).to_numpy()
    meta = recall.train_head(F, n_true, kept0, np.searchsorted(rows, F["s1_row"].to_numpy()), taken, y, rdir)
    log.info("recall head: %d new pairs of %d Source 1 records, %d true; %s", F.height, len(rows), int(y.sum()), meta)
    return meta
