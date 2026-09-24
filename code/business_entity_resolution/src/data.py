"""Reading inputs with schema checks, and writing outputs with the submission
rules enforced before anything touches disk."""
import csv
import logging
import os
import tempfile
from pathlib import Path

import pandas as pd

log = logging.getLogger(__name__)

SOURCE_COLUMNS = ["entity_id", "business_name", "business_address", "country"]
TRUTH_COLUMNS = ["source1_entity_id", "matched_entity_ids"]
MATCH_COLUMN = "matched_entity_ids"
CANDIDATE_COLUMN = "candidate_entity_ids"


class DataError(ValueError):
    """An input or output file breaks the expected format."""


def read_tsv(path: Path, columns: list[str]) -> pd.DataFrame:
    """Read a TSV as strings. Quote characters are kept as text (QUOTE_NONE), so a
    stray '"' in an address can never swallow the following rows."""
    if not path.is_file():
        raise DataError(f"missing file: {path}")
    try:
        df = pd.read_csv(path, sep="\t", dtype=str, keep_default_na=False,
                         quoting=csv.QUOTE_NONE, encoding="utf-8")
    except pd.errors.ParserError as exc:
        raise DataError(f"{path}: {exc}") from exc
    missing = [c for c in columns if c not in df.columns]
    if missing:
        raise DataError(f"{path}: missing columns {missing}; found {list(df.columns)}")
    return df


def load_split(data_dir: Path, split: str) -> pd.DataFrame:
    """The three source files of a split in one frame, with a `src` column (1, 2, 3)."""
    frames = []
    for s in (1, 2, 3):
        path = data_dir / split / f"{split}_source{s}.tsv"
        df = read_tsv(path, SOURCE_COLUMNS)[SOURCE_COLUMNS]
        df = df.assign(entity_id=df["entity_id"].str.strip())
        bad = ~df["entity_id"].str.startswith(f"S{s}-")
        if bad.any():
            raise DataError(f"{path}: {bad.sum()} ids without prefix S{s}-, "
                            f"e.g. {df.loc[bad, 'entity_id'].head(3).tolist()}")
        if df["entity_id"].duplicated().any():
            dups = df.loc[df["entity_id"].duplicated(), "entity_id"].head(3).tolist()
            raise DataError(f"{path}: duplicate entity ids, e.g. {dups}")
        empty = (df["business_name"].str.strip() == "").sum()
        if empty:
            log.warning("%s: %d records with an empty business_name", path.name, empty)
        log.info("%s: %d records", path.name, len(df))
        frames.append(df.assign(src=s))
    return pd.concat(frames, ignore_index=True)


def parse_ids(cell: str) -> list[str]:
    """Split a comma-separated id list; an empty cell is an empty list."""
    return [x.strip() for x in cell.split(",") if x.strip()]


def load_truth(path: Path, recs: pd.DataFrame) -> dict[str, frozenset]:
    """Ground truth as {source1_entity_id: frozenset(matched ids)}, checked against
    the loaded records. Source 1 ids absent from the file count as singletons."""
    gt = read_tsv(path, TRUTH_COLUMNS)
    s1 = set(recs.loc[recs["src"] == 1, "entity_id"])
    others = set(recs.loc[recs["src"] != 1, "entity_id"])
    truth = {r.source1_entity_id.strip(): frozenset(parse_ids(r.matched_entity_ids))
             for r in gt.itertuples()}
    unknown_s1 = set(truth) - s1
    unknown_ids = set().union(*truth.values()) - others
    if unknown_s1 or unknown_ids:
        raise DataError(f"{path}: {len(unknown_s1)} unknown Source 1 ids, "
                        f"{len(unknown_ids)} unknown matched ids")
    owners = pd.Series([c for ids in truth.values() for c in ids]).value_counts()
    log.info("ground truth: %d Source 1 rows (%d singletons), %d matched ids, "
             "%d ids matched to more than one Source 1 entity, %d Source 1 ids not listed",
             len(truth), sum(1 for v in truth.values() if not v), int(owners.sum()),
             int((owners > 1).sum()), len(s1 - set(truth)))
    return truth


def check_submission(ids: list[str], matches: dict, candidates: dict, valid: set[str]) -> None:
    """Enforce the submission rules; raise DataError listing every violation."""
    problems = []
    if len(ids) != len(set(ids)):
        problems.append("duplicate Source 1 rows")
    for name, mapping in (("matches", matches), ("candidates", candidates)):
        stray = set(mapping) - set(ids)
        if stray:
            problems.append(f"{name}: {len(stray)} keys are not test Source 1 ids")
        bad = {x for v in mapping.values() for x in v} - valid
        if bad:
            problems.append(f"{name}: {len(bad)} ids not in test Source 2/3, e.g. {sorted(bad)[:3]}")
    outside = sum(len(set(v) - set(candidates.get(k, ()))) for k, v in matches.items())
    if outside:
        problems.append(f"{outside} matched ids missing from the candidate set")
    if problems:
        raise DataError("submission check failed: " + "; ".join(problems))


def write_ids(path: Path, column: str, ids: list[str], mapping: dict) -> None:
    """One row per Source 1 id (input order), ids sorted and comma-joined.
    Written to a temp file and renamed, so a crash never leaves a partial file."""
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp = tempfile.mkstemp(dir=path.parent, prefix=f".{path.name}.")
    try:
        with os.fdopen(fd, "w", encoding="utf-8", newline="") as fh:
            fh.write(f"source1_entity_id\t{column}\n")
            for i in ids:
                fh.write(f"{i}\t{','.join(sorted(mapping.get(i, ())))}\n")
        os.replace(tmp, path)
    except BaseException:
        os.unlink(tmp)
        raise
