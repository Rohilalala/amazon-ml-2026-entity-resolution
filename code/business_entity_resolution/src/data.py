"""Reading inputs with schema checks, and writing outputs with the submission
rules enforced before anything touches disk."""
import logging
import os
import tempfile
from pathlib import Path

import numpy as np
import polars as pl

log = logging.getLogger(__name__)

SOURCE_COLUMNS = ["entity_id", "business_name", "business_address", "country"]
TRUTH_COLUMNS = ["source1_entity_id", "matched_entity_ids"]
MATCH_COLUMN = "matched_entity_ids"
CANDIDATE_COLUMN = "candidate_entity_ids"


class DataError(ValueError):
    """An input or output file breaks the expected format."""


def read_tsv(path: Path, columns: list[str]) -> pl.DataFrame:
    """Read a TSV with every column as text. Quote characters are data, never
    delimiters, so a stray '"' in an address cannot swallow following rows."""
    if not path.is_file():
        raise DataError(f"missing file: {path}")
    try:
        df = pl.read_csv(path, separator="\t", quote_char=None, infer_schema=False,
                         missing_utf8_is_empty_string=True)
    except pl.exceptions.PolarsError as exc:
        raise DataError(f"{path}: {exc}") from exc
    missing = [c for c in columns if c not in df.columns]
    if missing:
        raise DataError(f"{path}: missing columns {missing}; found {df.columns}")
    return df.select(columns)


def load_split(data_dir: Path, split: str) -> pl.DataFrame:
    """All three sources of a split in one frame: `row` (position), `src` (1/2/3)
    plus the source columns. Raises DataError on bad ids."""
    frames = []
    for s in (1, 2, 3):
        path = data_dir / split / f"{split}_source{s}.tsv"
        df = read_tsv(path, SOURCE_COLUMNS).with_columns(
            pl.col("entity_id").str.strip_chars(), pl.lit(s, dtype=pl.Int8).alias("src"))
        bad = df.filter(~pl.col("entity_id").str.starts_with(f"S{s}-"))
        if bad.height:
            raise DataError(f"{path}: {bad.height} ids without prefix S{s}-, "
                            f"e.g. {bad['entity_id'].head(3).to_list()}")
        dups = df.filter(pl.col("entity_id").is_duplicated())
        if dups.height:
            raise DataError(f"{path}: duplicate entity ids, e.g. {dups['entity_id'].head(3).to_list()}")
        empty = df.filter(pl.col("business_name").str.strip_chars() == "").height
        if empty:
            log.warning("%s: %d records with an empty business_name", path.name, empty)
        log.info("%s: %d records", path.name, df.height)
        frames.append(df)
    return pl.concat(frames).with_row_index("row")


def load_truth(path: Path, recs: pl.DataFrame) -> pl.DataFrame:
    """True pairs as row positions (i: Source 1, j: Source 2/3), checked against
    the loaded records. Source 1 ids absent from the file count as singletons."""
    gt = read_tsv(path, TRUTH_COLUMNS)
    pairs = (gt.select(pl.col("source1_entity_id").str.strip_chars().alias("s1"),
                       pl.col("matched_entity_ids").str.split(",").alias("m"))
             .explode("m").with_columns(pl.col("m").str.strip_chars())
             .filter(pl.col("m").is_not_null() & (pl.col("m") != "")))
    ids = recs.select("entity_id", "row", "src")
    s1_rows = ids.filter(pl.col("src") == 1).select(pl.col("entity_id").alias("s1"), pl.col("row").alias("i"))
    other_rows = ids.filter(pl.col("src") != 1).select(pl.col("entity_id").alias("m"), pl.col("row").alias("j"))
    unknown_s1 = gt.filter(~pl.col("source1_entity_id").str.strip_chars().is_in(s1_rows["s1"].implode())).height
    out = pairs.join(s1_rows, on="s1", how="left").join(other_rows, on="m", how="left")
    unknown_ids = out.filter(pl.col("j").is_null()).height
    if unknown_s1 or unknown_ids or out.filter(pl.col("i").is_null()).height:
        raise DataError(f"{path}: {unknown_s1} unknown Source 1 ids, {unknown_ids} unknown matched ids")
    owners = out.group_by("j").len().filter(pl.col("len") > 1).height
    log.info("ground truth: %d Source 1 rows, %d true pairs, %d Source 2/3 records matched to "
             "more than one Source 1 record", gt.height, out.height, owners)
    return out.select("i", "j")


def id_lists(ids: pl.Series, pairs: pl.DataFrame, id_of: np.ndarray) -> pl.DataFrame:
    """One row per Source 1 id (in `ids` order) with its partners joined by commas.

    `pairs` has integer rows (i, j); `id_of` maps a row to its entity id.
    """
    lists = (pairs.select(pl.Series("s1", id_of[pairs["i"].to_numpy()]),
                          pl.Series("id", id_of[pairs["j"].to_numpy()]))
             .group_by("s1").agg(pl.col("id").unique().sort().str.join(",")))
    return (pl.DataFrame({"s1": ids}).join(lists, on="s1", how="left", maintain_order="left")
            .with_columns(pl.col("id").fill_null("")))


def check_submission(ids: pl.Series, matches: pl.DataFrame, candidates: pl.DataFrame,
                     valid: set[str] | None = None) -> None:
    """Enforce the submission rules on the two id-list frames; raise DataError
    listing every violation. `valid` (Source 2/3 ids) enables the existence check."""
    problems = []
    for name, df in (("matches", matches), ("candidates", candidates)):
        if df.height != len(ids) or df["s1"].n_unique() != df.height or not df["s1"].equals(ids):
            problems.append(f"{name}: rows must be exactly the test Source 1 ids, once each, in order")
        exploded = df.select("s1", pl.col("id").str.split(",")).explode("id").filter(pl.col("id") != "")
        if exploded.filter(pl.col("id").str.starts_with("S1-")).height:
            problems.append(f"{name}: contains Source 1 ids")
        if exploded.select(pl.struct("s1", "id").is_duplicated().any()).item():
            problems.append(f"{name}: duplicate ids within a list")
        if valid is not None:
            bad = exploded.filter(~pl.col("id").is_in(list(valid))).height
            if bad:
                problems.append(f"{name}: {bad} ids not in test Source 2/3")
    m = matches.select("s1", pl.col("id").str.split(",")).explode("id").filter(pl.col("id") != "")
    c = candidates.select("s1", pl.col("id").str.split(",")).explode("id").filter(pl.col("id") != "")
    outside = m.join(c, on=["s1", "id"], how="anti").height
    if outside:
        problems.append(f"{outside} matched ids missing from the candidate set")
    if problems:
        raise DataError("submission check failed: " + "; ".join(problems))


def write_id_lists(path: Path, column: str, frame: pl.DataFrame) -> None:
    """Write `source1_entity_id<TAB>column` rows. Written to a temp file and
    renamed, so a crash never leaves a partial file."""
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp = tempfile.mkstemp(dir=path.parent, prefix=f".{path.name}.")
    os.close(fd)
    try:
        (frame.select(pl.col("s1").alias("source1_entity_id"), pl.col("id").alias(column))
         .write_csv(tmp, separator="\t", quote_style="never"))
        os.replace(tmp, path)
    except BaseException:
        os.unlink(tmp)
        raise
