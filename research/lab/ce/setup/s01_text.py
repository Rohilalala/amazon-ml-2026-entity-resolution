"""Pair-text tables: one row per record, in the pipeline's row order (data.load_split: S1, S2, S3 concatenated),
text = lower(business_name) + ' | ' + lower(business_address), raw (accents, punctuation, legal forms kept).
Checks the row order against the split's cache (entity_id equality on every row).
usage: s01_text.py -> out/text_{train,test}.parquet (row UInt32, text String)"""
from pathlib import Path
import polars as pl

ROOT = Path("/Users/adityarohilla/Documents/amazon-ml-2026")
DATA = ROOT / "student_resource/dataset"
CACHE = {"train": ROOT / "cache/58f0a60c5c264fdd", "test": ROOT / "cache/3ee42f05d06efbcb"}
OUT = Path(__file__).parent / "out"

for split, cache in CACHE.items():
    frames = [pl.read_csv(DATA / split / f"{split}_source{s}.tsv", separator="\t", quote_char=None, infer_schema=False,
                          missing_utf8_is_empty_string=True).select("entity_id", "business_name", "business_address")
              for s in (1, 2, 3)]
    raw = pl.concat(frames).with_row_index("row")
    del frames
    ids = pl.read_parquet(cache / "recs.parquet", columns=["row", "entity_id"])
    same = (ids["row"] == raw["row"]).all() and (ids["entity_id"] == raw["entity_id"].str.strip_chars()).all()
    print(split, "records", raw.height, "row order equals cache recs:", same, flush=True)
    assert same
    del ids
    t = raw.select("row", pl.concat_str([pl.col("business_name").fill_null("").str.strip_chars().str.to_lowercase(),
                                         pl.col("business_address").fill_null("").str.strip_chars().str.to_lowercase()],
                                        separator=" | ").alias("text"))
    t.write_parquet(OUT / f"text_{split}.parquet")
    print(split, "text chars p50/p99", t["text"].str.len_chars().quantile(0.5), t["text"].str.len_chars().quantile(0.99),
          "example", t["text"].head(2).to_list(), flush=True)
