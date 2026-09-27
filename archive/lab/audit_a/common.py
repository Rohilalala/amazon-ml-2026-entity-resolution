"""Shared helpers for the audit scripts (read-only use of the repo and caches)."""
import sys
from pathlib import Path
import numpy as np, polars as pl
from rapidfuzz import process
from rapidfuzz.distance import Levenshtein

ROOT = Path("/Users/adityarohilla/Documents/amazon-ml-2026")
REPO = ROOT / "code/business_entity_resolution"
OUT = Path(__file__).parent / "out"
TRAIN_CACHE = ROOT / "cache/58f0a60c5c264fdd"
TEST_CACHE = ROOT / "cache/3ee42f05d06efbcb"
SHARE = {"us": 663106 / 1732544, "india": 809986 / 1732544, "france": 259452 / 1732544}
sys.path.insert(0, str(REPO))


def kept_pairs(version: str, rec_ids: pl.DataFrame) -> pl.DataFrame:
    """(i, j) rows kept by an output folder; rec_ids has row, entity_id."""
    m = pl.read_csv(ROOT / version / "matching_results.tsv", separator="\t", quote_char=None, infer_schema=False)
    ex = (m.select(pl.col("source1_entity_id").alias("s1"), pl.col("matched_entity_ids").fill_null("").str.split(",").alias("m"))
          .explode("m").filter(pl.col("m") != ""))
    return (ex.join(rec_ids.select(pl.col("row").alias("i"), pl.col("entity_id").alias("s1")), on="s1")
            .join(rec_ids.select(pl.col("row").alias("j"), pl.col("entity_id").alias("m")), on="m").select("i", "j"))


def relation(p: pl.DataFrame, recs: pl.DataFrame) -> pl.DataFrame:
    """Add house relation (hrel) and core-name relation (nrel) for pairs p(i, j); recs has row, core, addr, house."""
    r = recs.select("row", "core", "addr", "house")
    p = (p.join(r.select(pl.col("row").alias("i"), pl.col("core").alias("ci"), pl.col("house").alias("hi"), pl.col("addr").alias("ai")), on="i", how="left")
         .join(r.select(pl.col("row").alias("j"), pl.col("core").alias("cj"), pl.col("house").alias("hj"), pl.col("addr").alias("aj")), on="j", how="left"))
    lev = process.cpdist(p["hi"].to_list(), p["hj"].to_list(), scorer=Levenshtein.distance, workers=4, dtype=np.int32)
    ni = p["hi"].str.extract(r"(\d+)", 1).str.slice(0, 9).cast(pl.Int64, strict=False)
    nj = p["hj"].str.extract(r"(\d+)", 1).str.slice(0, 9).cast(pl.Int64, strict=False)
    p = p.with_columns(pl.Series("lev", lev), (ni - nj).abs().alias("nd"))
    hrel = (pl.when(pl.col("aj") == "").then(pl.lit("no_addr"))
            .when((pl.col("hi") == "") | (pl.col("hj") == "")).then(pl.lit("no_house"))
            .when(pl.col("hi") == pl.col("hj")).then(pl.lit("eq"))
            .when(pl.col("hi").str.contains(pl.col("hj"), literal=True) | pl.col("hj").str.contains(pl.col("hi"), literal=True)).then(pl.lit("substr"))
            .when(pl.col("lev") <= 1).then(pl.lit("edit1"))
            .when(pl.col("nd") <= 20).then(pl.lit("off20"))
            .otherwise(pl.lit("far")))
    ta = pl.col("ci").str.split(" ").list.unique()
    tb = pl.col("cj").str.split(" ").list.unique()
    p = p.with_columns(hrel.alias("hrel"), ta.list.set_difference(tb).alias("la"), tb.list.set_difference(ta).alias("lb"),
                       ta.list.set_intersection(tb).list.len().alias("ninter"))
    one = (p["la"].list.len() == 1) & (p["lb"].list.len() == 1)
    a1 = p["la"].list.first().fill_null("").to_list(); b1 = p["lb"].list.first().fill_null("").to_list()
    sim = process.cpdist(a1, b1, scorer=Levenshtein.normalized_similarity, workers=4, dtype=np.float32)
    p = p.with_columns(pl.Series("sim1", np.where(one.to_numpy(), sim, np.nan)))
    nrel = (pl.when((pl.col("la").list.len() == 0) & (pl.col("lb").list.len() == 0)).then(pl.lit("same"))
            .when((pl.col("la").list.len() == 0) | (pl.col("lb").list.len() == 0)).then(pl.lit("subset"))
            .when(pl.col("sim1") >= 0.5).then(pl.lit("typo1"))
            .when(pl.col("sim1") < 0.5).then(pl.lit("swap1"))
            .when(pl.col("ninter") == 0).then(pl.lit("disjoint"))
            .otherwise(pl.lit("multi")))
    return p.with_columns(nrel.alias("nrel")).drop("ci", "cj", "hi", "hj", "ai", "aj", "lev", "nd", "la", "lb", "ninter", "sim1")


def pool_tables_country(cache: Path, country: str) -> dict:
    """alignment.pool_tables restricted to one country for the per-country tables (amb, floor, df_*; exact because
    they are keyed on country_key) plus the global trigram IDF over every record of the split, computed in chunks."""
    from src import alignment
    lz = pl.scan_parquet(cache / "recs.parquet")
    sub = lz.filter(pl.col("country_key") == country).select("country_key", "src", "core", "skel", "addr").collect()
    t = alignment.pool_tables(sub.with_columns(pl.lit("").alias("core_dummy")))  # idf here covers one country only
    del sub
    n = lz.select(pl.len()).collect().item()
    df = np.zeros(alignment._HASH.n_features, dtype=np.int64)
    for s in range(0, n, alignment.IDF_CHUNK):
        core = lz.select("core").slice(s, alignment.IDF_CHUNK).collect()["core"]
        x = alignment._HASH.transform(core.to_list())
        df += np.bincount(x.indices, minlength=alignment._HASH.n_features)
    t["idf"] = (np.log((1 + n) / (1 + df)) + 1).astype(np.float32)
    return t
