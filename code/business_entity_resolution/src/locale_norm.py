"""Label-free address normalisation for countries whose formatting differs between sources.

France (test only) writes its Source 1 addresses with a long region phrase
("... Bordeaux, Nouvelle-Aquitaine") while its Source 2/3 copies carry the
region, a department ("Gironde") or neither. A model trained on US/India, whose
state names are folded to one code, reads those extra words as disagreement.

For every country we compare how often each address word occurs in Source 1
against Source 2/3 records. A word is dropped from that country's addresses
(both sides) when it is strongly source-asymmetric:

  * frequent in Source 1 (share >= S1_MIN) and E times more frequent there, or
  * nearly absent from Source 1 (share < S1_RARE) but common in Source 2/3
    (share >= S23_MIN).

The rule is learned from the split being processed, uses no labels, and finds
nothing in the US/India training data (verified), so trained models see the
same inputs they were trained on. Glued markers such as "ndeg12" (from "N°12")
are split so the house number survives.
"""
import math

import polars as pl

S1_MIN = 0.2
RATIO = math.e
S1_RARE = 0.005
S23_MIN = 0.02


def _words() -> pl.Expr:
    """Distinct non-numeric words of `addr`."""
    return pl.col("addr").str.split(" ").list.eval(
        pl.element().filter((pl.element() != "") & ~pl.element().str.contains(r"\d"))).list.unique()


def learn(recs: pl.DataFrame) -> dict:
    """{country_key: sorted list of address words to drop}, from prepared records
    (`country_key`, `src`, `addr`). Countries with nothing to drop are omitted."""
    base = recs.select("country_key", (pl.col("src") == 1).alias("s1"), _words().alias("w"))
    sizes = base.group_by("country_key", "s1").len("n")
    shares = (base.explode("w").drop_nulls("w").group_by("country_key", "s1", "w").len("c")
              .join(sizes, on=["country_key", "s1"]).with_columns((pl.col("c") / pl.col("n")).alias("share"))
              .pivot(on="s1", index=["country_key", "w"], values="share")
              .rename({"true": "s1_share", "false": "s23_share"}, strict=False)
              .with_columns(pl.col("s1_share", "s23_share").fill_null(0.0)))
    if "s1_share" not in shares.columns or "s23_share" not in shares.columns:
        return {}
    drop = shares.filter(
        ((pl.col("s1_share") >= S1_MIN) & (pl.col("s1_share") >= RATIO * pl.col("s23_share")))
        | ((pl.col("s1_share") < S1_RARE) & (pl.col("s23_share") >= S23_MIN)))
    return {c: sorted(g["w"].to_list()) for (c,), g in drop.group_by(["country_key"], maintain_order=True)}


ARTICLES = frozenset({"de", "la", "le", "les", "du", "des", "d", "l"})


def _strip(addr: str, drop: frozenset) -> str:
    """Remove dropped words, and runs of articles that sit between two dropped words or
    trail one at the end ('hauts de france' -> ''; 'la teste de buch' keeps its 'la')."""
    toks = addr.split()
    removed = [t in drop for t in toks]
    n = 0
    while n < len(toks):
        if toks[n] in ARTICLES and not removed[n]:
            end = n
            while end < len(toks) and toks[end] in ARTICLES:
                end += 1
            left = n > 0 and removed[n - 1]
            right = end == len(toks) or removed[end]
            if left and right:
                for m in range(n, end):
                    removed[m] = True
            n = end
        else:
            n += 1
    return " ".join(t for t, r in zip(toks, removed) if not r)


def apply(recs: pl.DataFrame, rules: dict) -> pl.DataFrame:
    """Drop the learned words from the addresses of the affected countries and split
    glued number markers ('ndeg12' -> '12'). Other countries are left untouched."""
    if not rules:
        return recs
    out = []
    for (country,), grp in recs.group_by(["country_key"], maintain_order=True):
        words = rules.get(country)
        if words:
            drop = frozenset(words)
            addr = grp["addr"].str.replace_all(r"\b(?:ndeg|no)(\d+)\b", "$1")
            grp = grp.with_columns(addr.map_elements(lambda a: _strip(a, drop), return_dtype=pl.String).alias("addr"))
        out.append(grp)
    return pl.concat(out).sort("row")
