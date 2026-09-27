"""Pairs changed per country on final16b and an arithmetic expected-LB estimate (not a measurement): each add is true with
probability p; the S1's current keeps are assumed all true; true add: n_true = k+1 -> kept k+1; false add: n_true = k."""
import polars as pl
H = "/Users/adityarohilla/Documents/amazon-ml-2026/v8_lab/hunt/"
def f05(kept, n, tp):
    if kept == 0 and n == 0: return 1.0
    if kept == 0 or n == 0 or tp == 0: return 0.0
    P, R = tp / kept, tp / n; return 1.25 * P * R / (0.25 * P + R)
base = pl.read_csv(H + "final16b/matching_results.tsv", separator="\t", quote_char=None, infer_schema=False)
k = base.select(s1="source1_entity_id", k=pl.col("matched_entity_ids").fill_null("").str.split(",").list.eval(pl.element().filter(pl.element() != "")).list.len())
A = pl.read_parquet(H + "sweep/judge/final_adds_rule.parquet").join(k, on="s1")
N = base.height
P = {"fr_cie": (0.95, 0.97, 0.98), "fr_tailmid": (0.90, 0.93, 0.97), "in_copy": (0.69, 0.93, 0.99)}
for r, (lo, mid, hi) in P.items():
    a = A.filter(pl.col("rule") == r); ks = a["k"].to_list()
    g = lambda p: sum(p * (1 - f05(x, x + 1, x)) - (1 - p) * (1 - f05(x + 1, x, x)) for x in ks) / N
    print(r, "adds", a.height, "S1", a["s1"].n_unique(), "mean k", round(sum(ks) / len(ks), 2), "empty S1", sum(x == 0 for x in ks),
          "| LB est p=%.2f/%.2f/%.2f: %+.6f / %+.6f / %+.6f" % (lo, mid, hi, g(lo), g(mid), g(hi)))
