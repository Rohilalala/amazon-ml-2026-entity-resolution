import sys, polars as pl
R = "/Users/adityarohilla/Documents/amazon-ml-2026"
FN = pl.read_parquet("fn_all.parquet"); K = pl.read_parquet("hold_keeps.parquet")
TX = pl.read_parquet(f"{R}/v8_lab/ce/setup/out/text_train.parquet")
K = K.with_columns(pmax_j=pl.col("p").max().over("j"))
FN = FN.join(K.select("i", "j", "pmax_j"), on=["i", "j"], how="left")
q = {"hi": pl.col("incand") & (pl.col("p") >= 0.5) & (pl.col("p") >= pl.col("pmax_j")),
     "mid": pl.col("incand") & (pl.col("p") < 0.5) & (pl.col("p") >= 0.1) & (pl.col("p") >= pl.col("pmax_j")),
     "owner": pl.col("incand") & (pl.col("p") < pl.col("pmax_j")),
     "low": pl.col("incand") & (pl.col("p") < 0.1) & (pl.col("p") >= pl.col("pmax_j")),
     "none": ~pl.col("incand") & ~pl.col("indense")}[sys.argv[1]]
sub = FN.filter(q).filter(pl.col("country") == sys.argv[2]).sample(int(sys.argv[3]), seed=int(sys.argv[4]) if len(sys.argv) > 4 else 1)
t = dict(zip(TX["row"].to_list(), TX["text"].to_list())) if False else None
need = set(sub["i"].to_list()) | set(sub["j"].to_list())
kk = K.filter(pl.col("i").is_in(sub["i"].implode()) & pl.col("keep"))
need |= set(kk["j"].to_list())
TXs = TX.filter(pl.col("row").is_in(list(need))); t = dict(zip(TXs["row"].to_list(), TXs["text"].to_list()))
for r in sub.iter_rows(named=True):
    print(f"--- i={r['i']} p={r['p']} pmaxj={r['pmax_j'] if r['pmax_j'] is None else round(r['pmax_j'],3)}\n  S1 : {t[r['i']]}\n  MISS: {t[r['j']]}")
    for k in kk.filter(pl.col("i") == r["i"]).iter_rows(named=True):
        print(f"  kept{'+' if k['y'] else 'X'} {k['p']:.2f}: {t[k['j']]}")
