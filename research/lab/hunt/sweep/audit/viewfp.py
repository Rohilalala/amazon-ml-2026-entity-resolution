import sys, polars as pl
R = "/Users/adityarohilla/Documents/amazon-ml-2026"
K = pl.read_parquet("hold_keeps.parquet"); FP = pl.read_parquet("fp_keep.parquet").filter(pl.col("country") == sys.argv[1]).sample(int(sys.argv[2]), seed=3)
Tr = pl.read_parquet(f"{R}/v8_lab/ce/setup/out/truth_train.parquet").filter(pl.col("i").is_in(FP["i"].implode()))
kk = K.filter(pl.col("i").is_in(FP["i"].implode()) & pl.col("keep"))
need = set(FP["i"].to_list()) | set(kk["j"].to_list()) | set(Tr["j"].to_list())
TX = pl.scan_parquet(f"{R}/v8_lab/ce/setup/out/text_train.parquet").filter(pl.col("row").is_in(list(need))).collect(); t = dict(zip(TX["row"].to_list(), TX["text"].to_list()))
for r in FP.iter_rows(named=True):
    print(f"--- i={r['i']} n_true={r['n_true']} kept={r['kept']}\n  S1 : {t[r['i']]}")
    for k in kk.filter(pl.col("i") == r["i"]).iter_rows(named=True):
        print(f"  kept{'+' if k['y'] else 'X'} {k['p']:.2f}: {t[k['j']]}")
    for j in Tr.filter(pl.col("i") == r["i"]).join(kk.select("j"), on="j", how="anti")["j"]:
        print(f"  miss : {t[j]}")
