"""Holdout: copy groups among an S1's candidates. Truth agreement, decision splits, which side is right."""
import numpy as np, polars as pl
from pathlib import Path
D = Path(__file__).parent; R = Path("/Users/adityarohilla/Documents/amazon-ml-2026")
K = pl.read_parquet(D / "hold_keep.parquet")
A = pl.read_parquet(D / "hold_add.parquet")
# candidate universe: blocked pairs + recall-head candidates not already blocked
C = pl.concat([K.select("i", "j", "y", "p", dec="keep"),
               A.rename({"prob": "p", "add": "dec"}).select("i", "j", pl.col("y").cast(pl.Int8), "p", "dec")]).unique(["i", "j"], keep="first", maintain_order=True)
rows = pl.concat([C["i"], C["j"]]).unique()
rec = (pl.scan_parquet(R / "cache/58f0a60c5c264fdd/recs.parquet").select("row", "src", "name", "core", "addr")
       .join(pl.scan_parquet(R / "v8_lab/ce/setup/out/text_train.parquet"), on="row")
       .filter(pl.col("row").is_in(rows.implode())).collect())
rec = rec.select("row", "src", k_text=pl.col("text").hash(), k_norm=pl.concat_str("name", "addr", separator="|").hash(),
                 k_core=pl.concat_str("core", "addr", separator="|").hash(), k_name=pl.col("name").hash(), k_addr=pl.col("addr").hash())
C = C.join(rec.rename({"row": "j"}), on="j", how="left")
C.write_parquet(D / "hold_cand.parquet")
print("cands", C.height, "S1", C["i"].n_unique(), "kept", C["dec"].sum(), "true", C["y"].sum())
for key in ["k_text", "k_norm", "k_core"]:
    G = (C.group_by("i", key).agg(n=pl.len(), ny=pl.col("y").sum(), nd=pl.col("dec").sum(), pmax=pl.col("p").max())
         .filter(pl.col("n") >= 2))
    agree = ((G["ny"] == 0) | (G["ny"] == G["n"])).mean()
    S = G.filter((pl.col("nd") > 0) & (pl.col("nd") < pl.col("n")))
    print(f"\n{key}: groups>=2 {G.height} members {G['n'].sum()} truth-uniform {agree:.4f} "
          f"(all-true {(G['ny']==G['n']).sum()}, all-false {(G['ny']==0).sum()}, mixed {((G['ny']>0)&(G['ny']<G['n'])).sum()})")
    print(f"  decision-split groups {S.height}: all-true {(S['ny']==S['n']).sum()} all-false {(S['ny']==0).sum()} mixed {((S['ny']>0)&(S['ny']<S['n'])).sum()}")
    # split detail: kept members true?  unkept members true?
    Sm = C.join(S.select("i", key), on=["i", key])
    print("  split members: kept y=1", Sm.filter(pl.col("dec") & (pl.col("y") == 1)).height, "kept y=0", Sm.filter(pl.col("dec") & (pl.col("y") == 0)).height,
          "unkept y=1", Sm.filter(~pl.col("dec") & (pl.col("y") == 1)).height, "unkept y=0", Sm.filter(~pl.col("dec") & (pl.col("y") == 0)).height)
