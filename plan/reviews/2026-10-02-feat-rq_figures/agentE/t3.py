import pandas as pd, re
from pretrain.ladder_report import CELL_RE, on_grid
w=pd.read_csv("/private/tmp/claude-501/-Users-mariagrandury-Projects-epfl-snr-multilingual/f7d6f8d9-67db-4234-9ff3-0c61069faa4c/scratchpad/ladder-pub/ladder_report.csv",usecols=["cell","size","L","arch","scheme","seed","iter","run__diverged","run__complete","run__target_iters"],low_memory=False)
print(len(w), w.cell.isna().sum())
c=w.dropna(subset=["cell"]).drop_duplicates("cell")
m=c.cell.map(CELL_RE.match)
c=c.assign(matched=m.map(bool), ong=[bool(x) and on_grid(x) for x in m], gbs=[x["gbs"] if x else None for x in m])
print("unmatched:",c[~c.matched].cell.tolist()[:10])
print("off-grid cells:",c[c.matched&~c.ong].cell.tolist())
print(c[c.ong].groupby("size").agg(n=("cell","size"),gbs=("gbs",lambda s: sorted(set(map(str,s))))))
# does regex size agree with column, scheme, seed
bad=[(r.cell,r["size"]) for r,x in zip(c.itertuples(),m) if x and (x["size"]!=r.size)]
print("size mismatch", bad[:5])
print(c[c.ong].scheme.value_counts().to_dict())
for name in ["lm-90M-L1-deep-seed1904","lm-90M-L1-b84-deep-seed1904","lm-175M-L2-b168-deep-seed64","lm-175M-L2-deep-seed64","lm-350M-L8-b168-deep-seed1904","lm-90M-L1-b504-deep-seed1904","diag-lm-90M-L1-b84-deep-seed1904","lm-90M-L1-b84-deep-seed1904-old"]:
    x=CELL_RE.match(name); print(name, bool(x) and on_grid(x))
