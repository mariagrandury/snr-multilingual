import pandas as pd, numpy as np, sys
D=sys.argv[1]
v=pd.read_csv(D+"/surrogate_values.csv",low_memory=False)
print(v.shape); print(list(v.columns[:70]))
defs=pd.read_csv(D+"/surrogate_definitions.csv"); print(len(defs), defs.family.value_counts().to_dict())
names=[c for c in v.columns if c not in ["task","language","benchmark","kind","tier","proxy_size","axes"]]
print("n surrogate cols",len(names), "not in defs:", set(names)-set(defs.surrogate), "defs not in cols:", set(defs.surrogate)-set(names))
print(v.groupby("proxy_size").size())
print(v.kind.value_counts())
lit=[c for c in names if not c.startswith(("signal__","noise__","snr__"))]
print(v[lit].notna().mean().round(2).to_string())
print(v[lit].describe().T[["min","max","mean"]].round(3).to_string())
print("inf counts", {c:int(np.isinf(v[c]).sum()) for c in names if np.isinf(v[c]).sum()})
t=pd.read_csv(D+"/surrogate_targets.csv",dtype={"L":str})
print(t.shape, t.columns.tolist()); print(t.groupby(["kind","metric","axes","L"]).size().to_string())
print(t.n_pairs.describe())
