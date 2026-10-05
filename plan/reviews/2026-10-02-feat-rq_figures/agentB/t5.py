import numpy as np, pandas as pd, warnings
warnings.simplefilter("ignore")
from analysis.rq04_surrogates import search as Srch
rng=np.random.default_rng(7)
n,K=240,200; res=[]; scr=[]
for rep in range(15):
    names=[f"s{i}" for i in range(K)]
    g=pd.DataFrame(rng.normal(size=(n,K)),columns=names)
    g["value"]=rng.normal(size=n); g["kind"]="benchmark"; g["half"]=rng.integers(0,2,n); g["stratum"]=0; g["unit"]=np.arange(n); g["task"]=[f"t{i}" for i in range(n)]
    y=g["value"].to_numpy()
    rows=[{"surrogate":s, **Srch.full_stats(g[s].to_numpy(),y,g["stratum"].to_numpy(),g["unit"].to_numpy())} for s in names]
    c=pd.DataFrame(rows); scr.append((c.p<.05).mean())
    top=c.reindex(c.rho.abs().sort_values(ascending=False).index).head(Srch.TOP_K)
    f=pd.DataFrame(Srch.filters_truth(({"t_kind":"size"},g,top,"full")))
    ind=f[f.surrogate=="[filter indicator]"]; ins=f[f.surrogate!="[filter indicator]"]
    res.append(((ind.p<.05).mean(), (ins.p<.05).mean(), len(ind), ins.p.notna().sum()))
r=np.array(res); print("null data (y independent of all 200 surrogates), nominal alpha .05")
print("screen: share p<.05 =", np.mean(scr))
print("filter indicator rows: share p<.05 =", r[:,0].mean(), "| top-K inside filters: share p<.05 =", r[:,1].mean(), "| rows per truth:", r[0,2], r[0,3])
