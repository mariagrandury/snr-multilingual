import os, sys, numpy as np, pandas as pd
os.environ["MPLBACKEND"]="Agg"
from pathlib import Path
from scipy.stats import spearmanr
from analysis.rq04_surrogates import search as S
P=Path(S.__file__).parent/"pretraining"/"predictivity"
d,names=S.load(P,"predictivity")
print(len(d), len(names), d.groupby(S.TRUTH).ngroups, "half sizes", d.drop_duplicates("cluster").half.value_counts().to_dict())
def cm(g,s,ycol="value",mask=None, centre=False):
    g=g if mask is None else g[mask]
    x,y=g[s].to_numpy(float),g[ycol].to_numpy(float)
    ok=np.isfinite(x)&np.isfinite(y); g=g[ok]; x=x[ok]; y=y[ok]
    if centre:
        x=S.strat_ranks(x,g.stratum.to_numpy()); y=S.strat_ranks(y,g.stratum.to_numpy())
    m=pd.DataFrame({"u":g.unit.to_numpy(),"x":x,"y":y}).groupby("u").mean()
    return round(spearmanr(m.x,m.y).statistic,3), len(m)
for kind in ("size","goal","ckpt"):
    g=d[(d.t_kind==kind)&(d.metric=="da")&(d["axes"]=="multi-axis")&(d.L=="all")&(d.kind=="benchmark")]
    print("==",kind,len(g))
    # composition: proxy index and number of cells per cluster vs truth
    pi=g.proxy_size.map({"90M":0,"175M":1,"350M":2,"600M":3,"1B":4})
    m=pd.DataFrame({"u":g.unit,"pi":pi,"y":g.value}).groupby("u").agg(pi=("pi","mean"),y=("y","mean"),n=("y","size"))
    print("  rho(mean proxy idx, mean truth)",round(spearmanr(m.pi,m.y).statistic,3),"rho(n cells, mean truth)",round(spearmanr(m.n,m.y).statistic,3), "clusters w/ all 5 proxies", (g.groupby("unit").proxy_size.nunique()==5).sum(), "of", g.unit.nunique())
    for s in ["da_ckpt_mean","sign_consistency_window","window_kendall","kendall_w_window","consecutive_kendall_late","n_items","autocorr","icc_window","g_coefficient","gain_over_noise","snr__rel_std__ckpt_rel"]:
        if kind=="ckpt" and s in S.CIRCULAR: continue
        out=[s, "raw", cm(g,s), "stratum-ranked means", cm(g,s,centre=True)]
        if kind=="ckpt":
            out+=["f<=0.7",cm(g,s,mask=g.frac<=0.7), "f>=0.8", cm(g,s,mask=g.frac>=0.8)]
        print("  ",*out)
