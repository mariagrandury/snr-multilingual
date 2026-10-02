import numpy as np, pandas as pd, warnings, sys, re
warnings.simplefilter("ignore")
from scipy.stats import spearmanr, rankdata
D=sys.argv[1]
v=pd.read_csv(D+"/surrogate_values.csv",low_memory=False); tg=pd.read_csv(D+"/surrogate_targets.csv",dtype={"L":str})
print("sign_persistence vs da_ckpt_mean max abs diff:", (v.sign_persistence-v.da_ckpt_mean).abs().max(), "n differing >1e-9:", ((v.sign_persistence-v.da_ckpt_mean).abs()>1e-9).sum(), "of", len(v))
print("signal__dispersion vs signal__range max diff:", (v.signal__dispersion-v.signal__range).abs().max())
KEYS=["task","language","benchmark","kind","tier","proxy_size","axes"]
names=[c for c in v.columns if c not in KEYS]
v[names]=v[names].replace([np.inf,-np.inf],np.nan)
TW=re.compile(r"^(rfgm|rf)_(.+)$")
rng=np.random.default_rng(1)
for L in ["15","8"]:
    x=tg[(tg.kind=="size")&(tg.metric=="da")&(tg["axes"]=="multi-axis")&(tg.L==L)].drop(columns=["language","n_pairs"]).merge(v[v["axes"]=="multi-axis"],on=["task","proxy_size"])
    x=x[x.kind_y=="benchmark"]; x["cluster"]=x.benchmark.str.replace(TW,r"\2",regex=True)+"|"+x.language
    cm=x.groupby("cluster")[["value"]+names].mean()   # approx: per-surrogate finite mask ignored for y mean
    y=cm["value"].to_numpy(); X=cm[names]
    def maxrho(yy):
        best=0
        for c in names:
            xx=X[c].to_numpy(); ok=np.isfinite(xx)
            if ok.sum()>=10 and np.ptp(xx[ok])>0:
                r=abs(spearmanr(xx[ok],yy[ok]).statistic)
                if r>best: best=r
        return best
    obs=maxrho(y); null=[maxrho(rng.permutation(y)) for _ in range(100)]
    print("L",L,"clusters",len(cm),"observed max|rho| over",len(names),"surrogates:",round(obs,3),"| null max|rho|: median",round(np.median(null),3),"95th",round(np.percentile(null,95),3),"P(null>=obs)",np.mean(np.array(null)>=obs))
import numpy; 
try:
    cfg=numpy.show_config(mode="dicts"); print("numpy", numpy.__version__, cfg["Build Dependencies"]["blas"]["name"])
except Exception as e: print(e)
