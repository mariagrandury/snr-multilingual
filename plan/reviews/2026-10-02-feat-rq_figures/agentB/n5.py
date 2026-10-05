import os; os.environ["MPLBACKEND"]="Agg"
import numpy as np, pandas as pd
from scipy.stats import rankdata, spearmanr
from analysis.rq04_surrogates import catalogue as C, search as S
P=os.path.dirname(C.__file__)+"/pretraining/predictivity/"
d=pd.read_csv(P+"surrogate_definitions.csv")
reg={m:(f,sg,fo,so) for m,(f,sg,fo,so) in C.SURROGATES.items()}
dd=d[~d.family.isin(["AllenAI signal","noise","SNR grid"])].set_index("surrogate")
print("registry == csv:", set(reg)==set(dd.index), all(reg[m][0]==dd.family[m] and reg[m][2]==dd.formula[m] and reg[m][1]==dd.expected_sign[m] for m in reg), "signals",len(C.SIGNALS),"SMALL",C.SMALL_SIZES,"AXES",C.AXES,"FRACS",C.FRACS, "MIN_PAIRS",C.MIN_PAIRS,"MIN_UNITS",S.MIN_UNITS)
t=pd.read_csv(P+"surrogate_targets.csv",dtype={"L":str})
print("targets rows",len(t),"n_pairs min",t.n_pairs.min(), "sizes",sorted(t.proxy_size.unique()), "langs",t.language.nunique(), "multi/?? present", t.language.isin(["multi","??"]).any())
print(t.groupby(["kind","metric","axes"]).size().unstack().fillna(0).astype(int).to_string())
print("fracs", sorted(t.frac.unique()))
r=t[(t.metric=="da")&t.retest.notna()]
k=(r.retest*r.n_pairs); off=(k-k.round()).abs()>1e-6
print("retest DA rows",len(r),"where retest*n_pairs not integer (different pair population):",off.sum(), r[off].groupby("kind").size().to_dict())
v=pd.read_csv(P+"surrogate_values.csv",low_memory=False)
print("values rows",len(v),"cols",v.shape[1],"n_pairs<3 rows",(v.n_pairs<3).sum(), "with any stat", v[v.n_pairs<3].drop(columns=["task","language","benchmark","kind","proxy_size","axes","tier","n_pairs"]).notna().any(axis=1).sum())
sub=v[v.n_pairs<3]; cols=[c for c in v.columns if c not in ("task","language","benchmark","kind","proxy_size","axes","tier","n_pairs")]
print("  stats present where n_pairs<MIN_PAIRS:", sub[cols].notna().sum().loc[lambda s:s>0].sort_values(ascending=False).head(12).to_dict())
# perm_p calibration and strat_ranks vs brute force
rng=np.random.default_rng(1); ps=[]
for _ in range(1500):
    x=rng.normal(size=15); y=rng.normal(size=15); ps.append(S.perm_p(rankdata(x),rankdata(y)))
ps=np.array(ps); print("perm_p null n=15: share<.05",(ps<.05).mean().round(3),"share<.01",(ps<.01).mean().round(3))
x=np.arange(12.); print("perm_p perfect n=12:",S.perm_p(rankdata(x),rankdata(x)), "scipy:",spearmanr(x,x).pvalue)
vv=rng.integers(0,5,60).astype(float); st=rng.integers(0,4,60)
br=np.empty(60)
for s in np.unique(st):
    m=st==s; br[m]=(rankdata(vv[m])-.5)/m.sum()
print("strat_ranks == brute:",np.allclose(S.strat_ranks(vv,st),br))
p=rng.random(50); p[3]=p[7]
q=S.bh(p); o=np.argsort(p); qq=np.minimum.accumulate((p[o]*50/np.arange(1,51))[::-1])[::-1]
print("bh ok",np.allclose(q[o],np.minimum(qq,1)))
