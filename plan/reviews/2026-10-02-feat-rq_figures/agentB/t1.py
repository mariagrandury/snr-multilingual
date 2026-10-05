import numpy as np, pandas as pd, warnings
warnings.simplefilter("ignore")
from analysis.rq04_surrogates import catalogue as C, search as Srch
from scipy.stats import false_discovery_control, spearmanr, rankdata
rng=np.random.default_rng(0)
# 1. kfold closed form vs simulation
n,k,p=1000,5,0.3
items=np.zeros(n); items[:int(p*n)]=1
v0=[];v1=[];sd0=[]
for _ in range(20000):
    f=rng.permutation(items).reshape(k,-1).mean(1)
    v0.append(f.var()); v1.append(f.var(ddof=1)); sd0.append(f.std())
print("kfold: closed", p*(1-p)*(k-1)/(n-1), "sim var ddof0", np.mean(v0), "ddof1", np.mean(v1), "| sqrt(closed)", np.sqrt(p*(1-p)*(k-1)/(n-1)), "E[std ddof0]", np.mean(sd0))
print(C.kfold_noise(np.array([0.3,0.3]),1000), C.kfold_noise(np.array([0.3,0.0]),1000), C.kfold_noise(np.array([-2.0]),1000))
# 2. window anova
w=rng.normal(size=(8,5))+rng.normal(size=(8,1))*2
out=C._window_anova(w)
n_,m_=w.shape
gm=w.mean(); ssb=m_*((w.mean(1)-gm)**2).sum(); ssw=((w-w.mean(1,keepdims=True))**2).sum()
msb=ssb/(n_-1); msw=ssw/(n_*(m_-1))
print("icc", out["icc_window"], (msb-msw)/(msb+(m_-1)*msw), "eta2", out["eta2_window"], ssb/(ssb+ssw))
# cronbach via cov
cv=np.cov(w.T); print("alpha", out["cronbach_alpha"], m_/(m_-1)*(1-np.trace(cv)/cv.sum()))
# ICC(3,k) two-way consistency = alpha
ssc=n_*((w.mean(0)-gm)**2).sum(); sse=ssw-ssc; mse=sse/((n_-1)*(m_-1)); print("ICC(3,k)", (msb-mse)/msb, "g_coef(ICC1k)", out["g_coefficient"])
# Kendall W with ties
wt=np.round(w,0); o=C._window_anova(wt); 
R=np.apply_along_axis(rankdata,0,wt); Rs=R.sum(1); Sv=((Rs-Rs.mean())**2).sum()
T=sum(((np.unique(wt[:,j],return_counts=True)[1])**3-(np.unique(wt[:,j],return_counts=True)[1])).sum() for j in range(m_))
print("W code", o["kendall_w_window"], "W tie-corrected", 12*Sv/(m_**2*(n_**3-n_)-m_*T))
# all-tied: W
print("const cols:", C._window_anova(np.tile(np.arange(5.),(6,1))))
x=np.full((6,5),0.1+0.2); x=x+0; print("identical values:", C._window_anova(x))
y=np.tile(np.array([[0.31],[0.31],[0.47],[0.47],[0.13],[0.29]]),(1,5)); print("no within var:", C._window_anova(y))
# 3. strat_ranks, bh
v=rng.normal(size=50); st=rng.integers(0,4,50); v[:10]=np.round(v[:10])
r=Srch.strat_ranks(v,st); ref=np.empty(50)
for s in np.unique(st):
    m=st==s; ref[m]=(rankdata(v[m])-0.5)/m.sum()
print("strat_ranks ok", np.allclose(r,ref))
p=rng.uniform(size=200)**3; print("bh ok", np.allclose(Srch.bh(p), false_discovery_control(p,method="bh")))
pn=p.copy(); pn[3]=np.nan; print("bh with NaN:", np.isnan(Srch.bh(pn)).sum(), "nan out; finite q equal to clean?", np.allclose(np.delete(Srch.bh(pn),3), false_discovery_control(np.delete(p,3)),equal_nan=True))
# perm_p
a=rankdata(rng.normal(size=12)); b=rankdata(rng.normal(size=12)); print("perm_p", Srch.perm_p(a,b), "scipy t", spearmanr(a,b).pvalue, "perfect:", Srch.perm_p(a,a), spearmanr(a,a).pvalue)
# null calibration of perm_p at n=10
ps=[Srch.perm_p(rankdata(rng.normal(size=10)),rankdata(rng.normal(size=10))) for _ in range(2000)]; print("null P(p<.05)", np.mean(np.array(ps)<.05))
