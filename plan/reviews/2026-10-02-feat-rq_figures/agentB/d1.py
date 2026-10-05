import pandas as pd, numpy as np, sys, re, warnings
warnings.simplefilter("ignore")
from scipy.stats import spearmanr, false_discovery_control
D=sys.argv[1]
c=pd.read_csv(D+"/surrogate_correlations.csv",low_memory=False,dtype={"L":str}); c["L"]=c.L.fillna("all")
print("rows",len(c),"with p",c.p.notna().sum(),"q<.05",(c.q<.05).sum())
hp=c.p.notna(); print("BH reproduces q:", np.allclose(false_discovery_control(c.p[hp].to_numpy()), c.q[hp].to_numpy()))
print("q<.05 by subset_type:", c[c.q<.05].subset_type.value_counts().to_dict())
print("p rows by subset_type:", c[hp].subset_type.value_counts().to_dict())
nf=c[c.subset_type!="filter"]; h2=nf.p.notna(); q2=false_discovery_control(nf.p[h2].to_numpy()); print("without filter rows: tests",h2.sum(),"q<.05",(q2<.05).sum())
nr=nf[~nf.subset_type.str.startswith("reliable")]; h3=nr.p.notna(); q3=false_discovery_control(nr.p[h3].to_numpy()); print("without filter+reliable rows: tests",h3.sum(),"q<.05",(q3<.05).sum())
b=c[(c.subset=="benchmarks")&(c.subset_type=="all")&(c["axes"]=="multi-axis")&(c.L=="all")&(c.metric=="da")]
P=b.pivot_table(index="surrogate",columns="t_kind",values="rho")
print(P.loc[["signal__range","signal__rel_std","signal__mpd","signal__dispersion","signal__rms_deviation","signal__rel_dispersion","sign_persistence","autocorr","monotonicity","cronbach_alpha","n_items","noise__kfold_abs","noise__kfold_rel","da_ckpt_mean","pseudo_ref_da"]].round(3))
cat=P[~P.index.str.contains("__")]; 
for k in ["size","goal","ckpt"]: print(k,"top8 catalogue:", cat[k].abs().sort_values(ascending=False).head(8).round(2).to_dict())
sig=[s[8:] for s in P.index if s.startswith("signal__")]
allsig=sorted(set(s.split("__")[1] for s in P.index if s.startswith("snr__")))
for k in ["size","goal","ckpt"]:
    kr=np.array([P.loc[f"snr__{a}__kfold_rel",k] for a in allsig]); cr=np.array([P.loc[f"snr__{a}__ckpt_rel",k] for a in allsig])
    print(k,"kfold_rel > ckpt_rel for", int(np.nansum(kr>cr)),"of",len(allsig),"; not:",[a for a,x,y in zip(allsig,kr,cr) if not x>y])
    ka=np.array([P.loc[[f"snr__{a}__kfold_rel",f"snr__{a}__kfold_abs"],k].max() for a in allsig]); ca=np.array([P.loc[[f"snr__{a}__ckpt_rel",f"snr__{a}__ckpt_abs"],k].max() for a in allsig]); print("   best-kfold > best-ckpt:",int(np.nansum(ka>ca)))
    g=P[P.index.str.startswith("snr__")][k]; print("   signed best",g.idxmax(),round(g.max(),3),"| abs best",g.abs().idxmax(),round(g[g.abs().idxmax()],3))
ce=pd.read_csv(D+"/da_all_retest_multi_axes.csv"); print(ce[["t_kind","metric","axes","L","kind","rho","rho_cells","n_units"]].round(3).to_string())
val=pd.read_csv(D+"/surrogate_validated.csv",low_memory=False); print("validated",len(val),(val.q_val<.05).sum(), val.subset_type.value_counts().to_dict()); print("val sig by type",val[val.q_val<.05].subset_type.value_counts().to_dict())
print("pseudo_ref_da rows among validated sig:", ((val.surrogate=="pseudo_ref_da")&(val.q_val<.05)).sum())
print("p_val min", val.p_val.min(), "BH ok", np.allclose(false_discovery_control(val.p_val.to_numpy()), val.q_val.to_numpy()))
byl=pd.read_csv(D+"/surrogates_by_L.csv"); print(byl.round(3).to_string())
# units range for caption
for k in ["size","goal","ckpt"]: print(k,"n_units min/max", b[b.t_kind==k].n_units.min(), b[b.t_kind==k].n_units.max())
# by-L truth mono vs multi identical?
t=pd.read_csv(D+"/surrogate_targets.csv",dtype={"L":str})
for L in ["8","15","30","all"]:
    a=t[(t.metric=="da")&(t.L==L)&(t["axes"]=="multi-axis")].set_index(["task","proxy_size","frac","kind"]); m=t[(t.metric=="da")&(t.L==L)&(t["axes"]=="mono-axis")].set_index(["task","proxy_size","frac","kind"])
    j=a.join(m,lsuffix="_a",rsuffix="_m",how="inner"); print("L",L,"rows multi",len(a),"mono",len(m),"common",len(j),"identical value share",np.isclose(j.value_a,j.value_m).mean().round(3),"n_pairs multi median",j.n_pairs_a.median(),"mono",j.n_pairs_m.median())
print("retest notna by axes/L:", t[t.retest.notna()].groupby(["axes","L","metric"]).size().to_dict())
