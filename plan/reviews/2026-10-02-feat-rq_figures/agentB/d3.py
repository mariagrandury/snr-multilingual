import pandas as pd, numpy as np, sys, warnings
warnings.simplefilter("ignore")
from scipy.stats import false_discovery_control
D=sys.argv[1]
c=pd.read_csv(D+"/surrogate_correlations.csv",low_memory=False,dtype={"L":str}); c["L"]=c.L.fillna("all")
b=c[(c.subset=="benchmarks")&(c.subset_type=="all")]
d=b[b.surrogate.isin(["signal__range","signal__rel_std","signal__mpd"])]
print("range/rel_std/mpd over all",d.groupby(["t_kind","metric","axes","L"]).ngroups,"truths: min",d.rho.min().round(3),"max",d.rho.max().round(3),"n positive",(d.rho>0).sum(),"of",len(d))
m=d[(d.L=="all")&(d.metric=="da")&(d["axes"]=="multi-axis")]; print(" 3 main DA truths: min",m.rho.min().round(3),"max",m.rho.max().round(3))
print(d[d.rho>0][["t_kind","metric","axes","L","surrogate","rho"]].round(2).to_string())
nf=c[c.subset_type!="filter"].copy(); h=nf.p.notna(); nf.loc[h,"q2"]=false_discovery_control(nf.p[h].to_numpy())
print("non-filter rows sig with filter rows in family:",(nf.q<.05).sum(),"| family without them:",(nf.q2<.05).sum())
t=nf[(nf.subset=="benchmarks")&(nf.subset_type=="all")&(nf["axes"]=="multi-axis")&(nf.L=="all")&(nf.metric=="da")]
print("paper-table cells: bold now",(t.q<.05).sum(),"bold without filter rows",(t.q2<.05).sum(),"of",len(t), "; flips:", t[(t.q<.05)!=(t.q2<.05)][["t_kind","surrogate","rho","p","q","q2"]].round(4).to_string())
print("largest p still q<.05 (current):", c[c.q<.05].p.max(), "| without filters:", nf[nf.q2<.05].p.max())
f=pd.read_csv(D+"/surrogate_filters.csv",low_memory=False,usecols=["basis","surrogate","p","rho"]); print("filters file basis counts",f.basis.value_counts().to_dict())
s=pd.read_csv(D+"/surrogates_significant.csv"); print("surrogates_significant rows",len(s),"filter rows",(s.subset_type=="filter").sum(), "reliable rows",(s.subset_type.str.startswith("reliable")).sum(), "pseudo_ref",(s.surrogate=="pseudo_ref_da").sum())
