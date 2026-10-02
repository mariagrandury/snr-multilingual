import pandas as pd, numpy as np
P="/private/tmp/claude-501/-Users-mariagrandury-Projects-epfl-snr-multilingual/f7d6f8d9-67db-4234-9ff3-0c61069faa4c/scratchpad/review/wt/src/signal-and-noise/analysis/rq04_surrogates/pretraining/predictivity/"
c=pd.read_csv(P+"surrogate_correlations.csv",dtype={"L":str},low_memory=False)
print("rows",len(c),"p",c.p.notna().sum(),"q<.05",(c.q<.05).sum())
print("subset_type counts\n",c.subset_type.value_counts().to_string())
print("filter rows q<.05",((c.subset_type=="filter")&(c.q<.05)).sum(), "non-filter q<.05",((c.subset_type!="filter")&(c.q<.05)).sum())
print("truths",c.groupby(["t_kind","metric","axes","L"]).ngroups)
nf=c[c.subset_type!="filter"]
print("surrogates (non-filter)",nf.surrogate.nunique())
d=pd.read_csv(P+"surrogate_definitions.csv"); print("defs",len(d)); print(d.family.value_counts().to_string())
print("snr__ in corr",nf.surrogate[nf.surrogate.str.startswith("snr__")].nunique())
print(set(d.surrogate)-set(nf.surrogate), set(nf.surrogate)-set(d.surrogate))
b=c[(c.subset=="benchmarks")&(c.subset_type=="all")]
m=b[(b["axes"]=="multi-axis")&(b.L=="all")&(b.metric=="da")]
def g(k,s):
    r=m[(m.t_kind==k)&(m.surrogate==s)]
    return None if r.empty else (round(r.rho.iloc[0],3), round(r.rho_w.iloc[0],3), int(r.n_units.iloc[0]), r.q.iloc[0])
for k in ("size","goal","ckpt"):
    print(k,"units range",m[m.t_kind==k].n_units.min(),m[m.t_kind==k].n_units.max())
    for s in ["sign_consistency_window","sign_persistence","da_ckpt_mean","n_items","snr__rel_std__ckpt_rel","snr__rel_dispersion__ckpt_rel","snr__dispersion_shifted__kfold_abs","noise__kfold_abs","signal__dispersion_shifted","signal__range","signal__std","signal__mpd","signal__dispersion","icc_window","g_coefficient","window_kendall","pseudo_ref_da","consecutive_kendall_late"]:
        print("  ",s,g(k,s))
    t=m[m.t_kind==k].copy(); t=t.reindex(t.rho.abs().sort_values(ascending=False).index)
    print(t[["surrogate","rho","rho_w","n_units"]].head(6).round(3).to_string())
