import pandas as pd, numpy as np
P="/private/tmp/claude-501/-Users-mariagrandury-Projects-epfl-snr-multilingual/f7d6f8d9-67db-4234-9ff3-0c61069faa4c/scratchpad/review/wt/src/signal-and-noise/analysis/rq04_surrogates/pretraining/predictivity/"
c=pd.read_csv(P+"surrogate_correlations.csv",dtype={"L":str},low_memory=False)
b=c[(c.subset=="benchmarks")&(c.subset_type=="all")].copy()
b["truth"]=b.t_kind+"|"+b.metric+"|"+b["axes"].str[:5]+"|"+b.L
for s in ["signal__range","signal__rel_std","signal__rms_deviation","signal__mpd","signal__rel_dispersion"]:
    r=b[b.surrogate==s].set_index("truth").rho.round(2)
    print(s, r[r.index.str.endswith("|all")].to_dict()); print("    per-L:", r[~r.index.str.endswith("|all")].to_dict())
m=b[(b["axes"]=="multi-axis")&(b.L=="all")&(b.metric=="da")]
t=m.pivot_table(index="surrogate",columns="t_kind",values="rho")[["size","goal","ckpt"]].round(2)
print(t.loc[["cronbach_alpha","tie_rate_items","monotonicity","autocorr","sign_persistence","pseudo_ref_da","n_items","bpb_corr_training","noise__ckpt_rel","noise__kfold_rel","snr__rel_dispersion__ckpt_rel","snr__rel_dispersion__kfold_rel","signal__rel_dispersion"]].to_string())
nc=m[~m.surrogate.str.contains("__")]
for k in ("size","goal","ckpt"):
    x=nc[nc.t_kind==k]; x=x.reindex(x.rho.abs().sort_values(ascending=False).index)
    print(k, [(a,round(r,2)) for a,r in zip(x.surrogate.head(9),x.rho.head(9))])
# particular cases: recompute README selection
sig=c[c.q<.05]; sig=sig.reindex(sig.rho.abs().sort_values(ascending=False).index)
top=sig[sig.subset!="benchmarks"].groupby(["t_kind","metric","subset_type"],sort=False).head(2).head(30)
print("particular: |rho| range",top.rho.abs().min().round(2),top.rho.abs().max().round(2),"units",top.n_units.min(),top.n_units.max(), top.subset_type.value_counts().to_dict())
bl=pd.read_csv(P+"surrogates_by_L.csv"); print(bl[(bl["axes"]=="multi-axis")&(bl.t_kind=="size")][["L","mean","sd","tasks","surrogate","rho"]].round(3).to_string())
print("p==min perm p share among p<30 units:", (c[(c.n_units<30)&c.p.notna()].p<=1/20001+1e-12).mean().round(3), "n",((c.n_units<30)&c.p.notna()).sum(), "min q among perm rows", c[(c.n_units<30)].q.min())
print("rows with p but n_units<10:", ((c.n_units<10)&c.p.notna()).sum(), "; filter-row p NaN share", c[c.subset_type=="filter"].p.isna().mean().round(3))
