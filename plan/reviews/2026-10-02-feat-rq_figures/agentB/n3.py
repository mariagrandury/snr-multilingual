import pandas as pd, numpy as np
P="/private/tmp/claude-501/-Users-mariagrandury-Projects-epfl-snr-multilingual/f7d6f8d9-67db-4234-9ff3-0c61069faa4c/scratchpad/review/wt/src/signal-and-noise/analysis/rq04_surrogates/pretraining/predictivity/"
c=pd.read_csv(P+"surrogate_correlations.csv",dtype={"L":str},low_memory=False)
v=pd.read_csv(P+"surrogate_validated.csv",dtype={"L":str},low_memory=False)
print("validated",len(v),(v.q_val<.05).sum(), "filter among validated",(v.subset_type=="filter").sum(),"sig filter",((v.subset_type=="filter")&(v.q_val<.05)).sum(), "min p_val",v.p_val.min(), "reliable sig",((v.subset_type.str.startswith("reliable"))&(v.q_val<.05)).sum())
print("validated by surrogate top", v[v.q_val<.05].surrogate.value_counts().head(6).to_dict())
b=c[(c.subset=="benchmarks")&(c.subset_type=="all")]
# dispersion-alone claim across every truth
for s in ["signal__range","signal__rel_std","signal__mpd","signal__dispersion","signal__rel_dispersion"]:
    r=b[b.surrogate==s]
    print(s, "all-L truths:", r[r.L=="all"].rho.min().round(3), r[r.L=="all"].rho.max().round(3), "| every truth:", r.rho.min().round(3), r.rho.max().round(3), len(r))
print(sorted(x for x in b.surrogate.unique() if x.startswith("signal__")))
# kfold vs ckpt noise per signal
m=b[(b["axes"]=="multi-axis")&(b.L=="all")&(b.metric=="da")&b.surrogate.str.startswith("snr__")].copy()
m["sig"]=m.surrogate.str.split("__").str[1]; m["noi"]=m.surrogate.str.split("__").str[2]
for k in ("size","goal","ckpt"):
    t=m[m.t_kind==k].pivot_table(index="sig",columns="noi",values="rho")
    kf=t[["kfold_rel","kfold_abs"]].max(axis=1); ck=t[["ckpt_rel","ckpt_abs"]].max(axis=1)
    print(k,"signals",len(t),"kfold best > ckpt best for",(kf>ck).sum(),"; fails:",list(t.index[~(kf>ck)]), "best", t.stack().idxmax(), round(t.stack().max(),3))
    print("   kfold_rel>ckpt_rel:",(t.kfold_rel>t.ckpt_rel).sum(),"kfold_abs>ckpt_abs:",(t.kfold_abs>t.ckpt_abs).sum(), "depth cols max", t[["tukey_depth","projection_depth"]].max().round(2).to_dict())
# particular cases range
sig=c[(c.q<.05)&(c.subset!="benchmarks")]
print("pseudo_ref_da rows by proxy subset:", c[(c.surrogate=="pseudo_ref_da")&(c.subset_type=="proxy")].subset.unique())
vals=pd.read_csv(P+"surrogate_values.csv",usecols=["task","proxy_size","axes","pseudo_ref_da","prev_rung_da","ladder_da","kind","language","n_pairs"])
print(vals.groupby("proxy_size")[["pseudo_ref_da","prev_rung_da","ladder_da"]].count().to_string())
print("languages in values", vals.language.nunique(), sorted(vals.language.unique())[:60])
