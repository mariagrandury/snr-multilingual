import pandas as pd, numpy as np, sys
D=sys.argv[1]
c=pd.read_csv(D+"/surrogate_correlations.csv",low_memory=False, dtype={"L":str})
c["L"]=c.L.fillna("all")
print(c.L.unique(), c.t_kind.unique(), c.metric.unique())
b=c[(c.subset=="benchmarks")&(c.subset_type=="all")]
def row(tk,me,ax,L,s):
    r=b[(b.t_kind==tk)&(b.metric==me)&(b["axes"]==ax)&(b.L==L)&(b.surrogate==s)]
    return r[["rho","p","q","rho_cells","rho_w","n","n_units"]].round(4).to_dict("records")
for a in [("ckpt","da","multi-axis","all","sign_consistency_window"),("goal","da","multi-axis","all","sign_persistence"),("size","da","multi-axis","all","da_ckpt_mean"),
          ("size","da","multi-axis","all","snr__rel_std__ckpt_rel"),("size","da","multi-axis","all","snr__rel_dispersion__ckpt_rel"),("size","da","multi-axis","all","n_items"),
          ("size","da","multi-axis","all","snr__dispersion_shifted__kfold_abs"),("size","da","multi-axis","all","noise__kfold_abs"),("size","da","multi-axis","all","signal__dispersion_shifted"),
          ("size","da","multi-axis","all","pseudo_ref_da"),("ckpt","da","multi-axis","all","window_kendall"),("ckpt","tau_b","multi-axis","all","consecutive_kendall_late"),
          ("ckpt","da","multi-axis","all","snr__gini__kfold_rel"),("goal","da","multi-axis","all","snr__tukey__kfold_rel"),("size","da","multi-axis","all","signal__range"),("size","da","multi-axis","all","signal__rel_std"),("size","da","multi-axis","all","signal__mpd"),("size","da","multi-axis","all","settling_time"),("size","da","multi-axis","all","gain_over_noise"),("size","da","multi-axis","all","split_half")]:
    print(a, row(*a))
st=c[(c.subset_type=="stage")&(c.t_kind=="ckpt")&(c["axes"]=="multi-axis")&(c.L=="all")&c.surrogate.isin(["sign_consistency_window","window_kendall","consecutive_kendall_late","kendall_w_window","dior","crossings","consecutive_kendall","icc_window","gap_over_noise","monotonicity","n_items"])]
print(st.pivot_table(index=["surrogate"],columns=["metric","subset"],values="rho").round(2).to_string())
# share of main p-values from t approx with p==0
print("p==0:", (c.p==0).sum(), "min p>0", c.p[c.p>0].min())
print("n_units distribution with p:", c[c.p.notna()].n_units.describe().to_dict())
print("perm branch count:", ((c.n_units>=10)&(c.n_units<30)&c.p.notna()).sum(), "sig in perm branch:", ((c.n_units<30)&(c.q<.05)).sum())
# filter rows: indicator
f=c[c.subset_type=="filter"]; print("filter indicator rows:", (f.surrogate=="[filter indicator]").sum(), "sig:", ((f.surrogate=="[filter indicator]")&(f.q<.05)).sum())
print("basis values", c.basis.value_counts(dropna=False).to_dict())
# DA-size dispersion signals
d=b[(b.t_kind=="size")&(b.metric=="da")&(b["axes"]=="multi-axis")&(b.L=="all")&b.surrogate.str.startswith("signal__")][["surrogate","rho"]].sort_values("rho"); print(d.round(2).to_string())
