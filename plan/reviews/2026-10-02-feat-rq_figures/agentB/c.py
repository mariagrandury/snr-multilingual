import pandas as pd, numpy as np, sys
D=sys.argv[1]
c=pd.read_csv(D+"/surrogate_correlations.csv",low_memory=False, dtype={"L":str})
print(c.shape, c.columns.tolist())
print("with p:", c.p.notna().sum(), "q<.05:", (c.q<.05).sum())
print(c.subset_type.value_counts().to_dict())
sig=c[c.q<.05]; print("sig by subset_type", sig.subset_type.value_counts().to_dict())
print("p by subset_type", c[c.p.notna()].subset_type.value_counts().to_dict())
print("truth groups", c.groupby(["t_kind","metric","axes","L"]).ngroups)
b=c[(c.subset=="benchmarks")]
def row(tk,me,ax,L,s):
    r=b[(b.t_kind==tk)&(b.metric==me)&(b.axes==ax)&(b.L==L)&(b.surrogate==s)]
    return r[["rho","p","q","rho_cells","rho_w","n","n_units"]].round(4).to_dict("records")
for a in [("ckpt","da","multi-axis","all","sign_consistency_window"),("goal","da","multi-axis","all","sign_persistence"),("size","da","multi-axis","all","da_ckpt_mean"),
          ("size","da","multi-axis","all","snr__rel_std__ckpt_rel"),("size","da","multi-axis","all","snr__rel_dispersion__ckpt_rel"),("size","da","multi-axis","all","n_items"),
          ("size","da","multi-axis","all","snr__dispersion_shifted__kfold_abs"),("size","da","multi-axis","all","noise__kfold_abs"),("size","da","multi-axis","all","signal__dispersion_shifted"),
          ("size","da","multi-axis","all","pseudo_ref_da"),("ckpt","da","multi-axis","all","window_kendall"),("ckpt","tau_b","multi-axis","all","consecutive_kendall_late"),
          ("ckpt","da","multi-axis","all","snr__gini__kfold_rel"),("goal","da","multi-axis","all","snr__tukey__kfold_rel"),("size","da","multi-axis","all","signal__range"),("size","da","multi-axis","all","signal__rel_std"),("size","da","multi-axis","all","signal__mpd")]:
    print(a, row(*a))
# circularity: stage subsets
st=c[(c.subset_type=="stage")&(c.t_kind=="ckpt")&(c.axes=="multi-axis")&(c.L=="all")&c.surrogate.isin(["sign_consistency_window","window_kendall","consecutive_kendall_late","kendall_w_window","dior","crossings","consecutive_kendall","icc_window","gap_over_noise"])]
print(st.pivot_table(index=["surrogate"],columns=["metric","subset"],values="rho").round(2).to_string())
ce=pd.read_csv(D+"/da_all_retest_multi_axes.csv"); print(ce.round(3).to_string())
val=pd.read_csv(D+"/surrogate_validated.csv",low_memory=False); print(len(val),(val.q_val<.05).sum(), val.p_val.min(), (val.p_val<=val.p_val.min()+1e-12).sum(), val.subset_type.value_counts().to_dict())
print("val sig by type", val[val.q_val<.05].subset_type.value_counts().to_dict())
print("sign flips among sig:", (np.sign(val.rho_disc)!=np.sign(val.rho_val))[val.q_val<.05].sum())
print("shrink: median |disc|, |val| over all tested", val.rho_disc.abs().median(), val.rho_val.abs().median())
byl=pd.read_csv(D+"/surrogates_by_L.csv"); print(byl.round(3).to_string())
