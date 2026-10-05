import pandas as pd, numpy as np, sys
D=sys.argv[1]
v=pd.read_csv(D+"/surrogate_values.csv",low_memory=False)
print(v[v.task.str.contains("truthful")][["task","proxy_size","axes","chance","n_items","nonrandom","z_above_chance"]].drop_duplicates(["task"]).to_string())
print("n_items small:", v[v.n_items<100][["task","n_items"]].drop_duplicates().to_string())
print("benchmarks w/o n_items:", sorted(v[(v.kind=="benchmark")&v.n_items.isna()].benchmark.unique()))
for c,thr in [("gain_over_noise",1e6),("scale_gain_over_noise",1e6)]:
    b=v[v[c].abs()>thr]; print(c, len(b), b[["task","proxy_size","axes",c,"gap_over_noise","cronbach_alpha","icc_window","kendall_w_window","noise_to_binomial"]].head(8).to_string())
print("alpha>1:", (v.cronbach_alpha>1+1e-9).sum(), v[v.cronbach_alpha>1+1e-9][["task","proxy_size","axes","cronbach_alpha","icc_window","g_coefficient","eta2_window"]].head().to_string())
print("split_half<-1:", (v.split_half< -1).sum(), " g_coef<-1:", (v.g_coefficient<-1).sum(), "alpha<-1", (v.cronbach_alpha<-1).sum())
print("gap_over_noise==50:", (v.gap_over_noise==50).sum())
print("settling_time dist", v.settling_time.value_counts().sort_index().to_dict())
print(v[["noise__ckpt_rel","noise__ckpt_abs","noise__tukey_depth","noise__projection_depth","noise__kfold_rel","noise__kfold_abs"]].describe().T.to_string())
print("kind bpb kfold notna:", v[v.kind=="bpb"]["noise__kfold_abs"].notna().sum())
# n_pairs relation between axes
print(v.groupby(["axes","proxy_size"]).n_pairs.agg(["min","median","max"]))
print("rows w n_pairs<3:", (v.n_pairs<3).sum())
