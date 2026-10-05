import numpy as np, pandas as pd, matplotlib, sys
P="/private/tmp/claude-501/-Users-mariagrandury-Projects-epfl-snr-multilingual/f7d6f8d9-67db-4234-9ff3-0c61069faa4c/scratchpad/review/wt/src/signal-and-noise/analysis/rq04_surrogates/pretraining/predictivity/"
t=pd.read_csv(P+"surrogate_targets.csv",dtype={"L":str})
a=t[(t.metric=="da")&(t.L=="all")]
print(a.groupby(["kind","axes"]).agg(rows=("value","size"),n_pairs_mean=("n_pairs","mean"),n_pairs_min=("n_pairs","min"),da=("value","mean"),retest_n=("retest","count")).round(3).to_string())
print("L values:", t.groupby(["L","axes"]).size().unstack().to_string())
w=a.pivot_table(index=["task","proxy_size","frac","kind"],columns="axes",values="value")
print("cells in both axes",w.dropna().shape[0],"only multi",w["mono-axis"].isna().sum(),"only mono",w["multi-axis"].isna().sum(),"identical value share",(w["mono-axis"]==w["multi-axis"]).mean().round(3))
v=pd.read_csv(P+"surrogate_values.csv",usecols=["task","proxy_size","axes","n_pairs","gap_over_noise","kind","chance","n_items"])
print(v.groupby("axes").n_pairs.describe()[["count","min","mean","max"]].to_string())
print("bpb rows with n_items/chance:", v[v.kind=="bpb"][["n_items","chance"]].notna().sum().to_dict())
import numpy; print(numpy.__version__, sys.version.split()[0], matplotlib.get_backend())
try:
    from threadpoolctl import threadpool_info; print([(i["internal_api"],i["num_threads"]) for i in threadpool_info()])
except Exception as e: print("no threadpoolctl", e)
