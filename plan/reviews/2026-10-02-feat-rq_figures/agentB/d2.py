import pandas as pd, numpy as np, sys, re, warnings
warnings.simplefilter("ignore")
from scipy.stats import spearmanr
D=sys.argv[1]
v=pd.read_csv(D+"/surrogate_values.csv",low_memory=False); t=pd.read_csv(D+"/surrogate_targets.csv",dtype={"L":str})
TW=re.compile(r"^(rfgm|rf)_(.+)$")
x=t[(t.kind=="ckpt")&(t.metric=="da")&(t["axes"]=="multi-axis")&(t.L=="all")].drop(columns=["language","n_pairs","kind"]).merge(v[v["axes"]=="multi-axis"],on=["task","proxy_size"])
x=x[x.kind=="benchmark"]; x["cluster"]=x.benchmark.str.replace(TW,r"\2",regex=True)+"|"+x.language
S=["sign_consistency_window","window_kendall","kendall_w_window","dior","consecutive_kendall","consecutive_kendall_late","crossings","cronbach_alpha","autocorr","gain_over_noise","n_items"]
x[S]=x[S].replace([np.inf,-np.inf],np.nan)
rows=[]
for f,g in x.groupby("frac"):
    cm=g.groupby("cluster")[["value"]+S].mean()
    rows.append({"frac":f, **{s: round(spearmanr(cm[s],cm.value,nan_policy="omit").statistic,2) for s in S}})
print(pd.DataFrame(rows).to_string(index=False))
print("twin regex in grids:", __import__("analysis.grids",fromlist=["x"])._TWIN.pattern)
# twins as language_consensus peers: how many benchmark rows have a twin in same language
vb=v[(v.kind=="benchmark")&(v["axes"]=="multi-axis")&(v.proxy_size=="1B")]
base=vb.benchmark.str.replace(TW,r"\2",regex=True); k=(base+"|"+vb.language)
print("1B benchmark tasks:",len(vb),"with >=1 twin in the same (base benchmark, language):",int((k.map(k.value_counts())>1).sum()))
print("spearman language_consensus/item_total_corr summary for twin-having vs not:")
has=(k.map(k.value_counts())>1)
print(vb.assign(has=has.values).groupby("has")[["language_consensus","item_total_corr"]].agg(["mean","count"]).round(3))
