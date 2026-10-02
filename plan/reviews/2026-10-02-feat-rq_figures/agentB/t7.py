import numpy as np, pandas as pd, sys, warnings
from analysis.rq04_surrogates import catalogue as C
from analysis import grids as G
from analysis.rq00_gate_and_curves.above_random import task_chance, task_n_options
warnings.simplefilter("ignore")
from analysis.utils import *
df=ladder_frame("predictivity")
print(df.shape, "seeds", sorted(df.seed.unique()), "sizes", sorted(df["size"].unique()))
print("models per (size,family) max", df.groupby(["size","family"]).model.nunique().max())
win=noise_checkpoints(df).assign(p=lambda d:(d["frac"]*NOISE_GRID).round().astype(int))
npts=win[~win.task.str.startswith("bpb")&(win.task!="train_loss")].groupby(["size","model","task"]).p.nunique()
print("benchmark window points per (model,task) by size:"); print(npts.groupby("size").apply(lambda s: s.value_counts().to_dict()).to_dict())
langs={}
for t in df.task.unique():
    if t.startswith("bpb_") and t!="bpb_macro": langs.setdefault(assign_language(t),[]).append(t)
print("languages with >1 bpb task:", {k:v for k,v in langs.items() if len(v)>1})
print("kinds", df["kind"].value_counts().to_dict())
print("frac max per model !=1:", (df.groupby("model").frac.max()!=1).sum())
D=sys.argv[2]
print("families:", benchmark_family("rf_belebele_fra_Latn"), benchmark_family("belebele_fra_Latn"), assign_language("rf_belebele_fra_Latn"))
# 1 float noise case
for t,s in [("multiblimp_dan","350M"),("bbh_cloze_sports_understanding","600M"),("acp_bench_cloze_app","600M")]:
    g=noise_checkpoints(df[(df.task==t)&(df["size"]==s)])
    W=g.assign(p=(g.frac*20).round()).pivot_table(index="family",columns="p",values="primary_score")
    sd=np.nanstd(W.to_numpy(),axis=1,ddof=1); print(t,s,"window sd per family:",np.unique(sd)[:6], "distinct scores:", np.unique(W.to_numpy())[:6], "n fam", len(W))
# 2 chance mismatch
for t in ["truthfulqa_mc2","truthfulqa-multi_mc1_en"]: print(t, "task_chance", task_chance(t), "1/n_opt", 1/task_n_options(t))
# 3 cluster dependence
v=pd.read_csv(D+"/surrogate_values.csv",low_memory=False); tg=pd.read_csv(D+"/surrogate_targets.csv",dtype={"L":str})
x=tg[(tg.kind=="size")&(tg.metric=="da")&(tg["axes"]=="multi-axis")&(tg.L=="all")].merge(v[v["axes"]=="multi-axis"],on=["task","proxy_size"])
x=x[x.kind_y=="benchmark"]; x["base"]=x.benchmark.str.replace(G._TWIN,r"\2",regex=True); x["cluster"]=x.base+"|"+x.language_y
cl=x.groupby(["cluster","base","language_y"]).agg(da=("value","mean"),n_items=("n_items","mean"),da_ckpt_mean=("da_ckpt_mean","mean"),kf=("noise__kfold_abs","mean")).reset_index()
print("clusters",len(cl),"benchmarks",cl.base.nunique(),"languages",cl.language_y.nunique(), "distinct n_items values", cl.n_items.round().nunique())
print(cl.base.value_counts().head(12).to_dict())
from scipy.stats import spearmanr
def icc1(d,col,by):
    g=d.groupby(by)[col]; k=g.size().mean(); msb=(g.size()*(g.mean()-d[col].mean())**2).sum()/(g.ngroups-1); msw=((d[col]-g.transform("mean"))**2).sum()/(len(d)-g.ngroups); return (msb-msw)/(msb+(k-1)*msw)
for col in ["da","n_items","da_ckpt_mean","kf"]:
    print(col,"ICC by benchmark",round(icc1(cl.dropna(subset=[col]),col,"base"),3),"ICC by language",round(icc1(cl.dropna(subset=[col]),col,"language_y"),3))
bm=cl.groupby("base")[["da","n_items","da_ckpt_mean"]].mean(); print("benchmark-level (n=%d) spearman da~n_items"%len(bm), spearmanr(bm.da,bm.n_items,nan_policy="omit"), "da~da_ckpt_mean", spearmanr(bm.da,bm.da_ckpt_mean,nan_policy="omit"))
print("cluster-level da~n_items", spearmanr(cl.da,cl.n_items,nan_policy="omit"))
# within-benchmark (demeaned ranks) spearman
for col in ["n_items","da_ckpt_mean"]:
    d=cl.dropna(subset=[col]); r=d.groupby("base")[["da",col]].rank(pct=True); m=d.groupby("base")["da"].transform("size")>=3
    print(col,"within-benchmark rank corr", np.corrcoef(r.da[m],r[col][m])[0,1], "n",m.sum())
# 4 rq03 identity
import glob
f=glob.glob(D.replace("rq04_surrogates","rq03_noise_and_snr")+"/snr_variants_per_task.csv"); print(f)
if f:
    r3=pd.read_csv(f[0]); cols=[c for c in r3.columns if c.startswith("snr_rel_std_")]; print(cols[:8])
    m=v[v["axes"]=="multi-axis"][["task","proxy_size","snr__rel_std__ckpt_rel"]]
    for s in ["90M","1B"]:
        c="snr_rel_std_"+s
        if c in r3: 
            j=m[m.proxy_size==s].merge(r3[["task",c]],on="task"); ok=j.dropna(); print(s,len(j),"both finite",len(ok),"max abs diff",(ok.iloc[:,2]-ok[c]).abs().max(), "rq03 NaN but cat finite", (j[c].isna()&j.iloc[:,2].notna()).sum())
