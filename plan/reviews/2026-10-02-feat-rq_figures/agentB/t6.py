import numpy as np, pandas as pd, sys, warnings
warnings.simplefilter("ignore")
from analysis.utils import *
df=pd.read_pickle(sys.argv[1])
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
