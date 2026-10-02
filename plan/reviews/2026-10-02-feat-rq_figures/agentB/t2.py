import numpy as np, pandas as pd, warnings, time
warnings.simplefilter("ignore")
from analysis.rq04_surrogates import catalogue as C
from analysis.utils import *
from analysis.rq00_gate_and_curves.above_random import load_mask, task_chance
t0=time.time()
df=ladder_frame("predictivity"); print(df.shape, time.time()-t0, df.columns.tolist())
print("kinds", df["kind"].value_counts().to_dict()); print("seeds", df.seed.unique(), "sizes", df["size"].unique())
print("families per size", df.groupby("size").family.nunique().to_dict())
print("models per (size,family) max", df.groupby(["size","family"]).model.nunique().max())
win=noise_checkpoints(df).assign(p=lambda d:(d["frac"]*NOISE_GRID).round().astype(int))
npts=win.groupby(["size","model","task"]).p.nunique()
print("window points per (model,task):", npts.value_counts().to_dict())
print(npts.groupby("size").apply(lambda s: s.value_counts().to_dict()).to_dict())
b=win[win.kind=="benchmark"] if "benchmark" in set(df.kind) else win
ax=design_axes(df); ps=pair_sets(ax); print({k:len(v) for k,v in ps.items()})
# families at reference vs proxies
ref=set(df[df["size"]==TARGET_SIZE].family)
for s in SMALL_SIZES: print(s, "families not at reference:", sorted(set(df[df["size"]==s].family)-ref))
print("bpb tasks:", sorted(t for t in df.task.unique() if t.startswith("bpb_"))[:60])
langs={}
for t in df.task.unique():
    if t.startswith("bpb_") and t!="bpb_macro": langs.setdefault(assign_language(t),[]).append(t)
print("languages with >1 bpb task:", {k:v for k,v in langs.items() if len(v)>1})
df.to_pickle(__import__("sys").argv[1])
