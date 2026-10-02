import pandas as pd, numpy as np, itertools, collections
from analysis import utils as U
from snr.metrics import decision_acc_fast
from pretrain.launch_trainings import DATA_SCHEMES, mix_label
import inspect
print(inspect.getsource(decision_acc_fast)[-900:])
# synthetic grid: every registered (scheme, L, arch) at seed 1904 + replicate seeds for L1/L2/L50 deep A
rows=[]
for sch,d in DATA_SCHEMES.items():
    for L in sorted(d["langs"]):
        for arch in d.get("arches_by_L",{}).get(L, d["arches"]):
            seeds=[1904]+([64,313] if sch=="A" and arch=="deep" and L in (1,2,50) else [])
            for sd in seeds:
                rows.append(dict(family=f"lm-{mix_label(L,arch,sch)}-seed{sd}",L=L,arch=arch,scheme=sch,seed=sd))
df=pd.DataFrame(rows)
a=U.design_axes(df)
print(a.drop(columns="seed").drop_duplicates().to_string())
ps=U.pair_sets(a)
print({k:len(v) for k,v in ps.items()})
c=collections.Counter()
for x,y in ps["mono-axis"]:
    d=[k for k in U.DESIGN_AXES if a.loc[x,k]!=a.loc[y,k]]; assert len(d)==1; c[d[0]]+=1
print("mono by axis",dict(c))
n1904=(a.seed==1904).sum(); print("families@1904",n1904,"C(n,2)",n1904*(n1904-1)//2)
print("L1 mono pairs:",[p for p in ps["mono-axis"] if any("-L1-" in q or q.startswith("lm-L1-") for q in p)])
print("L1 vs L2 pairs in multi but not mono:",[p for p in ps["multi-axis"] if p not in ps["mono-axis"] and any(q.startswith("lm-L1-") for q in p) and any(q.startswith("lm-L2-") for q in p)])
print("null:",ps["seed"])
# pool without grid seed
b=a[a.seed!=1904]; pb=U.pair_sets(b); print("no-1904 pool:",{k:len(v) for k,v in pb.items()}, pb["multi-axis"][:4])
assert all(b.loc[x,"seed"]==b.loc[y,"seed"] for x,y in pb["multi-axis"])
# pair_agreement vs kernel, random with ties
rng=np.random.default_rng(0); bad=0
for _ in range(3000):
    n=rng.integers(3,8); s=rng.integers(0,4,n).astype(float); t=rng.integers(0,4,n).astype(float)
    fam=[f"f{i}" for i in range(n)]
    da,np_=U.pair_agreement(dict(zip(fam,s)),dict(zip(fam,t)))
    k=decision_acc_fast(s,t)
    if not np.isclose(da,k): bad+=1
print("pair_agreement != decision_acc_fast in",bad,"of 3000")
print("2 fams:",U.pair_agreement({"a":1,"b":2},{"a":1,"b":2}), "explicit 2 pairs:",U.pair_agreement({"a":1,"b":2,"c":3},{"a":1,"b":2,"c":3},[("a","b"),("b","c")]))
print("NaN score:",U.pair_agreement({"a":1,"b":np.nan,"c":3},{"a":1,"b":2,"c":3}))
# one_axes
t=pd.DataFrame({"task":["x","x","x"],"axes":["multi-axis","mono-axis","seed"],"v":[1,2,3]})
print(U.one_axes(t).to_dict("records"), U.one_axes(t,"mono-axis").to_dict("records"))
# passes_gate
m=pd.DataFrame({"350M":pd.array([1,0,pd.NA],dtype="Int64"),"1.7B":pd.array([1,1,pd.NA],dtype="Int64")},index=["a","b","bpb_x"])
print(U.passes_gate(m,["a","b","bpb_x","unknown","a"],"350M","1.7B").tolist(), U.passes_gate(None,["a"],"350M").tolist(), U.passes_gate(m,["b"],"3B").tolist())
# noise_checkpoints
fr=[.5,.8,.85,.897,.9,.95,1.0,.75,.801]
d=pd.DataFrame({"model":"m","task":"t","frac":fr,"primary_score":range(len(fr))})
print(sorted(U.noise_checkpoints(d).frac.tolist()))
print(U.size_order(["1.7B","175M","1B","350M","600M","90M","3B","7-9B"]), U.ANALYSIS_SIZES)
# agreement_measures identity
for _ in range(500):
    n=rng.integers(3,9); s=rng.integers(0,4,n); t=rng.integers(0,4,n)
    r=U.agreement_measures(s,t)
    assert np.isclose(2*r["da"]-1, r["tau_a"]+ (r["tied_both"]-r["tied_one"])/r["n_pairs"])
    assert np.isclose(r["da"], decision_acc_fast(s.astype(float),t.astype(float)))
print("agreement_measures ok")
# jackknife
dec=pd.DataFrame([dict(g=1,family_a=x,family_b=y,match=int(rng.random()<.7)) for x,y in itertools.combinations("abcde",2)])
print(U.jackknife_ratio(dec,["g"]).to_dict("records"))
