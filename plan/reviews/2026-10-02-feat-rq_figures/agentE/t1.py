import pandas as pd, numpy as np, collections
from analysis import utils as U
from evals.scripts.utils.configs import load_pools
print({k:(v.get('members'), v.get('stage')) for k,v in load_pools().items() if 'predictivity' in k})
print("MIN_PAIRS",U.MIN_PAIRS,"ANALYSIS_SIZES",U.ANALYSIS_SIZES,"SMALL",U.SMALL_SIZES,"TARGET",U.TARGET_SIZE, U.CKPT_DA_EARLY_FRACS)
for pool in ["predictivity","predictivity_schemes","predictivity_all","predictivity_seeds","predictivity_seeds_train"]:
    if pool not in load_pools(): print("no pool",pool); continue
    df = U.build_snr_pool(pool)
    a = U.design_axes(df)
    ps = U.pair_sets(a)
    print(pool, len(a), {k:len(v) for k,v in ps.items()}, sorted(df['size'].unique()), sorted(df.seed.unique()))
    c = collections.Counter()
    for x,y in ps["mono-axis"]:
        d=[k for k in U.DESIGN_AXES if a.loc[x,k]!=a.loc[y,k]]; c[d[0]]+=1
    print("  mono by axis", dict(c))
    if pool=="predictivity_schemes":
        print(a.drop(columns=[]).to_string())
        print([p for p in ps["mono-axis"] if any("L1-" in q for q in p) ])
    # leak check
    print("  models w/o -b at 90M/175M:", sorted(set(m for m in df.model if (m.startswith("lm-90M") or m.startswith("lm-175M")) and "-b" not in m))[:5])
    print("  tokens/step by size", df.assign(t=df.tokens/df.step).groupby("size").t.unique().to_dict())
