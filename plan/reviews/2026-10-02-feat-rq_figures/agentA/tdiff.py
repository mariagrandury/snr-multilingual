import json,sys
x=json.load(open(sys.argv[1])); y=json.load(open(sys.argv[2]))
for g in sorted(set(x['groups'])|set(y['groups'])):
    gx=x['groups'].get(g,[]); gy=y['groups'].get(g,[])
    rem=[t for t in gx if t not in gy]; add=[t for t in gy if t not in gx]
    if rem or add: print(f"  group {g}: -{rem} +{add}")
print("  tasks +%d -%d"%(len(set(y['tasks'])-set(x['tasks'])),len(set(x['tasks'])-set(y['tasks']))), "bench +%s -%s"%(sorted(set(y['benchmarks'])-set(x['benchmarks'])),sorted(set(x['benchmarks'])-set(y['benchmarks']))))
ch={}
for t in set(x['tasks'])&set(y['tasks']):
    if x['tasks'][t]!=y['tasks'][t]:
        keys=tuple(sorted(k for k in set(x['tasks'][t])|set(y['tasks'][t]) if x['tasks'][t].get(k)!=y['tasks'][t].get(k)))
        ch.setdefault(keys,[]).append(t)
for k,v in ch.items(): print("  meta changed",k,len(v),sorted(v)[:4], {kk:(x['tasks'][v[0]].get(kk),y['tasks'][v[0]].get(kk)) for kk in k})
for b in set(x['benchmarks'])&set(y['benchmarks']):
    if x['benchmarks'][b]!=y['benchmarks'][b]: print("  bench changed",b)
