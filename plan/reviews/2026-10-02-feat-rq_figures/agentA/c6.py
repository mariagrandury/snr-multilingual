import json,sys,collections
SC=sys.argv[1]
m=json.load(open(SC+"/tasks_main.json")); h=json.load(open(SC+"/tasks_head.json"))
for g in sorted(set(m['groups'])|set(h['groups'])):
    a,b=m['groups'].get(g),h['groups'].get(g)
    if a is None: print('GROUP new',g,len(b)); continue
    if b is None: print('GROUP DELETED',g,len(a),a[:12]); continue
    ra=[x for x in a if x not in b]; ad=[x for x in b if x not in a]
    print('group',g,'main',len(a),'head',len(b),'removed',ra,'added',len(ad), ad if len(ad)<45 else ad[:8]+['...'])
rem=sorted(set(m['tasks'])-set(h['tasks'])); print('tasks removed',len(rem),rem[:40])
ch=collections.defaultdict(list)
for t in set(m['tasks'])&set(h['tasks']):
    x,y=m['tasks'][t],h['tasks'][t]
    d=tuple(sorted(k for k in set(x)|set(y) if x.get(k)!=y.get(k)))
    if d: ch[d].append(t)
for d,ts in ch.items():
    t=sorted(ts)[0]; print('task fields changed',d,len(ts),'e.g.',t,{k:(m['tasks'][t].get(k),h['tasks'][t].get(k)) for k in d}); print('   benchmarks:',dict(collections.Counter(h['tasks'][x].get('benchmark') for x in ts)))
print('benchmarks removed',sorted(set(m['benchmarks'])-set(h['benchmarks'])))
for b in set(m['benchmarks'])&set(h['benchmarks']):
    if m['benchmarks'][b]!=h['benchmarks'][b]: print('benchmark meta changed',b,{k:(m['benchmarks'][b].get(k),h['benchmarks'][b].get(k)) for k in set(m['benchmarks'][b])|set(h['benchmarks'][b]) if m['benchmarks'][b].get(k)!=h['benchmarks'][b].get(k)})
