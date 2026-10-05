import json,sys,subprocess
def show(rev,path): return subprocess.run(["git","show",f"{rev}:{path}"],capture_output=True,text=True).stdout
for c in subprocess.run(["git","log","--reverse","--format=%h","origin/main..HEAD","--","configs/tasks.json"],capture_output=True,text=True).stdout.split():
    a=json.loads(show(c+"~1","configs/tasks.json")); b=json.loads(show(c,"configs/tasks.json"))
    out=[]
    for g in sorted(set(a['groups'])|set(b['groups'])):
        x,y=a['groups'].get(g),b['groups'].get(g)
        if x is None: out.append(f"+group {g}({len(y)})")
        elif y is None: out.append(f"-GROUP {g} {x}")
        else:
            r=[t for t in x if t not in y]; ad=[t for t in y if t not in x]
            if r or ad: out.append(f"{g}: removed {r} added {len(ad)}")
    rt=sorted(set(a['tasks'])-set(b['tasks']))
    ch={}
    for t in set(a['tasks'])&set(b['tasks']):
        if a['tasks'][t]!=b['tasks'][t]:
            k=tuple(sorted(k for k in set(a['tasks'][t])|set(b['tasks'][t]) if a['tasks'][t].get(k)!=b['tasks'][t].get(k))); ch[k]=ch.get(k,0)+1
    print(c,'|','; '.join(out)[:1100],'| tasks removed',len(rt),rt[:8],'| entries changed',ch)
