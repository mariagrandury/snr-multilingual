import json,subprocess,collections,re
def show(rev): return json.loads(subprocess.run(["git","show",f"{rev}:configs/models.json"],capture_output=True,text=True).stdout)
cs=subprocess.run(["git","log","--reverse","--format=%h","origin/main..HEAD","--","configs/models.json"],capture_output=True,text=True).stdout.split()
for c in cs:
    a=show(c+"~1")["models"]; b=show(c)["models"]
    rem=sorted(set(a)-set(b)); add=sorted(set(b)-set(a)); ch=[n for n in set(a)&set(b) if a[n]!=b[n]]
    keys=collections.Counter()
    for n in ch:
        fa,fb=json.dumps(a[n],sort_keys=True),json.dumps(b[n],sort_keys=True)
        for k in set(a[n])|set(b[n]):
            if a[n].get(k)!=b[n].get(k): keys[k]+=1
    sz=lambda L: dict(collections.Counter(re.match(r'lm-([^-]+)',n).group(1) if n.startswith('lm-') else 'other' for n in L))
    print(c,'removed',len(rem),sz(rem),'added',len(add),sz(add),'changed',len(ch),dict(keys), sorted(ch)[:3])
