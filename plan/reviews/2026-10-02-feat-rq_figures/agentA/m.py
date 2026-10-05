import json,sys,collections
SC=sys.argv[1]
m=json.load(open(SC+"/models_main.json")); a=json.load(open(SC+"/models_8ac.json")); h=json.load(open(SC+"/models_head.json"))
print("top keys", {k:(type(v).__name__, len(v)) for k,v in h.items()})
print("8ac==head:", a==h)
for k in h:
    if k!="models":
        print(k, "main:", json.dumps(m.get(k))[:400]); print(k, "head:", json.dumps(h.get(k))[:400])
mm,hm=m["models"],h["models"]
rem=sorted(set(mm)-set(hm)); add=sorted(set(hm)-set(mm))
print("removed",len(rem)); print("added",len(add))
import re
def fam(n): return re.sub(r"-seed\d+$","",n)
print("removed:",rem)
print("added:",add)
ch=collections.defaultdict(list)
def flat(d,p=""):
    out={}
    for k,v in d.items():
        if isinstance(v,dict): out.update(flat(v,p+k+"."))
        else: out[p+k]=v
    return out
for n in sorted(set(mm)&set(hm)):
    x,y=flat(mm[n]),flat(hm[n])
    keys=tuple(sorted(k for k in set(x)|set(y) if x.get(k)!=y.get(k)))
    if keys: ch[keys].append(n)
for k,v in ch.items():
    print("CHANGED keys",k,len(v)); 
    for n in v[:4]:
        x,y=flat(mm[n]),flat(hm[n]); print("   ",n,{kk:(str(x.get(kk))[:80],str(y.get(kk))[:80]) for kk in k})
    print("    sizes:",collections.Counter(re.match(r"lm-([^-]+)",n).group(1) if n.startswith("lm-") else "other" for n in v))
n="lm-175M-L1-deep-seed1904"
print(json.dumps(hm.get(n),indent=1)[:1500])
