import json,sys,re,collections
SC=sys.argv[1]
m=json.load(open(SC+'/cells_main.json')); h=json.load(open(SC+'/cells_head.json'))
mc,hc=m['cells'],h['cells']
print('cells main',len(mc),'head',len(hc))
rem=sorted(set(mc)-set(hc)); add=sorted(set(hc)-set(mc))
size=lambda n: re.match(r'lm-([^-]+)',n).group(1)
print('removed',len(rem),collections.Counter(size(n) for n in rem)); print(' ',[n for n in rem if size(n) not in('90M','175M')])
print('added',len(add),collections.Counter(size(n) for n in add)); print(' ',[n for n in add if size(n) not in('90M','175M')])
ch=collections.Counter()
for n in sorted(set(mc)&set(hc)):
    d=[k for k in set(mc[n])|set(hc[n]) if mc[n].get(k)!=hc[n].get(k)]
    if d: ch[tuple(sorted(d))]+=1; print('CHANGED',n,{k:(mc[n].get(k),hc[n].get(k)) for k in d})
print('common',len(set(mc)&set(hc)),'changed',sum(ch.values()))
for k in ('seeds','ladder','tok'): print(k,'same' if m[k]==h[k] else (m[k],h[k]))
for s in sorted(set(m['schemes'])|set(h['schemes'])):
    a,b=m['schemes'].get(s),h['schemes'].get(s)
    print('scheme',s,'ABSENT@main' if a is None else 'ABSENT@head' if b is None else 'identical' if a==b else {k:(a.get(k),b.get(k)) for k in set(a)|set(b) if a.get(k)!=b.get(k)})
# old vs new 90M/175M
for old,new in (('lm-90M-L8-deep-seed1904','lm-90M-L8-b84-deep-seed1904'),('lm-175M-L8-deep-seed1904','lm-175M-L8-b168-deep-seed1904'),('lm-90M-L8-shallow-seed1904','lm-90M-L8-b84-shallow-seed1904')):
    a,b=mc.get(old),hc.get(new)
    if a and b: print(old,'->',new,{k:(a[k] if k in a else None,b[k]) for k in b if a.get(k)!=b[k] and k not in('EXP_NAME',)}, 'same:',[k for k in b if a.get(k)==b[k] and k not in ('langs','subsets')])
