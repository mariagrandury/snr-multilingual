import re,os,collections,glob
tracked=open(os.path.dirname(__file__)+'/tracked.txt').read().split('\n')
A='src/signal-and-noise/analysis/'
arts=[t for t in tracked if t.startswith(A) and '/pretraining/' in t and 'predictivity' in t and t.rsplit('.',1)[-1] in('png','csv','svg','json','md','tex')]
code=''
for t in tracked:
    if t.endswith(('.py','.sh')) and t.startswith(('src/signal-and-noise/','documents/','scripts/')) and os.path.exists(t):
        code+=open(t,errors='ignore').read()+'\n'
stems=collections.defaultdict(set)
for a in arts:
    rq=a[len(A):].split('/')[0]
    if 'per_benchmark_plots' in a or '/curves/' in a: continue
    s=os.path.basename(a).rsplit('.',1)[0]
    stems[(rq,s)].add(a.split('/pretraining/')[1].split('/')[0])
no=[]
for (rq,s),pools in sorted(stems.items()):
    if s in code: continue
    s2=re.sub(r'_(by_benchmark|by_language)(?=_|$)','',s)
    s3=re.sub(r'_paper$','',s)
    if s2 in code or s3 in code: continue
    no.append((rq,s,sorted(pools)))
print(len(stems),'stems;',len(no),'not literal')
for rq,s,p in no: print(rq,s,','.join(p))
