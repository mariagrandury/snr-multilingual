import os,re,json,collections,subprocess,sys
SC=os.getcwd(); R='/Users/mariagrandury/Projects/epfl/snr-multilingual'
tracked=[l.strip() for l in open('lsfiles2.txt')]
tstems=collections.defaultdict(list)
for t in tracked: tstems[os.path.splitext(os.path.basename(t))[0]].append(t)
old=json.load(open('oldstems.json'))
for f in ('deleted_cf3.txt','deleted_68d.txt'):
    for l in open(f):
        s=os.path.splitext(os.path.basename(l.strip()))[0]
        old.setdefault(s,[])
# keep only old stems no longer tracked under a data ext anywhere in analysis predictivity pools
gone={s:v for s,v in old.items() if not any(os.path.splitext(t)[1] in('.csv','.png','.pdf','.svg','.json') for t in tstems.get(s,[]))}
still={s:[t for t in tstems[s]][:2] for s in old if s not in gone}
print('old stems',len(old),'gone',len(gone),'still-tracked-somewhere',len(still))
json.dump(still,open('old_still.json','w'),indent=0)
files=[t for t in tracked if (t.endswith(('.md','.tex','.py','.sh','.vue','.ts','.json')) and not t.startswith(('venv/','node_modules'))) ]
pat=re.compile(r'(?<![A-Za-z0-9_])('+'|'.join(sorted(map(re.escape,gone),key=len,reverse=True))+r')(?![A-Za-z0-9_])(\.(?:csv|png|pdf|svg))?')
hits=collections.defaultdict(list)
for f in files:
    try: txt=subprocess.run(['git','-C',R,'show','HEAD:'+f],capture_output=True,text=True).stdout
    except Exception: continue
    for i,l in enumerate(txt.split('\n'),1):
        for m in pat.finditer(l):
            s=m.group(1)
            # require ext, or a quoted/backticked bare stem
            pre=l[max(0,m.start()-1):m.start()]; 
            if m.group(2) or pre in '`"\'/':
                # skip script names: stem.py right after
                if l[m.end():m.end()+3]=='.py': continue
                hits[f].append((i,s+(m.group(2) or ''),gone[s]))
for f,h in sorted(hits.items()):
    print('##',f,len(h))
    for i,s,n in h[:400]: print('  ',i,s,'->',','.join(n) if n else '(deleted)')
