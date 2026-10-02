import re,os,sys,glob,collections
WT=sys.argv[1]
tracked=set(l.strip() for l in open('lsfiles.txt'))
dirs=set()
for t in tracked:
    p=t
    while '/' in p:
        p=p.rsplit('/',1)[0]; dirs.add(p)
base=collections.defaultdict(list)
for t in tracked: base[os.path.basename(t)].append(t)
docs=glob.glob(WT+'/src/signal-and-noise/analysis/**/README.md',recursive=True)+[WT+'/src/signal-and-noise/README.md',WT+'/src/signal-and-noise/CLAUDE.md',WT+'/src/signal-and-noise/analysis/RULES.md',WT+'/CLAUDE.md']+glob.glob(WT+'/plan/*.md')
EXT=r'(?:png|csv|svg|pdf|json|py|sh|md)'
dead=[];bare=[]
for d in sorted(docs):
    rel=os.path.relpath(d,WT); folder=os.path.dirname(rel)
    parts=rel.split('/')
    rq=None
    if 'analysis' in parts and len(parts)>parts.index('analysis')+2: rq='/'.join(parts[:parts.index('analysis')+2])
    for i,line in enumerate(open(d),1):
        for m in re.finditer(r'\]\(([^)\s]+)\)',line):
            t=m.group(1)
            if t.startswith('#') or t.startswith('mailto'): continue
            t0=t.split('#')[0]
            if not t0: continue
            if t0.startswith('http'):
                mm=re.match(r'https://github.com/mariagrandury/snr-multilingual/(?:blob|tree|raw)/main/(.*)',t0)
                if not mm: continue
                p=mm.group(1).split('?')[0]
            else:
                p=os.path.normpath(os.path.join(folder,t0))
            p=p.rstrip('/')
            if p not in tracked and p not in dirs:
                dead.append((rel,i,t,p))
        for m in re.finditer(r'`([^`\s]+\.'+EXT+r')`',line):
            t=m.group(1)
            if any(c in t for c in '*<>{}$[]|'): continue
            b=os.path.basename(t)
            cands=base.get(b,[])
            ok=False
            if '/' in t:
                # try resolve relative to folder, repo root, src/signal-and-noise, or suffix match
                for root in (folder,'','src/signal-and-noise','src/signal-and-noise/analysis','src'):
                    if os.path.normpath(os.path.join(root,t)) in tracked: ok=True
                if not ok and any(c.endswith('/'+t.lstrip('./')) for c in cands): ok=True
            else:
                if rq: ok=any(c.startswith(rq+'/') for c in cands)
                else: ok=bool(cands)
            if not ok:
                bare.append((rel,i,t,'elsewhere:'+cands[0] if cands else 'NOWHERE'))
print('DEAD LINKS',len(dead))
for x in dead: print('\t'.join(map(str,x)))
print('BARE',len(bare))
for x in bare: print('\t'.join(map(str,x)))
