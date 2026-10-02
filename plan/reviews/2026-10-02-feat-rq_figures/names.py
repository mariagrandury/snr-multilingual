import re,os,sys,subprocess,fnmatch,collections
R='/Users/mariagrandury/Projects/epfl/snr-multilingual'
tracked=open(os.path.dirname(__file__)+'/tracked.txt').read().split('\n')
EXT=('png','csv','svg','pdf','json')
base=collections.defaultdict(list)
for t in tracked:
    base[os.path.basename(t)].append(t)
allb=list(base)
src=[t for t in tracked if t.endswith(('.py','.sh')) and (t.startswith(('src/signal-and-noise/','documents/','scripts/')))]
pat=re.compile(r'''["'`/ (=]([A-Za-z0-9_\-{}\[\].:*+!'"(), ]*?\.(?:png|csv|svg|pdf|json))\b''')
miss=collections.defaultdict(list)
for f in src:
    txt=subprocess.run(['git','-C',R,'show','HEAD:'+f],capture_output=True,text=True).stdout
    for i,line in enumerate(txt.split('\n'),1):
        for m in re.finditer(r'''([A-Za-z0-9_\-{}.]*(?:\{[^{}]*\}[A-Za-z0-9_\-.]*)*\.(?:png|csv|svg|json))(?![A-Za-z0-9_])''',line):
            name=m.group(1)
            if name.startswith('.') or len(name)<6: continue
            g=re.sub(r'\{[^{}]*\}','*',name)
            if g.strip('*.')in EXT or g in('*.png','*.csv','*.svg','*.json'): continue
            if g.startswith('*.' ) : continue
            if '*' in g:
                ok=any(fnmatch.fnmatchcase(b,g) for b in allb)
            else: ok=g in base
            if not ok: miss[f].append((i,name))
for f in sorted(miss):
    print('##',f)
    seen=set()
    for i,n in miss[f]:
        if n in seen: continue
        seen.add(n); print('   %d: %s'%(i,n))
