import re,os,sys,glob,collections
D=sys.argv[1]; tracked=set(l.strip() for l in open('lsfiles.txt'))
secs=sorted(glob.glob(D+'/sections/*.tex'))
# bib keys
keys=collections.defaultdict(list)
for b in glob.glob(D+'/*.bib'):
    for i,l in enumerate(open(b),1):
        m=re.match(r'\s*@(\w+)\s*\{\s*([^,\s]+)\s*,',l)
        if m and m.group(1).lower() not in('comment','string','preamble'): keys[m.group(2)].append((os.path.basename(b),i))
print('bib keys',len(keys)); 
dups={k:v for k,v in keys.items() if len(v)>1}; print('DUP keys',len(dups)); [print(' ',k,v) for k,v in dups.items()]
# case-insensitive dups
low=collections.defaultdict(set)
for k in keys: low[k.lower()].add(k)
print('case-dups',[v for v in low.values() if len(v)>1])
gp=None
for s in secs:
    txt=open(s).read()
    rel=os.path.relpath(s,os.path.dirname(os.path.dirname(os.path.dirname(D))))
    for i,l in enumerate(txt.split('\n'),1):
        ls=l.split('%')[0] if not l.strip().startswith('%') else ''
        commented = l.strip().startswith('%')
        for m in re.finditer(r'\\cite[a-zA-Z]*\*?(?:\[[^\]]*\])*\{([^}]*)\}',l):
            for k in m.group(1).split(','):
                k=k.strip()
                if k and k not in keys: print('UNRESOLVED',os.path.basename(s),i,k,'(commented)' if commented else '')
        for m in re.finditer(r'\\includegraphics(?:\[[^\]]*\])?\{([^}]*)\}',l):
            print('FIG',os.path.basename(s),i,m.group(1),'(commented)' if commented else '')
        for m in re.finditer(r'\\(?:input|include)\{([^}]*)\}',l):
            print('INPUT',os.path.basename(s),i,m.group(1),'(commented)' if commented else '')
