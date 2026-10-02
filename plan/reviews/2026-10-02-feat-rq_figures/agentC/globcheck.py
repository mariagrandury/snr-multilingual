import re,os,glob,fnmatch,collections,sys
WT=sys.argv[1]
tracked=[l.strip() for l in open('lsfiles2.txt')]
A='src/signal-and-noise/analysis/'
docs=sorted(glob.glob(WT+'/'+A+'**/README.md',recursive=True))+[WT+'/src/signal-and-noise/'+x for x in('README.md','CLAUDE.md','analysis/RULES.md','analysis/paper_figures.md')]+[WT+'/CLAUDE.md']+sorted(glob.glob(WT+'/plan/*.md'))
EXTS=('png','csv','svg','pdf','json')
for d in docs:
    rel=os.path.relpath(d,WT); parts=rel.split('/')
    rq=None
    if rel.startswith(A) and len(parts)>4: rq='/'.join(parts[:4])
    scope=[t for t in tracked if (t.startswith(rq+'/') if rq else t.startswith(A))]
    bn=[os.path.basename(t) for t in scope]
    for i,line in enumerate(open(d),1):
        for m in re.finditer(r'`([^`\s]+)`',line):
            t=m.group(1)
            # compound ext  a.png/.pdf/.csv
            mm=re.match(r'^(.*?)\.((?:png|csv|svg|pdf|json)(?:/\.(?:png|csv|svg|pdf|json))+)$',t)
            if mm:
                stem=os.path.basename(mm.group(1)); 
                if '*' in stem or '<' in stem or '{' in stem: continue
                for e in mm.group(2).split('/.'):
                    if not any(b==stem+'.'+e for b in [os.path.basename(x) for x in tracked]): print('COMPOUND-MISSING',rel,i,stem+'.'+e)
                continue
            if '*' in t and not any(c in t for c in '<>{}$|[] '):
                b=os.path.basename(t)
                if b in('*','*.csv','*.png','*.py','*.md','*.json','*.sh','*.pdf'): continue
                # names only: treat as glob on basenames (with or without ext)
                hit=any(fnmatch.fnmatch(x,b) or fnmatch.fnmatch(os.path.splitext(x)[0],b) or fnmatch.fnmatch(x,b+'*') for x in bn)
                if not hit:
                    allhit=any(fnmatch.fnmatch(os.path.basename(x),b) or fnmatch.fnmatch(os.path.basename(x),b+'*') for x in tracked)
                    print('GLOB-NOMATCH',rel,i,t,'(matches elsewhere)' if allhit else '')
