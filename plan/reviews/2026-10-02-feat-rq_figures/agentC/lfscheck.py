import subprocess,hashlib,os,sys
R,WT=sys.argv[1],sys.argv[2]
out=subprocess.run(['git','-C',R,'lfs','ls-files','-l','HEAD'],capture_output=True,text=True).stdout
n=bad=miss=0
for l in out.splitlines():
    oid,_,p=l.split(' ',2)
    if not p.startswith('src/signal-and-noise/analysis') : continue
    f=os.path.join(WT,p)
    if not os.path.exists(f): miss+=1; continue
    n+=1
    h=hashlib.sha256(open(f,'rb').read()).hexdigest()
    if h!=oid:
        bad+=1
        if bad<10: print('DIFF',p,open(f,'rb').read(40))
print(n,bad,miss)
