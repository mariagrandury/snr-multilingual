import sys, time, numpy as np
import matplotlib as mpl, matplotlib.pyplot as plt
from multiprocessing import get_context
def work(seed):
    r=np.random.default_rng(seed); a=r.random((20000,25)); x=r.random(25)
    s=0.
    for _ in range(20): s+=float((a@x).sum())+float(np.corrcoef(r.random(3000),r.random(3000))[0,1])
    b=r.random((600,600)); s+=float((b@b).sum())
    return round(s,3)
if __name__=="__main__":
    mode=sys.argv[1]
    if mode=="blas_first":
        b=np.random.default_rng(0).random((1500,1500)); (b@b).sum(); np.linalg.svd(b[:300,:300])
    if mode=="fig_first":
        f,ax=plt.subplots(); plt.close(f)
    t=time.time()
    with get_context("fork").Pool(4) as p: out=p.map(work, range(8))
    print(mode, mpl.get_backend(), "ok", len(out), round(time.time()-t,1),"s")
