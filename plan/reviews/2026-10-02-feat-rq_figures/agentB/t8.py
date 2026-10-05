import numpy as np, warnings; warnings.simplefilter("ignore")
from analysis.rq04_surrogates import catalogue as C
fams=list("abcd"); pairs=[(a,b) for i,a in enumerate(fams) for b in fams[i+1:]]
# all four variants identical and noiseless: nothing is separated
Cm=np.full((4,10),0.5); W=np.full((4,5),0.5)
o=C._curve_stats("nonexistent_task",fams,Cm,W,pairs,np.random.default_rng(0))
print({k:o.get(k) for k in ["gap_over_noise","resolved_pairs_noise","retest_agreement","sign_consistency_window","sign_persistence","da_ckpt_mean","kendall_w_window","settling_time","crossings"]})
# float-noise window
W2=np.tile(np.array([[0.1+0.2],[0.3],[0.3],[0.1+0.2]]),(1,5)); W2[0,2]=0.3; Cm2=np.tile(W2[:,:1],(1,10)); Cm2[:,0]-=0.1
o=C._curve_stats("nonexistent_task",fams,Cm2,W2,pairs,np.random.default_rng(0)); print({k:o.get(k) for k in ["gain_over_noise","gap_over_noise","cronbach_alpha","icc_window","_pooled_sd"]})
import multiprocessing as mp, sys; print(sys.platform, mp.get_start_method(allow_none=True), mp.get_all_start_methods())
