"""B-sweep null FPR under grf-faithful defaults (ci.group.size=2, sample.fraction=0.5)."""
import sys, os
sys.path.insert(0, '/home/user/statanomics/workingcode/grf_python')
import numpy as np
from grf.forest_numba import NumbaCausalForest
OUT = os.path.join(os.path.dirname(__file__), 'grf_defaults_bsweep.txt')
Xt = np.random.default_rng(42).standard_normal((5,5))
def cell(n,B,reps=30):
    taus=np.zeros((reps,5)); ses=np.zeros((reps,5))
    for r in range(reps):
        rng=np.random.default_rng(r*37+7)
        X=rng.standard_normal((n,5)); W=rng.binomial(1,.5,n).astype(float); Y=rng.standard_normal(n)
        cf=NumbaCausalForest(n_trees=B, max_depth=8, min_leaf_size=10, n_folds=4,
                             n_quantiles=20, n_jobs=1, random_state=r*13)  # grf defaults
        cf.fit(X,Y,W)
        tau,std=cf.predict(Xt, return_std=True)
        taus[r]=tau; ses[r]=std
    mc=taus.std(0).mean(); fpr=np.mean(np.abs(taus)>1.959964*ses)
    line=f"n={n:5d} B={B:5d} | FPR={fpr:.3f}  SE/MC={ses.mean()/mc:.3f}  groups=B/2={B//2}"
    print(line, flush=True)
    open(OUT,'a').write(line+"\n")
open(OUT,'w').write("grf-faithful defaults: subforest_size=2, subsample_ratio=0.5\n")
for n in (400,800):
    for B in (50,200,1000):
        cell(n,B)
print("Done.", flush=True); open(OUT,'a').write("Done.\n")
