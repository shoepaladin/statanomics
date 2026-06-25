import sys; sys.path.insert(0,'/home/user/statanomics/workingcode/grf_python')
import numpy as np
from grf.forest_numba import NumbaCausalForest
from grf.numba_core import compute_blb_variance, traverse_tree_batch, batch_predict_from_leaves
Xt=np.random.default_rng(42).standard_normal((5,5))

def run(n,B,reps,bypass_nuisance):
    taus=np.zeros((reps,5)); ses=np.zeros((reps,5))
    for r in range(reps):
        rng=np.random.default_rng(r*37+7)
        X=rng.standard_normal((n,5)); W=rng.binomial(1,.5,n).astype(float); Y=rng.standard_normal(n)
        cf=NumbaCausalForest(n_trees=B,max_depth=8,min_leaf_size=10,n_quantiles=20,n_jobs=1,subforest_size=4,variance='blb',random_state=r*13)
        if bypass_nuisance:
            cf._estimate_nuisance=lambda X,Y,W:(np.zeros(len(X)), np.full(len(X),0.5))
        cf.fit(X,Y,W)
        tau,se=cf.predict(Xt,return_std=True)
        taus[r]=tau; ses[r]=se
    mc=taus.std(0).mean()
    fpr=np.mean(np.abs(taus)>1.959964*ses)
    print(f"n={n} B={B} bypass_nuisance={bypass_nuisance} | BLB FPR={fpr:.3f} SE/MC={ses.mean()/mc:.3f} MC_SE={mc:.3f}",flush=True)

for byp in [False, True]:
    run(400,200,40,byp)
