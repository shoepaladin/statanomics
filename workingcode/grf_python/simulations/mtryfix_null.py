"""
High-rep null-FPR confirmation at the worst cells, current grf-faithful config
(BLB + ObjectiveBayes, subforest_size=2, subsample_ratio=0.5, OOB nuisance).

Reports FPR with a clustered 95% CI (rejection rate computed per replication
over the test points, then averaged across reps -> accounts for within-rep
correlation). If the CI brackets 0.05, the cell is calibrated.
"""
import sys, os
sys.path.insert(0, '/home/user/statanomics/workingcode/grf_python')
import numpy as np
from grf.forest_numba import NumbaCausalForest

OUT = os.path.join(os.path.dirname(__file__), 'mtryfix_null.txt')
Xt = np.random.default_rng(42).standard_normal((5, 5))
Z = 1.959964

def cell(n, B, reps):
    rej = np.zeros(reps)            # per-rep rejection rate over the 5 points
    taus = np.zeros((reps, 5)); ses = np.zeros((reps, 5))
    for r in range(reps):
        rng = np.random.default_rng(r * 37 + 7)
        X = rng.standard_normal((n, 5)); W = rng.binomial(1, .5, n).astype(float); Y = rng.standard_normal(n)
        cf = NumbaCausalForest(n_trees=B, max_depth=8, min_leaf_size=10,
                               n_jobs=1, random_state=r * 13).fit(X, Y, W)
        tau, se = cf.predict(Xt, return_std=True)
        taus[r] = tau; ses[r] = se
        rej[r] = np.mean(np.abs(tau) > Z * se)
    fpr = rej.mean()
    half = Z * rej.std(ddof=1) / np.sqrt(reps)   # clustered 95% CI
    mc = taus.std(0).mean(); ratio = ses.mean() / mc
    cal = "OK " if (fpr - half) <= 0.05 <= (fpr + half) else ("HIGH" if fpr - half > 0.05 else "LOW ")
    line = (f"n={n:5d} B={B:5d} reps={reps} | FPR={fpr:.3f}  95%CI=[{fpr-half:.3f},{fpr+half:.3f}]  "
            f"SE/MC={ratio:.3f}  [{cal}]")
    print(line, flush=True); open(OUT, 'a').write(line + "\n")

if __name__ == '__main__':
    reps = int(sys.argv[1]) if len(sys.argv) > 1 else 250
    open(OUT, 'w').write(f"High-rep confirmation (grf-faithful config), reps={reps}\n"
                         "Calibrated if 95% CI brackets 0.05.\n")
    for n, B in [(400,200),(400,1000),(800,200),(800,1000)]:
        cell(n, B, reps)
    print("Done.", flush=True); open(OUT, 'a').write("Done.\n")
