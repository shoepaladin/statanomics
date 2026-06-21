"""
Calibration of the bootstrap-of-little-bags (BLB) variance estimator.

Fits ONE grouped (little-bags) forest per replication and computes three SEs
on a fixed set of null test points:
  - blb   : bootstrap-of-little-bags (between-group minus within-group)
  - delta : delta-method OLS-weight variance
  - ij    : bias-corrected infinitesimal jackknife (IJ-U)

Reports null false-positive rate (target ~0.05) and SE/MC-SE ratio (target
~1.0).  Data are generated under the null tau == 0.

Usage:  python simulations/blb_calibration.py [n_reps]
Output: simulations/blb_calibration_results.txt
"""

import sys
import os
import numpy as np

sys.path.insert(0, '/home/user/statanomics/workingcode/grf_python')
from grf.forest_numba import NumbaCausalForest
from grf.numba_core import (
    traverse_tree_batch, batch_predict_from_leaves,
    accumulate_omega_tree, compute_delta_variance,
    compute_blb_variance, compute_ij_variance,
)

OUT_FILE = os.path.join(os.path.dirname(__file__), 'blb_calibration_results.txt')
P = 5
X_test_fixed = np.random.default_rng(42).standard_normal((5, P))


def log(msg):
    print(msg, flush=True)
    with open(OUT_FILE, 'a') as f:
        f.write(msg + '\n')


def run_cell(n, B, n_reps):
    n_test = len(X_test_fixed)
    X_c = np.ascontiguousarray(X_test_fixed, dtype=np.float64)
    tau_mat = np.zeros((n_reps, n_test))
    se_blb = np.zeros((n_reps, n_test))
    se_delta = np.zeros((n_reps, n_test))
    se_ij = np.zeros((n_reps, n_test))

    for rep in range(n_reps):
        rng = np.random.default_rng(rep * 37 + 7)
        X = rng.standard_normal((n, P))
        W = rng.binomial(1, 0.5, n).astype(float)
        Y = rng.standard_normal(n)               # null: tau == 0

        cf = NumbaCausalForest(
            n_trees=B, max_depth=8, min_leaf_size=10,
            n_folds=4, n_quantiles=20, n_jobs=1,
            subforest_size=4, variance='blb',
            random_state=rep * 13,
        )
        cf.fit(X, Y, W)
        B_eff = cf.n_trees                       # may be rounded to multiple of L

        tree_preds = np.zeros((B_eff, n_test))
        Omega = np.zeros((n, n_test), dtype=np.float64)
        for b, tree in enumerate(cf.trees):
            feats, threshs, lch, rch, starts, sizes, flat_idx = tree.to_arrays()
            max_leaf = int(sizes.max()) if sizes.max() > 0 else 1
            li, ls = traverse_tree_batch(X_c, feats, threshs, lch, rch,
                                         starts, sizes, flat_idx, max_leaf)
            tree_preds[b] = batch_predict_from_leaves(cf.Y_resid, cf.W_resid, li, ls)
            accumulate_omega_tree(Omega, li, ls, cf.W_resid, B_eff)

        flags = np.array([t.in_subsample_ for t in cf.trees], dtype=bool)

        tau_mat[rep] = np.mean(tree_preds, axis=0)
        se_blb[rep] = np.sqrt(compute_blb_variance(tree_preds, cf._subforest_size))
        se_delta[rep] = np.sqrt(np.maximum(compute_delta_variance(Omega, cf.Y_resid), 0))
        se_ij[rep] = np.sqrt(compute_ij_variance(
            tree_preds, flags, cf._subsample_size, n, bias_correction=True))

    mc_se = tau_mat.std(axis=0).mean()

    def summ(se_mat):
        fpr = np.mean(np.abs(tau_mat) > 1.96 * se_mat)
        ratio = se_mat.mean() / mc_se if mc_se > 0 else np.nan
        return fpr, ratio

    f_b, r_b = summ(se_blb)
    f_d, r_d = summ(se_delta)
    f_i, r_i = summ(se_ij)
    return dict(n=n, B=B, mc_se=mc_se,
                f_blb=f_b, r_blb=r_b, f_delta=f_d, r_delta=r_d, f_ij=f_i, r_ij=r_i)


if __name__ == '__main__':
    with open(OUT_FILE, 'w') as f:
        f.write('')

    n_reps = int(sys.argv[1]) if len(sys.argv) > 1 else 40
    n_vals = [200, 400, 800, 1600]
    B_grid = [50, 200, 1000]

    log('=' * 80)
    log(f'BLB calibration: little-bags forest, blb vs delta vs IJ-U '
        f'(n_reps={n_reps}, 5 null pts/cell)')
    log('=' * 80)
    log(f"{'n':>5} {'B':>5} | {'FPR_blb':>8} {'rat_blb':>8} | "
        f"{'FPR_delt':>8} {'rat_delt':>8} | {'FPR_IJU':>8} {'rat_IJU':>8}")
    log('-' * 80)

    for n in n_vals:
        for B in B_grid:
            r = run_cell(n=n, B=B, n_reps=n_reps)
            log(f"{n:>5} {B:>5} | {r['f_blb']:>8.4f} {r['r_blb']:>8.3f} | "
                f"{r['f_delta']:>8.4f} {r['r_delta']:>8.3f} | "
                f"{r['f_ij']:>8.4f} {r['r_ij']:>8.3f}")
        log('-' * 80)

    log('\nTarget: FPR ~ 0.05, ratio ~ 1.0.')
    log('Done.')
