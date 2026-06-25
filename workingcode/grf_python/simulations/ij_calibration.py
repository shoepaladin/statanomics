"""
Re-examination of the IJ variance estimator.

Tests the hypothesis that the 0% FPR of the original IJ was caused by the
MISSING finite-B Monte-Carlo bias correction (not an algebraic bug, and not a
fundamental "B must be >> 9000" limitation).

For each (n, B) cell we fit ONE forest per replication and compute three SEs
on a fixed set of null test points:
  - IJ raw  : uncorrected infinitesimal jackknife  (n/s) sum_j cov_j^2
  - IJ-U    : bias-corrected IJ (Wager-Hastie-Efron 2014)
  - delta   : the current delta-method SE

Reports null false-positive rate (target ~5%) and SE / Monte-Carlo-SE ratio
(target ~1.0) for each.  Data are generated under the null tau == 0.

Output written incrementally to simulations/ij_calibration_results.txt.
"""

import sys
import os
import numpy as np

sys.path.insert(0, '/home/user/statanomics/workingcode/grf_python')
from grf.forest_numba import NumbaCausalForest
from grf.numba_core import (
    traverse_tree_batch, batch_predict_from_leaves,
    accumulate_omega_tree, compute_delta_variance, compute_ij_variance,
)

OUT_FILE = os.path.join(os.path.dirname(__file__), 'ij_calibration_results.txt')
P = 5
X_test_fixed = np.random.default_rng(42).standard_normal((5, P))


def log(msg):
    print(msg, flush=True)
    with open(OUT_FILE, 'a') as f:
        f.write(msg + '\n')


def run_cell(n, B, n_reps):
    n_test = len(X_test_fixed)
    tau_mat = np.zeros((n_reps, n_test))
    se_ij_raw = np.zeros((n_reps, n_test))
    se_ij_u = np.zeros((n_reps, n_test))
    se_delta = np.zeros((n_reps, n_test))

    X_c = np.ascontiguousarray(X_test_fixed, dtype=np.float64)

    for rep in range(n_reps):
        rng = np.random.default_rng(rep * 37 + 7)
        X = rng.standard_normal((n, P))
        W = rng.binomial(1, 0.5, n).astype(float)
        Y = rng.standard_normal(n)          # null: Y independent of W

        cf = NumbaCausalForest(
            n_trees=B, max_depth=8, min_leaf_size=10,
            n_quantiles=20, n_jobs=1,
            random_state=rep * 13,
        )
        cf.fit(X, Y, W)

        tree_preds = np.zeros((B, n_test))
        Omega = np.zeros((n, n_test), dtype=np.float64)
        for b, tree in enumerate(cf.trees):
            feats, threshs, lch, rch, starts, sizes, flat_idx = tree.to_arrays()
            max_leaf = int(sizes.max()) if sizes.max() > 0 else 1
            li, ls = traverse_tree_batch(X_c, feats, threshs, lch, rch,
                                         starts, sizes, flat_idx, max_leaf)
            tree_preds[b] = batch_predict_from_leaves(cf.Y_resid, cf.W_resid, li, ls)
            accumulate_omega_tree(Omega, li, ls, cf.W_resid, B)

        tau = np.mean(tree_preds, axis=0)
        flags = np.array([t.in_subsample_ for t in cf.trees], dtype=bool)
        s = cf._subsample_size

        v_raw = compute_ij_variance(tree_preds, flags, s, n, bias_correction=False)
        v_u = compute_ij_variance(tree_preds, flags, s, n, bias_correction=True)
        v_delta = compute_delta_variance(Omega, cf.Y_resid)

        tau_mat[rep] = tau
        se_ij_raw[rep] = np.sqrt(np.maximum(v_raw, 0.0))
        se_ij_u[rep] = np.sqrt(np.maximum(v_u, 0.0))
        se_delta[rep] = np.sqrt(np.maximum(v_delta, 0.0))

    mc_se = tau_mat.std(axis=0).mean()

    def summarize(se_mat):
        fpr = np.mean(np.abs(tau_mat) > 1.96 * se_mat)
        ratio = se_mat.mean() / mc_se if mc_se > 0 else np.nan
        return fpr, ratio

    fpr_raw, r_raw = summarize(se_ij_raw)
    fpr_u, r_u = summarize(se_ij_u)
    fpr_d, r_d = summarize(se_delta)

    return {
        'n': n, 'B': B, 'mc_se': mc_se,
        'fpr_raw': fpr_raw, 'ratio_raw': r_raw,
        'fpr_u': fpr_u, 'ratio_u': r_u,
        'fpr_delta': fpr_d, 'ratio_delta': r_d,
    }


if __name__ == '__main__':
    with open(OUT_FILE, 'w') as f:
        f.write('')

    n_reps = int(sys.argv[1]) if len(sys.argv) > 1 else 40
    n_vals = [200, 400, 800, 1600]
    B_grid = [50, 200, 1000]

    log('=' * 78)
    log(f'IJ re-examination: raw IJ vs bias-corrected IJ-U vs delta '
        f'(n_reps={n_reps}, 5 null test pts/cell)')
    log('=' * 78)
    log(f"{'n':>5} {'B':>5} | {'FPR_raw':>8} {'rat_raw':>8} | "
        f"{'FPR_IJU':>8} {'rat_IJU':>8} | {'FPR_delt':>8} {'rat_delt':>8}")
    log('-' * 78)

    for n in n_vals:
        for B in B_grid:
            r = run_cell(n=n, B=B, n_reps=n_reps)
            log(f"{n:>5} {B:>5} | {r['fpr_raw']:>8.4f} {r['ratio_raw']:>8.3f} | "
                f"{r['fpr_u']:>8.4f} {r['ratio_u']:>8.3f} | "
                f"{r['fpr_delta']:>8.4f} {r['ratio_delta']:>8.3f}")
        log('-' * 78)

    log('\nTarget: FPR ~ 0.05, ratio ~ 1.0.  raw IJ inflated => low FPR.')
    log('Done.')
