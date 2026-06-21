"""
Investigate why larger B (more trees) leads to higher FPR in the delta method.

Three experiments:
  1. Extended B sweep at n=800, n_reps=60, with Omega L2 norm tracking
  2. n x B grid: SE/MC_SE and FPR across sample sizes and ensemble sizes
  3. Omega L2 norm scaling: does ||Omega||_2 ~ 1/sqrt(B) or converge to a nonzero limit?

Output is written incrementally to simulations/b_effect_results.txt.
"""

import sys
import os
import numpy as np

sys.path.insert(0, '/home/user/statanomics/workingcode/grf_python')
from grf.forest_numba import NumbaCausalForest
from grf.numba_core import (
    traverse_tree_batch, batch_predict_from_leaves,
    accumulate_omega_tree, compute_delta_variance
)

OUT_FILE = os.path.join(os.path.dirname(__file__), 'b_effect_results.txt')
P = 5
X_test_fixed = np.random.default_rng(42).standard_normal((5, P))


def log(msg):
    print(msg, flush=True)
    with open(OUT_FILE, 'a') as f:
        f.write(msg + '\n')


def run_cell(n, B, n_reps):
    tau_mat  = np.zeros((n_reps, len(X_test_fixed)))
    se_mat   = np.zeros((n_reps, len(X_test_fixed)))
    ol2_vals = np.zeros(n_reps)

    for rep in range(n_reps):
        rng = np.random.default_rng(rep * 37 + 7)
        X = rng.standard_normal((n, P))
        W = rng.binomial(1, 0.5, n).astype(float)
        Y = rng.standard_normal(n)

        cf = NumbaCausalForest(
            n_trees=B, max_depth=8, min_leaf_size=10,
            n_folds=4, n_quantiles=20, n_jobs=1,
            random_state=rep * 13
        )
        cf.fit(X, Y, W)

        X_c = np.ascontiguousarray(X_test_fixed, dtype=np.float64)
        n_test = len(X_c)
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
        var = compute_delta_variance(Omega, cf.Y_resid)
        se  = np.sqrt(np.maximum(var, 0.0))

        tau_mat[rep]  = tau
        se_mat[rep]   = se
        ol2_vals[rep] = np.mean(np.sqrt(np.sum(Omega**2, axis=0)))

    mc_se   = tau_mat.std(axis=0)
    mean_se = se_mat.mean(axis=0)
    fpr     = np.mean(np.abs(tau_mat) > 1.96 * se_mat)
    ratio   = mean_se.mean() / mc_se.mean()

    return {
        'n': n, 'B': B,
        'FPR': fpr,
        'SE/MC_SE': ratio,
        'mean_SE': mean_se.mean(),
        'MC_SE': mc_se.mean(),
        'omega_l2': ol2_vals.mean(),
        'omega_l2_std': ol2_vals.std(),
    }


# -----------------------------------------------------------------------
# Clear output file
# -----------------------------------------------------------------------
with open(OUT_FILE, 'w') as f:
    f.write('')

# -----------------------------------------------------------------------
# Experiment 1: Extended B sweep at n=800
# -----------------------------------------------------------------------
log('\n' + '='*68)
log('Experiment 1: B sweep  (n=800, n_reps=60)')
log('='*68)
log(f"{'B':>6}  {'FPR':>7}  {'SE/MC_SE':>9}  {'MC_SE':>7}  {'mean_SE':>8}  {'Omega_L2':>9}  {'L2*sqrt(B)':>11}")
log('-'*68)

B_vals = [50, 100, 200, 500, 1000]
exp1 = []
for B in B_vals:
    r = run_cell(n=800, B=B, n_reps=60)
    exp1.append(r)
    log(f"{B:>6}  {r['FPR']:>7.4f}  {r['SE/MC_SE']:>9.4f}  {r['MC_SE']:>7.4f}  "
        f"{r['mean_SE']:>8.4f}  {r['omega_l2']:>9.6f}  {r['omega_l2']*B**0.5:>11.6f}")

# -----------------------------------------------------------------------
# Experiment 2: n x B grid
# -----------------------------------------------------------------------
log('\n' + '='*68)
log('Experiment 2: n x B grid  (n_reps=40 each cell)')
log('='*68)

n_vals = [200, 400, 800, 1600]
B_grid = [50, 200, 1000]

header = f"{'n':>6}" + "".join(f"  B={B:>5}" for B in B_grid)
sep    = '-'*50

log('\nSE/MC_SE table:')
log(header); log(sep)
grid = {}
for n in n_vals:
    row = f"{n:>6}"
    for B in B_grid:
        r = run_cell(n=n, B=B, n_reps=40)
        grid[(n, B)] = r
        row += f"  {r['SE/MC_SE']:>7.4f}"
        log(f"  [n={n}, B={B}] FPR={r['FPR']:.4f}  SE/MC_SE={r['SE/MC_SE']:.4f}  "
            f"MC_SE={r['MC_SE']:.4f}  Omega_L2={r['omega_l2']:.6f}")
    log(row)

log('\nFPR table:')
log(header); log(sep)
for n in n_vals:
    row = f"{n:>6}" + "".join(f"  {grid[(n,B)]['FPR']:>7.4f}" for B in B_grid)
    log(row)

log('\nOmega_L2 * sqrt(B) table  (constant iff L2 ~ 1/sqrt(B)):')
log(header); log(sep)
for n in n_vals:
    row = f"{n:>6}" + "".join(f"  {grid[(n,B)]['omega_l2']*B**0.5:>7.4f}" for B in B_grid)
    log(row)

log('\nDone.')
