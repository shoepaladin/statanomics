"""
Investigate why larger B (more trees) leads to higher FPR in the delta method.

Three experiments:
  1. Extended B sweep at n=800, n_reps=60, with SE decomposition
  2. n x B grid: SE/MC_SE ratio across sample sizes and ensemble sizes
  3. Omega norm analysis: how ||Omega||_2 scales with B

Hypothesis: As B increases, Omega weights become smoother (law of large numbers)
and the L2 norm of Omega decreases faster than the true MC_SE, causing
SE underestimation and inflated FPR.
"""

import sys
import numpy as np

sys.path.insert(0, '/home/user/statanomics/workingcode/grf_python')
from grf.forest_numba import NumbaCausalForest

rng_test = np.random.default_rng(42)
P = 5
X_test_fixed = rng_test.standard_normal((5, P))


def run_simulation(n, B, n_reps, X_test, label=""):
    tau_mat = np.zeros((n_reps, len(X_test)))
    se_mat  = np.zeros((n_reps, len(X_test)))
    omega_l2 = np.zeros(n_reps)  # mean ||Omega_col||_2 across test points

    for rep in range(n_reps):
        rng = np.random.default_rng(rep * 37 + 7)
        X = rng.standard_normal((n, P))
        W = rng.binomial(1, 0.5, n).astype(float)
        Y = rng.standard_normal(n)   # tau=0 null DGP

        cf = NumbaCausalForest(
            n_trees=B, max_depth=8, min_leaf_size=10,
            n_folds=4, n_quantiles=20, n_jobs=1,
            random_state=rep * 13
        )
        cf.fit(X, Y, W)

        # Retrieve Omega directly to compute L2 norm
        X_c = np.ascontiguousarray(X_test, dtype=np.float64)
        from grf.numba_core import (
            traverse_tree_batch, batch_predict_from_leaves,
            accumulate_omega_tree, compute_delta_variance
        )
        n_test = len(X_c)
        tree_preds = np.zeros((B, n_test))
        Omega = np.zeros((n, n_test), dtype=np.float64)

        for b, tree in enumerate(cf.trees):
            feats, threshs, left_ch, right_ch, starts, sizes, flat_idx = tree.to_arrays()
            max_leaf = int(sizes.max()) if sizes.max() > 0 else 1
            leaf_indices, leaf_sizes = traverse_tree_batch(
                X_c, feats, threshs, left_ch, right_ch,
                starts, sizes, flat_idx, max_leaf
            )
            tree_preds[b] = batch_predict_from_leaves(cf.Y_resid, cf.W_resid, leaf_indices, leaf_sizes)
            accumulate_omega_tree(Omega, leaf_indices, leaf_sizes, cf.W_resid, B)

        tau = np.mean(tree_preds, axis=0)
        variances = compute_delta_variance(Omega, cf.Y_resid)
        se = np.sqrt(np.maximum(variances, 0.0))

        tau_mat[rep] = tau
        se_mat[rep]  = se
        # L2 norm of each Omega column (one per test point), then average
        omega_l2[rep] = np.mean(np.sqrt(np.sum(Omega**2, axis=0)))

    mc_se   = tau_mat.std(axis=0)
    mean_se = se_mat.mean(axis=0)
    fpr     = np.mean(np.abs(tau_mat) > 1.96 * se_mat)
    ratio   = mean_se.mean() / mc_se.mean()

    result = {
        'n': n, 'B': B, 'n_reps': n_reps,
        'FPR': fpr,
        'SE/MC_SE': ratio,
        'mean_SE': mean_se.mean(),
        'MC_SE': mc_se.mean(),
        'mean_omega_l2': omega_l2.mean(),
        'std_omega_l2': omega_l2.std(),
    }
    return result


# -----------------------------------------------------------------------
# Experiment 1: Extended B sweep at fixed n=800
# -----------------------------------------------------------------------
print("\n" + "="*65)
print("Experiment 1: Extended B sweep  (n=800, n_reps=60)")
print("="*65)
print(f"{'B':>6}  {'FPR':>7}  {'SE/MC_SE':>9}  {'MC_SE':>7}  {'mean_SE':>8}  {'Omega_L2':>9}")
print("-"*65)
sys.stdout.flush()

B_values = [50, 100, 200, 500, 1000, 2000]
exp1 = []
for B in B_values:
    r = run_simulation(n=800, B=B, n_reps=60, X_test=X_test_fixed)
    exp1.append(r)
    print(f"{B:>6}  {r['FPR']:>7.4f}  {r['SE/MC_SE']:>9.4f}  {r['MC_SE']:>7.4f}  {r['mean_SE']:>8.4f}  {r['mean_omega_l2']:>9.6f}")
    sys.stdout.flush()


# -----------------------------------------------------------------------
# Experiment 2: n x B grid
# -----------------------------------------------------------------------
print("\n" + "="*65)
print("Experiment 2: n x B grid  (n_reps=40 each cell)")
print("SE/MC_SE ratio table")
print("="*65)

n_values = [200, 400, 800, 1600]
B_grid   = [50, 200, 1000]

header = f"{'n':>6}" + "".join(f"  B={B:>5}" for B in B_grid)
print(header)
print("-"*65)
sys.stdout.flush()

grid_results = {}
for n in n_values:
    row = f"{n:>6}"
    for B in B_grid:
        r = run_simulation(n=n, B=B, n_reps=40, X_test=X_test_fixed)
        grid_results[(n, B)] = r
        row += f"  {r['SE/MC_SE']:>7.4f}"
        sys.stdout.flush()
    print(row)
    sys.stdout.flush()

print("\nFPR table:")
print(header)
print("-"*65)
for n in n_values:
    row = f"{n:>6}"
    for B in B_grid:
        r = grid_results[(n, B)]
        row += f"  {r['FPR']:>7.4f}"
    print(row)
    sys.stdout.flush()


# -----------------------------------------------------------------------
# Experiment 3: Omega L2 norm decomposition
# -----------------------------------------------------------------------
print("\n" + "="*65)
print("Experiment 3: Omega L2 norm vs B  (n=800, n_reps=30)")
print("  Tests whether ||Omega||_2 decays as 1/sqrt(B) (pure averaging)")
print("  or slower (mean dominates variance at large B)")
print("="*65)
print(f"{'B':>6}  {'Omega_L2':>9}  {'Omega_L2*sqrt(B)':>16}  {'SE^2/sigma2':>12}")
sys.stdout.flush()

for r in exp1[:4]:  # use first 4 B values from exp1 (already have n=800)
    B = r['B']
    sigma2_approx = r['MC_SE']**2 / r['mean_omega_l2']**2 if r['mean_omega_l2'] > 0 else np.nan
    print(f"{B:>6}  {r['mean_omega_l2']:>9.6f}  {r['mean_omega_l2']*np.sqrt(B):>16.6f}  {sigma2_approx:>12.4f}")
    sys.stdout.flush()

print("\nDone.")
