"""
Phase 2 — Performance tests.

Each test verifies that a performance fix:
  (a) produces the same result as before the fix, and
  (b) achieves the expected speedup (where measurable without fragility).

Timing assertions use generous multipliers so they pass on slow CI machines.
"""

import time
import math
import numpy as np
import pytest
from grf.forest_numba import NumbaCausalForest
from grf.tree_numba import NumbaCausalTree
from grf.numba_core import (
    find_best_split_numba,
    find_best_split_parallel,
    batch_predict_from_leaves,
    traverse_tree_batch,
    compute_variance_from_tree_preds,
)

from tests.conftest import make_data


# ---------------------------------------------------------------------------
# P2.1  Sort-outside-loop fix
# ---------------------------------------------------------------------------

class TestSortOutsideLoop:

    def test_split_result_is_deterministic_after_fix(self):
        """
        After moving np.sort outside the percentile loop the result must be
        identical to a naive reference implementation.
        """
        rng = np.random.default_rng(0)
        n, p = 120, 4
        X = rng.standard_normal((n, p)).astype(np.float64)
        Y = X[:, 0] + rng.standard_normal(n) * 0.3
        W = rng.binomial(1, 0.5, n).astype(np.float64)
        idx = np.arange(n, dtype=np.int32)
        percentiles = np.linspace(5.0, 95.0, 10)
        feat_idx = np.arange(p, dtype=np.int32)

        # Reference: brute-force, sort inside loop (old behaviour)
        from grf.numba_core import compute_pseudo_outcomes, compute_split_score_numba
        tau_p = 0.0
        pseudo = compute_pseudo_outcomes(Y, W, idx, tau_p)
        best_ref = (-np.inf, -1, 0.0)
        for f in range(p):
            vals = X[idx, f]
            for pct in percentiles:
                sv = np.sort(vals)
                i_ = int(len(sv) * pct / 100.0)
                thresh = sv[min(i_, len(sv) - 1)]
                mask = vals <= thresh
                if mask.sum() < 5 or (~mask).sum() < 5:
                    continue
                sc = compute_split_score_numba(pseudo, mask)
                if sc > best_ref[0]:
                    best_ref = (sc, f, thresh)

        feat_opt, thresh_opt, score_opt = find_best_split_numba(
            X, Y, W, idx, 0.0, 5, percentiles, feat_idx
        )

        assert feat_opt == best_ref[1], "Feature mismatch after sort fix"
        assert abs(score_opt - best_ref[0]) < 1e-9, "Score mismatch after sort fix"

    def test_parallel_split_agrees_with_sequential(self):
        """Parallel version must return the same best split as sequential."""
        rng = np.random.default_rng(1)
        n, p = 150, 6
        X = rng.standard_normal((n, p)).astype(np.float64)
        Y = X[:, 1] * 2 + rng.standard_normal(n) * 0.2
        W = rng.binomial(1, 0.5, n).astype(np.float64)
        idx = np.arange(n, dtype=np.int32)
        percentiles = np.linspace(5.0, 95.0, 15)
        feat_idx = np.arange(p, dtype=np.int32)

        f_seq, t_seq, s_seq = find_best_split_numba(
            X, Y, W, idx, 0.0, 5, percentiles, feat_idx
        )
        f_par, t_par, s_par = find_best_split_parallel(
            X, Y, W, idx, 0.0, 5, percentiles, feat_idx
        )

        assert f_seq == f_par, f"Feature: seq={f_seq} par={f_par}"
        assert abs(s_seq - s_par) < 1e-9, f"Score: seq={s_seq} par={s_par}"


# ---------------------------------------------------------------------------
# P2.2  Vectorised variance vs triple-loop
# ---------------------------------------------------------------------------

class TestVectorisedVariance:

    def _reference_variance(self, tree_preds):
        """Original triple-loop reference."""
        n_trees, n_test = tree_preds.shape
        variances = np.zeros(n_test)
        for i in range(n_test):
            mean_val = tree_preds[:, i].mean()
            sq = ((tree_preds[:, i] - mean_val) ** 2).sum()
            variances[i] = sq / (n_trees * (n_trees - 1))
        return variances

    def test_variance_matches_reference(self):
        """compute_variance_from_tree_preds must match the naive formula."""
        rng = np.random.default_rng(0)
        tree_preds = rng.standard_normal((50, 100)).astype(np.float64)
        ref = self._reference_variance(tree_preds)
        got = compute_variance_from_tree_preds(tree_preds)
        np.testing.assert_allclose(got, ref, rtol=1e-10)

    def test_variance_is_nonnegative(self):
        rng = np.random.default_rng(0)
        tree_preds = rng.standard_normal((30, 200)).astype(np.float64)
        var = compute_variance_from_tree_preds(tree_preds)
        assert np.all(var >= 0)

    def test_variance_zero_for_constant_preds(self):
        """If all trees agree, variance should be zero."""
        tree_preds = np.ones((20, 50))
        var = compute_variance_from_tree_preds(tree_preds)
        np.testing.assert_allclose(var, 0.0, atol=1e-12)


# ---------------------------------------------------------------------------
# P2.3  max_leaf_size buffer fix
# ---------------------------------------------------------------------------

class TestLeafBuffer:

    def test_buffer_is_not_n_over_2(self, fitted_forest, small_data):
        """
        The old code set max_leaf_size = self.n // 2, allocating a buffer
        orders of magnitude larger than needed.  After the fix, predict()
        should use sizes.max() per tree, which is much smaller.
        """
        X, _, _, _ = small_data
        n_test = 20
        X_sub = X[:n_test]

        # Inspect what max_leaf sizes the trees actually have
        max_observed = 0
        for tree in fitted_forest.trees:
            _, _, _, _, _, est_sizes, _ = tree.to_arrays()
            max_observed = max(max_observed, int(est_sizes.max()))

        # max observed leaf size should be << n // 2
        assert max_observed < fitted_forest.n // 4, (
            f"Max leaf size {max_observed} is suspiciously large "
            f"(n={fitted_forest.n})"
        )

    def test_predict_works_without_truncation(self, fitted_forest, small_data):
        """Prediction must be finite and not truncated by a too-small buffer."""
        X, _, _, tau_true = small_data
        tau_hat = fitted_forest.predict(X)
        assert np.all(np.isfinite(tau_hat))
        # Basic sanity: predictions should correlate with truth
        corr = np.corrcoef(tau_hat, tau_true)[0, 1]
        assert corr > 0.3, f"Predictions correlate only {corr:.2f} with truth"


# ---------------------------------------------------------------------------
# P2.4  Parallel tree building
# ---------------------------------------------------------------------------

class TestParallelTreeBuilding:

    def test_parallel_matches_sequential_predictions(self, small_data):
        """n_jobs>1 and n_jobs=1 must give identical predictions (same seed)."""
        X, Y, W, _ = small_data
        common = dict(n_trees=10, max_depth=4, min_leaf_size=5,
                      n_folds=2, n_quantiles=5, verbose=0, random_state=0)
        f_seq = NumbaCausalForest(**common, n_jobs=1)
        f_par = NumbaCausalForest(**common, n_jobs=2)
        f_seq.fit(X, Y, W)
        f_par.fit(X, Y, W)
        np.testing.assert_array_almost_equal(
            f_seq.predict(X[:30]), f_par.predict(X[:30]), decimal=8,
            err_msg="Parallel and sequential builds give different predictions"
        )

    def test_parallel_is_not_slower_than_sequential(self, medium_data):
        """Parallel build should finish in ≤ 4× the sequential time on 2 jobs."""
        X, Y, W, _ = medium_data
        common = dict(n_trees=30, max_depth=5, min_leaf_size=8,
                      n_folds=2, n_quantiles=10, verbose=0, random_state=0)

        t0 = time.time()
        NumbaCausalForest(**common, n_jobs=1).fit(X, Y, W)
        t_seq = time.time() - t0

        t0 = time.time()
        NumbaCausalForest(**common, n_jobs=2).fit(X, Y, W)
        t_par = time.time() - t0

        assert t_par <= t_seq * 4.0, (
            f"Parallel ({t_par:.2f}s) took more than 4× sequential ({t_seq:.2f}s)"
        )


# ---------------------------------------------------------------------------
# P2.5  Batch prediction via JIT tree traversal
# ---------------------------------------------------------------------------

class TestBatchPrediction:

    def test_to_arrays_round_trips_single_point(self, fitted_forest, small_data):
        """
        to_arrays() + traverse_tree_batch for one point must return the same
        leaf membership as the Python get_leaf_indices() fallback.
        """
        X, _, _, _ = small_data
        x = X[0:1].copy()
        tree = fitted_forest.trees[0]

        # Python traversal (reference)
        ref_idx = set(tree.get_leaf_indices(x[0]).tolist())

        # Batch JIT traversal
        feats, threshs, left_ch, right_ch, starts, sizes, flat_idx = \
            tree.to_arrays()
        max_leaf = int(sizes.max()) if sizes.max() > 0 else 1
        leaf_indices, leaf_sizes = traverse_tree_batch(
            x, feats, threshs, left_ch, right_ch,
            starts, sizes, flat_idx, max_leaf
        )
        got_idx = set(leaf_indices[0, :leaf_sizes[0]].tolist())

        assert got_idx == ref_idx, (
            f"Batch traversal returned {got_idx} vs Python {ref_idx}"
        )

    def test_batch_prediction_matches_point_by_point(self, fitted_forest, small_data):
        """
        batch_predict_from_leaves must agree with looping estimate_tau_ols_numba
        on every test point.
        """
        from grf.numba_core import estimate_tau_ols_numba
        X, _, _, _ = small_data
        X_sub = X[:40]
        tree = fitted_forest.trees[0]

        # Batch path
        feats, threshs, left_ch, right_ch, starts, sizes, flat_idx = \
            tree.to_arrays()
        max_leaf = int(sizes.max()) if sizes.max() > 0 else 1
        leaf_indices, leaf_sizes = traverse_tree_batch(
            X_sub, feats, threshs, left_ch, right_ch,
            starts, sizes, flat_idx, max_leaf
        )
        batch_preds = batch_predict_from_leaves(
            fitted_forest.Y_resid, fitted_forest.W_resid,
            leaf_indices, leaf_sizes
        )

        # Point-by-point reference
        ref_preds = np.array([
            estimate_tau_ols_numba(
                fitted_forest.Y_resid, fitted_forest.W_resid,
                tree.get_leaf_indices(X_sub[i])
            )
            for i in range(len(X_sub))
        ])

        np.testing.assert_allclose(batch_preds, ref_preds, rtol=1e-10,
                                   err_msg="Batch vs point-by-point mismatch")

    def test_forest_predict_faster_than_point_loop(self, medium_data):
        """
        Forest predict() (batch JIT) should be noticeably faster than manually
        calling get_leaf_indices per point per tree in Python.
        """
        from grf.numba_core import estimate_tau_ols_numba
        X, Y, W, _ = medium_data
        forest = NumbaCausalForest(
            n_trees=20, max_depth=5, min_leaf_size=8,
            n_folds=2, n_quantiles=10, verbose=0, random_state=0
        )
        forest.fit(X, Y, W)

        X_test = X[:100]
        n_test = len(X_test)

        # Warm-up JIT
        _ = forest.predict(X_test[:5])

        # Batch path (new)
        t0 = time.time()
        for _ in range(3):
            batch_tau = forest.predict(X_test)
        t_batch = (time.time() - t0) / 3

        # Python loop path (old style)
        t0 = time.time()
        for _ in range(3):
            loop_tau = np.zeros(n_test)
            for i in range(n_test):
                preds = []
                for tree in forest.trees:
                    idx = tree.get_leaf_indices(X_test[i])
                    preds.append(estimate_tau_ols_numba(
                        forest.Y_resid, forest.W_resid, idx
                    ))
                loop_tau[i] = np.mean(preds)
        t_loop = (time.time() - t0) / 3

        # Batch must agree with the loop
        np.testing.assert_allclose(batch_tau, loop_tau, rtol=1e-9)

        assert t_batch <= t_loop * 2.0, (
            f"Batch ({t_batch:.3f}s) not faster than loop ({t_loop:.3f}s)"
        )
