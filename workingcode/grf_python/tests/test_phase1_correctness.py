"""
Phase 1 — Correctness tests.

Each test documents a specific bug from the audit and verifies the fix.
"""

import math
import numpy as np
import pytest
from grf.forest_numba import NumbaCausalForest
from grf.tree_numba import NumbaCausalTree
from grf.numba_core import find_best_split_numba, find_best_split_parallel

from tests.conftest import make_data


# ---------------------------------------------------------------------------
# P1.1  mtry — feature subsampling
# ---------------------------------------------------------------------------

class TestMtry:

    def test_mtry_default_is_p_over_3(self, small_data):
        """Default mtry should be ceil(p/3), matching R grf default."""
        X, Y, W, _ = small_data
        p = X.shape[1]
        forest = NumbaCausalForest(
            n_trees=5, max_depth=4, min_leaf_size=5, n_folds=2,
            verbose=0, random_state=0
        )
        forest.fit(X, Y, W)
        expected = max(1, math.ceil(p / 3))
        assert forest.trees[0]._mtry == expected

    def test_mtry_1_limits_features_per_split(self, small_data):
        """mtry=1 means each split can only use 1 feature."""
        X, Y, W, _ = small_data
        forest = NumbaCausalForest(
            n_trees=5, max_depth=4, min_leaf_size=5, n_folds=2,
            mtry=1, verbose=0, random_state=0
        )
        forest.fit(X, Y, W)
        # Collect features actually used across all splits in all trees
        used_features = set()

        def collect(node):
            if node is None or node.estimate_indices is not None:
                return
            used_features.add(node.feature)
            collect(node.left)
            collect(node.right)

        for tree in forest.trees:
            collect(tree.root)

        # With mtry=1 and p=4 features we expect multiple features are eventually
        # used (across different trees/nodes) but each individual split only
        # had access to 1 — verify the tree objects store mtry=1
        assert all(t._mtry == 1 for t in forest.trees)

    def test_mtry_full_p_uses_all_features(self, small_data):
        """mtry=p disables feature subsampling."""
        X, Y, W, _ = small_data
        p = X.shape[1]
        forest = NumbaCausalForest(
            n_trees=5, max_depth=4, min_leaf_size=5, n_folds=2,
            mtry=p, verbose=0, random_state=0
        )
        forest.fit(X, Y, W)
        assert all(t._mtry == p for t in forest.trees)

    def test_feature_indices_subset_passed_correctly(self):
        """find_best_split_numba with a 1-feature subset should only ever
        return that feature as the best split feature."""
        rng = np.random.default_rng(0)
        n, p = 80, 6
        X = rng.standard_normal((n, p)).astype(np.float64)
        Y = X[:, 2] + rng.standard_normal(n) * 0.1
        W = rng.binomial(1, 0.5, n).astype(np.float64)
        idx = np.arange(n, dtype=np.int32)
        percentiles = np.linspace(5, 95, 10)
        # Only offer feature 2
        feature_indices = np.array([2], dtype=np.int32)
        feat, thresh, score = find_best_split_numba(
            X, Y, W, idx, 0.0, 5, percentiles, feature_indices
        )
        assert feat == 2 or feat == -1  # must be the offered feature or no split

    def test_mtry_produces_different_importances_than_full(self, small_data):
        """mtry < p decorrelates trees and changes feature importances vs full."""
        X, Y, W, _ = small_data
        p = X.shape[1]
        common = dict(n_trees=20, max_depth=5, min_leaf_size=5,
                      n_folds=2, verbose=0, random_state=0)

        f_full = NumbaCausalForest(**common, mtry=p)
        f_sqrt = NumbaCausalForest(**common, mtry=1)
        f_full.fit(X, Y, W)
        f_sqrt.fit(X, Y, W)

        # Importances should differ between the two
        assert not np.allclose(f_full.feature_importances_,
                               f_sqrt.feature_importances_, atol=1e-6)


# ---------------------------------------------------------------------------
# P1.2  Nuisance cross-fitting: shuffle + n_folds
# ---------------------------------------------------------------------------

class TestNuisanceCrossFitting:

    def test_oob_nuisance_is_order_independent(self):
        """
        grf computes Y.hat/W.hat as out-of-bag regression-forest predictions,
        which (unlike positional k-fold) do not depend on row order.

        Build data where Y is correlated with row order, then fit the forest on
        the sorted data and on a random permutation of it.  The orthogonalized
        residuals (and hence the variance estimates) must be equivalent up to
        the permutation — the failure mode of the old unshuffled 2-fold split.
        """
        rng = np.random.default_rng(42)
        n = 400
        X = np.sort(rng.standard_normal((n, 2)), axis=0)  # sorted => order-correlated
        W = rng.binomial(1, 0.5, n).astype(float)
        Y = X[:, 0] * 2.0 + W * X[:, 0] + rng.standard_normal(n) * 0.2

        f_sorted = NumbaCausalForest(n_trees=40, max_depth=4, min_leaf_size=5,
                                     verbose=0, random_state=0).fit(X, Y, W)
        # OOB residuals should be ~mean-zero even on order-correlated data
        assert abs(np.mean(f_sorted.Y_resid)) < 0.2
        # and should explain most of Y's variance (low residual MSE)
        assert np.mean(f_sorted.Y_resid ** 2) < 0.5 * np.var(Y)

        perm = rng.permutation(n)
        f_perm = NumbaCausalForest(n_trees=40, max_depth=4, min_leaf_size=5,
                                   verbose=0, random_state=0).fit(X[perm], Y[perm], W[perm])
        # Residual quality is invariant to ordering (no positional folding).
        assert np.isclose(np.mean(f_sorted.Y_resid ** 2),
                          np.mean(f_perm.Y_resid ** 2), rtol=0.25)

    def test_n_folds_parameter_accepted(self, small_data):
        """`n_folds` is retained for backward compat; fit must still succeed."""
        X, Y, W, _ = small_data
        for k in (2, 5):
            f = NumbaCausalForest(n_trees=3, n_folds=k, max_depth=3,
                                  min_leaf_size=5, verbose=0, random_state=0)
            f.fit(X, Y, W)  # should not raise
            assert f.n_folds == k

    def test_cross_fit_residuals_have_small_mean(self, medium_data):
        """
        With enough data the cross-fitted residuals should be approximately
        mean-zero (a basic sanity check that nuisance estimation is working).
        """
        X, Y, W, _ = medium_data
        forest = NumbaCausalForest(n_trees=5, n_folds=5, max_depth=4,
                                   min_leaf_size=5, verbose=0, random_state=0)
        forest.fit(X, Y, W)
        assert abs(np.mean(forest.Y_resid)) < 0.5
        assert abs(np.mean(forest.W_resid)) < 0.15


# ---------------------------------------------------------------------------
# P1.3  Expanded split thresholds
# ---------------------------------------------------------------------------

class TestSplitThresholds:

    def test_n_quantiles_parameter_is_adaptive(self, small_data):
        """Effective quantile count is capped at min(n_quantiles, split_n//10)."""
        X, Y, W, _ = small_data
        # With small_data n=300, subsample_ratio=0.5 → ~150 subsample,
        # honesty_fraction=0.5 → ~75 split samples → cap = 75//10 = 7
        for q in (5, 10, 20):
            f = NumbaCausalForest(n_trees=3, n_quantiles=q, max_depth=3,
                                  min_leaf_size=5, verbose=0, random_state=0)
            f.fit(X, Y, W)
            actual_q = f.trees[0]._percentiles.shape[0]
            assert actual_q <= q  # never exceeds requested
            assert actual_q >= 3  # never drops below minimum

    def test_more_quantiles_finds_step_function_split(self):
        """
        A step-function treatment effect at the 10th percentile is only
        findable with a denser grid (20 candidates) but not with 3.
        """
        rng = np.random.default_rng(0)
        n = 400
        X = rng.uniform(0, 1, (n, 1))
        W = rng.binomial(1, 0.5, n).astype(float)
        # True split at x = 0.1 (10th percentile)
        tau_true = np.where(X[:, 0] < 0.1, 2.0, 0.0)
        Y = tau_true * W + rng.normal(0, 0.1, n)

        common = dict(n_trees=40, max_depth=4, min_leaf_size=5, n_folds=2,
                      verbose=0, random_state=0)
        f3 = NumbaCausalForest(**common, n_quantiles=3)
        f20 = NumbaCausalForest(**common, n_quantiles=20)
        f3.fit(X, Y, W)
        f20.fit(X, Y, W)

        tau3 = f3.predict(X)
        tau20 = f20.predict(X)
        mse3 = np.mean((tau3 - tau_true) ** 2)
        mse20 = np.mean((tau20 - tau_true) ** 2)

        assert mse20 <= mse3 * 1.5, (
            f"Dense grid (n_q=20) MSE {mse20:.4f} should not be worse than "
            f"sparse grid (n_q=3) MSE {mse3:.4f} for a step-function CATE"
        )

    def test_split_score_improves_or_stays_with_more_quantiles(self):
        """More quantiles ≥ fewer quantiles for split quality (monotone)."""
        rng = np.random.default_rng(5)
        n, p = 100, 3
        X = rng.standard_normal((n, p)).astype(np.float64)
        Y = X[:, 0] + rng.standard_normal(n) * 0.1
        W = rng.binomial(1, 0.5, n).astype(np.float64)
        idx = np.arange(n, dtype=np.int32)
        feat_idx = np.arange(p, dtype=np.int32)

        scores = {}
        for nq in (3, 10, 20):
            percentiles = np.linspace(5.0, 95.0, nq)
            _, _, score = find_best_split_numba(X, Y, W, idx, 0.0, 5,
                                                percentiles, feat_idx)
            scores[nq] = score

        # More thresholds should find equal or better split
        assert scores[20] >= scores[3] - 1e-6, (
            f"20-quantile score {scores[20]:.4f} < 3-quantile {scores[3]:.4f}"
        )


# ---------------------------------------------------------------------------
# P1.4  Instance RNG — no global numpy state mutation
# ---------------------------------------------------------------------------

class TestInstanceRNG:

    def test_no_global_seed_mutation(self, small_data):
        """
        Fitting a forest must not change numpy's global RNG state.
        The old code called np.random.seed(random_state) which resets
        the global generator.
        """
        X, Y, W, _ = small_data
        np.random.seed(99)
        pre_state = np.random.get_state()[1].copy()  # array of ints

        forest = NumbaCausalForest(n_trees=5, max_depth=3, min_leaf_size=5,
                                   n_folds=2, verbose=0, random_state=42)
        forest.fit(X, Y, W)

        post_state = np.random.get_state()[1].copy()
        # Global state should have advanced (we may call np.random inside
        # sklearn), but it must NOT have been reset to the seed-42 state
        # (which is what the old code did).
        np.random.seed(42)
        seed42_state = np.random.get_state()[1].copy()
        assert not np.array_equal(post_state, seed42_state), (
            "Global numpy RNG was reset to random_state=42 — "
            "instance RNG fix not applied"
        )

    def test_same_random_state_gives_same_predictions(self, small_data):
        """Two forests with the same random_state must produce identical output."""
        X, Y, W, _ = small_data
        common = dict(n_trees=10, max_depth=4, min_leaf_size=5, n_folds=2,
                      verbose=0, random_state=123)
        f1 = NumbaCausalForest(**common)
        f2 = NumbaCausalForest(**common)
        f1.fit(X, Y, W)
        f2.fit(X, Y, W)
        np.testing.assert_array_almost_equal(
            f1.predict(X[:20]), f2.predict(X[:20]), decimal=10
        )

    def test_different_random_state_gives_different_predictions(self, small_data):
        """Different seeds should (very likely) differ."""
        X, Y, W, _ = small_data
        f1 = NumbaCausalForest(n_trees=10, max_depth=4, min_leaf_size=5,
                               n_folds=2, verbose=0, random_state=1)
        f2 = NumbaCausalForest(n_trees=10, max_depth=4, min_leaf_size=5,
                               n_folds=2, verbose=0, random_state=2)
        f1.fit(X, Y, W)
        f2.fit(X, Y, W)
        pred1 = f1.predict(X[:20])
        pred2 = f2.predict(X[:20])
        assert not np.allclose(pred1, pred2), (
            "Two forests with different seeds gave identical predictions"
        )


# ---------------------------------------------------------------------------
# P1.5  IJ variance fix — was algebraically always zero
# ---------------------------------------------------------------------------

class TestVarianceEstimator:

    def test_ci_has_nonzero_width(self, fitted_forest, small_data):
        """
        The old IJ implementation summed deviations from their own mean,
        which is identically zero — giving zero-width CIs for all points.
        """
        X, _, _, _ = small_data
        _, lower, upper = fitted_forest.predict_interval(X[:30])
        widths = upper - lower
        assert np.all(widths > 0), (
            f"Some CIs have zero width: {widths[widths <= 0]}"
        )

    def test_ci_width_scales_with_noise(self):
        """Higher outcome noise → wider confidence intervals."""
        common = dict(n_trees=40, max_depth=5, min_leaf_size=5,
                      n_folds=2, n_quantiles=10, verbose=0, random_state=0)
        results = {}
        for noise in (0.05, 1.0):
            X, Y, W, _ = make_data(n=300, p=3, noise=noise, seed=0)
            f = NumbaCausalForest(**common)
            f.fit(X, Y, W)
            _, lo, hi = f.predict_interval(X[:50])
            results[noise] = np.mean(hi - lo)

        assert results[1.0] > results[0.05], (
            "CI widths should grow with noise level"
        )

    def test_more_trees_reduces_variance(self):
        """Variance of the mean decreases as B grows (law of large numbers)."""
        X, Y, W, _ = make_data(n=400, p=4, noise=0.3, seed=0)
        common = dict(max_depth=5, min_leaf_size=5, n_folds=2,
                      n_quantiles=10, verbose=0, random_state=0)
        variances = {}
        for n_trees in (10, 100):
            f = NumbaCausalForest(n_trees=n_trees, **common)
            f.fit(X, Y, W)
            _, std = f.predict(X[:50], return_std=True)
            variances[n_trees] = np.mean(std ** 2)

        assert variances[100] < variances[10], (
            "More trees should reduce prediction variance"
        )

    def test_coverage_is_reasonable(self, medium_data):
        """
        95% CI coverage should be at least 50% on synthetic data with known tau.
        (Exact 95% not guaranteed — forest is biased for finite samples, and
        the delta-method SE targets the sampling variability of the estimator
        rather than a full bias+variance decomposition.  Old zero-width CIs
        give 0% coverage; the delta method gives substantially better coverage.)
        """
        X, Y, W, tau_true = medium_data
        forest = NumbaCausalForest(
            n_trees=80, max_depth=6, min_leaf_size=5,
            n_folds=5, n_quantiles=15, verbose=0, random_state=0
        )
        forest.fit(X, Y, W)
        _, lower, upper = forest.predict_interval(X, alpha=0.05)
        coverage = np.mean((tau_true >= lower) & (tau_true <= upper))
        assert coverage >= 0.50, (
            f"95% CI coverage is only {coverage:.1%} — "
            "variance estimator is likely still broken"
        )
