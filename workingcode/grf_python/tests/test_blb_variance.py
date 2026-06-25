"""
Tests for the bootstrap-of-little-bags (BLB) variance estimator and the
grouped subsampling it relies on.
"""

import numpy as np
import pytest

from grf.forest_numba import NumbaCausalForest
from grf.numba_core import compute_blb_variance


@pytest.fixture
def null_data():
    """n=300 null DGP: Y independent of W, so true tau == 0."""
    rng = np.random.default_rng(0)
    n, p = 300, 5
    X = rng.standard_normal((n, p))
    W = rng.binomial(1, 0.5, n).astype(float)
    Y = rng.standard_normal(n)
    return X, Y, W


# ---------------------------------------------------------------------------
# Grouped (little-bags) subsampling structure
# ---------------------------------------------------------------------------

class TestLittleBagsStructure:

    def test_n_trees_is_multiple_of_subforest_size(self, null_data):
        X, Y, W = null_data
        f = NumbaCausalForest(n_trees=23, subforest_size=4, max_depth=4,
                              min_leaf_size=5, n_jobs=1, random_state=1).fit(X, Y, W)
        # 23 // 4 = 5 groups -> 20 trees actually grown
        assert len(f.trees) == 20
        assert len(f.trees) % f._subforest_size == 0
        assert len(f.slices_) == 5

    def test_get_params_echoes_requested_n_trees(self):
        # The public param must be untouched by the grouping (sklearn clone).
        f = NumbaCausalForest(n_trees=23, subforest_size=4)
        assert f.get_params()['n_trees'] == 23

    def test_bag_mates_share_one_half_sample(self, null_data):
        X, Y, W = null_data
        f = NumbaCausalForest(n_trees=16, subforest_size=4, max_depth=4,
                              min_leaf_size=5, n_jobs=1, random_state=2).fit(X, Y, W)
        n = len(X)
        half = n // 2
        for sl in f.slices_:
            # Union of all bag-mates' subsample members must fit inside a
            # single half-sample of size n//2.
            members = np.zeros(n, dtype=bool)
            for t in sl:
                members |= f.trees[t].in_subsample_
            assert members.sum() <= half, (
                "bag-mates span more than one half-sample"
            )

    def test_subsample_within_half(self, null_data):
        X, Y, W = null_data
        f = NumbaCausalForest(n_trees=8, subforest_size=4, subsample_ratio=0.45,
                              max_depth=4, min_leaf_size=5, n_jobs=1,
                              random_state=3).fit(X, Y, W)
        assert f._subsample_size <= len(X) // 2


# ---------------------------------------------------------------------------
# compute_blb_variance
# ---------------------------------------------------------------------------

class TestComputeBLBVariance:

    def test_nonnegative(self):
        rng = np.random.default_rng(0)
        preds = rng.standard_normal((40, 6))
        v = compute_blb_variance(preds, group_size=4)
        assert np.all(v >= 0)
        assert v.shape == (6,)

    def test_zero_for_constant_predictions(self):
        preds = np.full((40, 3), 2.5)
        v = compute_blb_variance(preds, group_size=4)
        assert np.allclose(v, 0.0, atol=1e-10)

    def test_core_formula_is_scale_invariant_in_group_size(self):
        """
        Regression guard for the "denominator mismatch" bug class.

        With many groups (G large) the objective-Bayes cushion -> 0, so
        compute_blb_variance must recover the TRUE between-bag variance
        regardless of trees-per-bag L (i.e. regardless of total B = G*L).
        If the within-bag correction were divided by B instead of (L-1),
        large-L estimates would collapse toward 0; here they must not.
        """
        rng = np.random.default_rng(0)
        G, n_test = 400, 4000
        true_between = 0.5 ** 2  # bag-level deviation std = 0.5

        def gen(L):
            bag_dev = rng.normal(0, 0.5, size=(G, 1, n_test))
            tree_noise = rng.normal(0, 2.0, size=(G, L, n_test))  # large within-bag noise
            return (bag_dev + tree_noise).reshape(G * L, n_test)

        means = {L: compute_blb_variance(gen(L), L).mean() for L in (2, 4, 50)}
        for L, m in means.items():
            assert abs(m - true_between) < 0.06, (
                f"L={L}: V_BLB={m:.4f} far from true {true_between:.4f} "
                "(scale-dependent core formula => denominator bug)"
            )
        # B grew 25x from L=2 to L=50; estimate must stay flat (no collapse).
        assert means[50] / means[2] > 0.6

    def test_matches_between_minus_within_formula(self):
        rng = np.random.default_rng(7)
        L, G, n_test = 5, 12, 4
        preds = rng.standard_normal((L * G, n_test))
        tp = preds.reshape(G, L, n_test)
        bag_means = tp.mean(axis=1)
        overall = bag_means.mean(axis=0)
        v_between = np.mean((bag_means - overall) ** 2, axis=0)
        v_within = np.mean(np.mean((tp - bag_means[:, None, :]) ** 2, axis=1), axis=0)
        naive = v_between - v_within / (L - 1)
        got = compute_blb_variance(preds, group_size=L)
        # got applies a positive-part (objective-Bayes) correction, so it must
        # be >= the naive estimate but never below 0.
        assert np.all(got >= np.maximum(naive, 0.0) - 1e-9)


# ---------------------------------------------------------------------------
# Forest-level variance method selection
# ---------------------------------------------------------------------------

class TestVarianceMethods:

    @pytest.mark.parametrize("method", ["blb", "delta", "ij"])
    def test_method_returns_valid_se(self, null_data, method):
        X, Y, W = null_data
        f = NumbaCausalForest(n_trees=40, subforest_size=4, variance=method,
                              max_depth=5, min_leaf_size=5, n_jobs=1,
                              random_state=4).fit(X, Y, W)
        tau, std = f.predict(X[:10], return_std=True)
        assert std.shape == (10,)
        assert np.all(np.isfinite(std))
        assert np.all(std >= 0)

    def test_unknown_method_raises(self, null_data):
        X, Y, W = null_data
        f = NumbaCausalForest(n_trees=8, subforest_size=4, variance='nope',
                              n_jobs=1, random_state=5)
        # validated early, at fit time
        with pytest.raises(ValueError):
            f.fit(X, Y, W)

    def test_blb_null_calibration_smoke(self):
        """
        Small cross-dataset null calibration smoke test (the only valid way to
        measure coverage — a single forest's points are correlated through one
        dataset).  Guards the two regressions this work fixed:

          * raw-IJ SE inflation (was 2.5-11x too large -> 0% FPR): here the
            mean BLB SE must be within a sane factor of the Monte-Carlo SE;
          * zero-width CIs: SEs must be strictly positive.

        Not a tight 5% check — too few reps for that; the full grid lives in
        simulations/blb_calibration.py.
        """
        reps, n, p = 12, 300, 5
        X_test = np.random.default_rng(42).standard_normal((3, p))
        taus = np.zeros((reps, 3))
        ses = np.zeros((reps, 3))
        for r in range(reps):
            rng = np.random.default_rng(r * 37 + 7)
            X = rng.standard_normal((n, p))
            W = rng.binomial(1, 0.5, n).astype(float)
            Y = rng.standard_normal(n)
            f = NumbaCausalForest(n_trees=100, subforest_size=4, variance='blb',
                                  max_depth=6, min_leaf_size=10, n_jobs=1, random_state=r * 13).fit(X, Y, W)
            tau, std = f.predict(X_test, return_std=True)
            taus[r] = tau
            ses[r] = std

        assert np.all(ses > 0), "BLB produced a zero-width CI (broken variance)"
        mc_se = taus.std(axis=0).mean()
        ratio = ses.mean() / mc_se
        fpr = np.mean(np.abs(taus) > 1.959964 * ses)
        # raw IJ gave ratio 2.5-11 here; BLB should be order-1.
        assert 0.5 <= ratio <= 2.0, f"BLB SE/MC-SE ratio off: {ratio:.2f}"
        assert fpr <= 0.25, f"BLB null FPR too high: {fpr:.2f}"
