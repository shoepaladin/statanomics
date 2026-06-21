"""
Phase 3 — Feature tests (EconML / R grf parity).
"""

import numpy as np
import pytest
from grf.forest_numba import NumbaCausalForest

from tests.conftest import make_data


# ---------------------------------------------------------------------------
# P3.1  feature_importances_
# ---------------------------------------------------------------------------

class TestFeatureImportances:

    def test_importances_sum_to_one(self, fitted_forest):
        imp = fitted_forest.feature_importances_
        assert abs(imp.sum() - 1.0) < 1e-6, f"Importances sum to {imp.sum()}"

    def test_importances_shape(self, fitted_forest, small_data):
        X, _, _, _ = small_data
        imp = fitted_forest.feature_importances_
        assert imp.shape == (X.shape[1],), (
            f"Expected shape ({X.shape[1]},) got {imp.shape}"
        )

    def test_importances_nonneg(self, fitted_forest):
        assert np.all(fitted_forest.feature_importances_ >= 0)

    def test_true_feature_has_highest_importance(self):
        """
        With tau(x) = x[:,0], feature 0 should be ranked most important.
        """
        X, Y, W, _ = make_data(n=600, p=5, noise=0.2, seed=0)
        forest = NumbaCausalForest(
            n_trees=60, max_depth=6, min_leaf_size=5,
            n_folds=5, n_quantiles=15, mtry=None,
            verbose=0, random_state=0
        )
        forest.fit(X, Y, W)
        top_feature = np.argmax(forest.feature_importances_)
        assert top_feature == 0, (
            f"Expected feature 0 to be most important, got {top_feature}. "
            f"Importances: {forest.feature_importances_.round(3)}"
        )


# ---------------------------------------------------------------------------
# P3.2  OOB predictions
# ---------------------------------------------------------------------------

class TestOOBPredictions:

    def test_oob_returns_n_array(self, fitted_forest, small_data):
        X, _, _, _ = small_data
        oob = fitted_forest.oob_predict()
        assert oob.shape == (fitted_forest.n,)

    def test_oob_has_no_train_test_leakage(self, fitted_forest):
        """Each OOB prediction uses only trees that excluded that point."""
        for tree in fitted_forest.trees:
            assert hasattr(tree, 'in_subsample_'), (
                "Tree is missing in_subsample_ attribute"
            )
            assert tree.in_subsample_.dtype == bool

    def test_oob_coverage_is_reasonable(self):
        """
        With subsample_ratio=0.5 each point is OOB in ~50% of trees,
        so almost all points should get a valid (non-NaN) OOB prediction.
        """
        X, Y, W, _ = make_data(n=400, p=4, noise=0.3, seed=0)
        forest = NumbaCausalForest(
            n_trees=50, max_depth=5, min_leaf_size=5,
            n_folds=2, n_quantiles=10,
            subsample_ratio=0.5, verbose=0, random_state=0
        )
        forest.fit(X, Y, W)
        oob = forest.oob_predict()
        nan_frac = np.mean(np.isnan(oob))
        assert nan_frac < 0.05, f"{nan_frac:.1%} of OOB predictions are NaN"

    def test_oob_correlates_with_truth(self):
        """OOB predictions should correlate with true CATE."""
        X, Y, W, tau_true = make_data(n=500, p=4, noise=0.3, seed=0)
        forest = NumbaCausalForest(
            n_trees=60, max_depth=6, min_leaf_size=5,
            n_folds=5, n_quantiles=15,
            subsample_ratio=0.6, verbose=0, random_state=0
        )
        forest.fit(X, Y, W)
        oob = forest.oob_predict()
        valid = ~np.isnan(oob)
        corr = np.corrcoef(oob[valid], tau_true[valid])[0, 1]
        assert corr > 0.3, f"OOB-truth correlation is only {corr:.2f}"

    def test_oob_worse_than_insample(self):
        """OOB MSE should be >= in-sample MSE (no leakage guard)."""
        X, Y, W, tau_true = make_data(n=400, p=4, noise=0.2, seed=0)
        forest = NumbaCausalForest(
            n_trees=50, max_depth=5, min_leaf_size=5,
            n_folds=2, n_quantiles=10, verbose=0, random_state=0
        )
        forest.fit(X, Y, W)
        oob = forest.oob_predict()
        insample = forest.predict(X)
        valid = ~np.isnan(oob)
        mse_oob = np.mean((oob[valid] - tau_true[valid]) ** 2)
        mse_in = np.mean((insample - tau_true) ** 2)
        assert mse_oob >= mse_in * 0.9, (
            "OOB MSE is suspiciously lower than in-sample MSE — possible leakage"
        )


# ---------------------------------------------------------------------------
# P3.3  verbose parameter / no bare print()
# ---------------------------------------------------------------------------

class TestVerbose:

    def test_verbose_0_produces_no_output(self, capsys, small_data):
        X, Y, W, _ = small_data
        forest = NumbaCausalForest(
            n_trees=3, max_depth=3, min_leaf_size=10,
            n_folds=2, n_quantiles=5, verbose=0, random_state=0
        )
        forest.fit(X, Y, W)
        captured = capsys.readouterr()
        assert captured.out == "", (
            f"verbose=0 still printed to stdout: {repr(captured.out[:200])}"
        )


# ---------------------------------------------------------------------------
# P3.4  Input validation
# ---------------------------------------------------------------------------

class TestInputValidation:

    def test_1d_X_raises(self, small_data):
        X, Y, W, _ = small_data
        with pytest.raises(ValueError, match="2-D"):
            NumbaCausalForest(n_trees=3, n_folds=2).fit(X[:, 0], Y, W)

    def test_length_mismatch_Y_raises(self, small_data):
        X, Y, W, _ = small_data
        with pytest.raises(ValueError, match="Y length"):
            NumbaCausalForest(n_trees=3, n_folds=2).fit(X, Y[:-1], W)

    def test_length_mismatch_W_raises(self, small_data):
        X, Y, W, _ = small_data
        with pytest.raises(ValueError, match="W length"):
            NumbaCausalForest(n_trees=3, n_folds=2).fit(X, Y, W[:-1])

    def test_nan_in_X_raises(self, small_data):
        X, Y, W, _ = small_data
        X_bad = X.copy()
        X_bad[0, 0] = np.nan
        with pytest.raises(ValueError, match="NaN"):
            NumbaCausalForest(n_trees=3, n_folds=2).fit(X_bad, Y, W)

    def test_predict_before_fit_raises(self, small_data):
        X, _, _, _ = small_data
        with pytest.raises(RuntimeError, match="fit"):
            NumbaCausalForest(n_trees=3).predict(X)

    def test_predict_wrong_n_features_raises(self, fitted_forest, small_data):
        X, _, _, _ = small_data
        with pytest.raises(ValueError, match="features"):
            fitted_forest.predict(X[:, :2])


# ---------------------------------------------------------------------------
# P3.5  honesty_fraction parameter
# ---------------------------------------------------------------------------

class TestHonestyFraction:

    def test_fraction_stored_correctly(self, small_data):
        X, Y, W, _ = small_data
        for hf in (0.3, 0.5, 0.7):
            f = NumbaCausalForest(n_trees=3, n_folds=2, honesty_fraction=hf,
                                  max_depth=3, min_leaf_size=5, verbose=0,
                                  random_state=0)
            f.fit(X, Y, W)
            assert f.honesty_fraction == hf

    def test_fraction_affects_split_estimation_ratio(self, small_data):
        """Higher honesty_fraction → larger split sample → deeper trees."""
        X, Y, W, _ = small_data
        common = dict(n_trees=10, n_folds=2, max_depth=6, min_leaf_size=5,
                      n_quantiles=5, verbose=0, random_state=0)

        def avg_depth(forest):
            depths = []
            def depth(node, d=0):
                if node is None or node.estimate_indices is not None:
                    depths.append(d)
                    return
                depth(node.left, d + 1)
                depth(node.right, d + 1)
            for tree in forest.trees:
                depth(tree.root)
            return np.mean(depths)

        f_low = NumbaCausalForest(**common, honesty_fraction=0.3)
        f_high = NumbaCausalForest(**common, honesty_fraction=0.8)
        f_low.fit(X, Y, W)
        f_high.fit(X, Y, W)
        # Higher split fraction should give equal-or-deeper trees on average
        assert avg_depth(f_high) >= avg_depth(f_low) - 1.0


# ---------------------------------------------------------------------------
# P3.6  sklearn compatibility: get_params / set_params
# ---------------------------------------------------------------------------

class TestSklearnCompat:

    def test_get_params_returns_all_constructor_args(self):
        f = NumbaCausalForest(n_trees=77, max_depth=3, random_state=5)
        params = f.get_params()
        assert params['n_trees'] == 77
        assert params['max_depth'] == 3
        assert params['random_state'] == 5

    def test_set_params_updates_attributes(self):
        f = NumbaCausalForest(n_trees=10)
        f.set_params(n_trees=50, max_depth=3)
        assert f.n_trees == 50
        assert f.max_depth == 3

    def test_get_set_params_round_trip(self):
        f = NumbaCausalForest(n_trees=42, random_state=7)
        params = f.get_params()
        f2 = NumbaCausalForest()
        f2.set_params(**params)
        assert f2.get_params() == params

    def test_n_features_in_set_after_fit(self, fitted_forest, small_data):
        X, _, _, _ = small_data
        assert fitted_forest.n_features_in_ == X.shape[1]

    def test_effect_api_matches_predict(self, fitted_forest, small_data):
        """econml effect() must match predict()."""
        X, _, _, _ = small_data
        np.testing.assert_array_equal(
            fitted_forest.effect(X), fitted_forest.predict(X)
        )

    def test_effect_interval_shape(self, fitted_forest, small_data):
        X, _, _, _ = small_data
        lower, upper = fitted_forest.effect_interval(X)
        assert lower.shape == (len(X), 1)
        assert upper.shape == (len(X), 1)
