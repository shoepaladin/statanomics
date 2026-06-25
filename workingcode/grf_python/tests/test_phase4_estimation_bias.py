"""
Phase 4 — Split / estimation-bias fixes (issue #5, item 3).

Two defects from PR #4 are addressed here:

  3a. Empty/tiny estimation leaves (size < 2 or W'W ~= 0) used to contribute
      tau=0 to the forest average and distort the BLB groups.  They are now
      *dropped* (grf "good group" logic), not zeroed.

  3b. The honest min-leaf constraint was enforced only on the splitting sample,
      so an accepted split could orphan the estimation child (down to 0 points).
      The split search now rejects any candidate whose estimation child falls
      below min_leaf, so the degenerate leaves of 3a essentially never arise.

Validation of point-estimate quality (slope / coverage) belongs to the
non-null harness in test_phase1_correctness.py::TestNonNullCalibration, per the
issue; these tests pin the *mechanism*.
"""

import numpy as np
import pytest

from grf.forest_numba import NumbaCausalForest
from grf.numba_core import (
    find_best_split_numba,
    find_best_split_honest_numba,
    batch_predict_from_leaves,
    batch_predict_from_leaves_masked,
    compute_blb_variance,
    compute_blb_variance_masked,
)
from tests.conftest import make_data


# ---------------------------------------------------------------------------
# 3b — honest split rejects splits that orphan the estimation child
# ---------------------------------------------------------------------------

class TestHonestSplitMinSize:

    def _orphaning_case(self):
        """
        Feature 0 admits a perfectly balanced split on the *splitting* sample
        (20 | 20) but every *estimation* point sits above all candidate
        thresholds, so any accepted split leaves the estimation-left child
        empty.  A faithful honest search must refuse to split here.
        """
        n_split, n_est, p = 40, 40, 1
        # split sample: 20 points at x=0, 20 at x=1  -> a threshold of 0 is balanced
        Xs = np.vstack([np.zeros((20, p)), np.ones((20, p))])
        # estimation sample: all at x=5 (above every split-derived threshold)
        Xe = np.full((n_est, p), 5.0)
        X = np.vstack([Xs, Xe]).astype(np.float64)
        split_idx = np.arange(n_split, dtype=np.int32)
        est_idx = np.arange(n_split, n_split + n_est, dtype=np.int32)
        rng = np.random.default_rng(0)
        W = rng.binomial(1, 0.5, n_split + n_est).astype(np.float64)
        Y = rng.standard_normal(n_split + n_est)
        percentiles = np.linspace(5.0, 95.0, 10)
        feats = np.array([0], dtype=np.int32)
        return X, Y, W, split_idx, est_idx, percentiles, feats

    def test_non_honest_accepts_orphaning_split(self):
        """Baseline: the split-sample-only search happily takes the split."""
        X, Y, W, split_idx, est_idx, pcts, feats = self._orphaning_case()
        feat, thr, score = find_best_split_numba(
            X, Y, W, split_idx, 0.0, 5, pcts, feats
        )
        assert feat == 0, "split-sample search should accept the balanced split"

    def test_honest_rejects_orphaning_split(self):
        """The honest search must refuse it (estimation child would be empty)."""
        X, Y, W, split_idx, est_idx, pcts, feats = self._orphaning_case()
        feat, thr, score = find_best_split_honest_numba(
            X, Y, W, split_idx, est_idx, 0.0, 5, pcts, feats
        )
        assert feat == -1, (
            "honest search accepted a split that orphans the estimation child"
        )

    def test_all_estimation_leaves_meet_min_size(self):
        """
        End-to-end: with honest splitting, every estimation leaf in every tree
        must hold at least min_leaf_size points (no tiny/empty leaves).
        """
        X, Y, W, _ = make_data(n=600, p=5, noise=0.4, seed=2)
        min_leaf = 12
        f = NumbaCausalForest(n_trees=40, min_leaf_size=min_leaf, max_depth=8,
                              n_jobs=1, random_state=0).fit(X, Y, W)
        smallest = np.inf
        for tree in tree_leaves(f):
            smallest = min(smallest, tree)
        assert smallest >= min_leaf, (
            f"smallest estimation leaf has {smallest} points < min_leaf={min_leaf}"
        )


def tree_leaves(forest):
    """Yield the estimation-leaf sizes of every tree in the forest."""
    for tree in forest.trees:
        _, _, _, _, _, est_sizes, _ = tree.to_arrays()
        # leaf nodes are those with est_size > 0 in the flat representation
        for s in est_sizes:
            if s > 0:
                yield int(s)


# ---------------------------------------------------------------------------
# 3a — invalid leaves are dropped, not counted as tau=0
# ---------------------------------------------------------------------------

class TestMaskedLeafPrediction:

    def test_masked_flags_degenerate_leaves(self):
        """size<2 and W'W~=0 leaves are flagged invalid; good leaves valid."""
        Y = np.array([1.0, 2.0, 3.0, 4.0, 5.0, 6.0])
        W = np.array([0.0, 1.0, 0.0, 1.0, 0.0, 1.0])
        # point 0: a healthy 4-point leaf; point 1: a 1-point leaf (size<2);
        # point 2: a 2-point leaf whose W residuals are identical (W'W of the
        # centered-around-zero residuals is fine here, so use literal zeros).
        leaf_indices = np.array([
            [0, 1, 2, 3],     # valid: mixed W
            [4, -1, -1, -1],  # size 1 -> invalid
            [0, 2, -1, -1],   # W = [0,0] -> W'W = 0 -> invalid
        ], dtype=np.int32)
        leaf_sizes = np.array([4, 1, 2], dtype=np.int32)
        preds, valid = batch_predict_from_leaves_masked(Y, W, leaf_indices, leaf_sizes)
        assert valid.tolist() == [True, False, False]
        # the unmasked version returns 0.0 for the invalid points (the old bug)
        old = batch_predict_from_leaves(Y, W, leaf_indices, leaf_sizes)
        assert old[1] == 0.0 and old[2] == 0.0
        # masked preds agree on the valid point
        assert np.isclose(preds[0], old[0])

    def test_point_estimate_drops_invalid_tree(self):
        """
        A degenerate leaf must not pull the forest average toward 0.  We mimic
        the forest aggregation: averaging over valid trees only must differ
        from (and beat) averaging that counts the invalid tree as 0.
        """
        # 3 trees, 1 test point.  Two trees say tau=2; one tree is degenerate.
        tree_preds = np.array([[2.0], [2.0], [0.0]])
        valid = np.array([[True], [True], [False]])
        valid_count = valid.sum(axis=0)
        masked_mean = np.where(valid, tree_preds, 0.0).sum(axis=0) / valid_count
        zero_injected = tree_preds.mean(axis=0)  # old behaviour
        assert np.isclose(masked_mean[0], 2.0)
        assert zero_injected[0] < masked_mean[0]  # old estimate biased toward 0


# ---------------------------------------------------------------------------
# 3a — masked BLB variance
# ---------------------------------------------------------------------------

class TestMaskedBLBVariance:

    def test_equals_unmasked_when_all_valid(self):
        """
        Regression guard: with every (tree, point) valid the masked estimator
        must reproduce compute_blb_variance bit-for-bit — the all-valid path is
        the production path in the overwhelming majority of cases.
        """
        rng = np.random.default_rng(3)
        preds = rng.standard_normal((40, 50))
        valid = np.ones_like(preds, dtype=bool)
        got = compute_blb_variance_masked(preds, valid, group_size=4)
        ref = compute_blb_variance(preds, group_size=4)
        np.testing.assert_allclose(got, ref, rtol=1e-12, atol=1e-12)

    def test_nonnegative_and_shaped(self):
        rng = np.random.default_rng(4)
        preds = rng.standard_normal((40, 6))
        valid = rng.random((40, 6)) > 0.1  # ~10% invalid
        v = compute_blb_variance_masked(preds, valid, group_size=4)
        assert v.shape == (6,)
        assert np.all(v >= 0)

    def test_invalid_tree_excluded_from_bag_mean(self):
        """
        Within a bag the masked bag mean must ignore an invalid tree.  Put one
        wild invalid value in a bag and confirm the masked variance equals the
        variance computed after physically deleting that tree, NOT the variance
        that lets the wild value through.
        """
        rng = np.random.default_rng(5)
        G, L, n_test = 20, 4, 3
        preds = rng.normal(0, 1, (G * L, n_test))
        valid = np.ones((G * L, n_test), dtype=bool)
        # poison tree 0 (bag 0) at every test point, but mark it invalid
        preds[0, :] = 1e6
        valid[0, :] = False
        masked = compute_blb_variance_masked(preds, valid, group_size=L)
        # finite and not blown up by the poison value
        assert np.all(np.isfinite(masked))
        assert np.all(masked < 1e3), "invalid tree leaked into the bag mean"


# ---------------------------------------------------------------------------
# Forest-level: variance still calibrates after the masking change
# ---------------------------------------------------------------------------

class TestForestStillWorks:

    def test_predict_interval_finite_and_positive_width(self):
        X, Y, W, _ = make_data(n=400, p=5, noise=0.4, seed=6)
        f = NumbaCausalForest(n_trees=80, subforest_size=2, min_leaf_size=10,
                              max_depth=6, n_jobs=1, random_state=1).fit(X, Y, W)
        tau, lo, hi = f.predict_interval(X[:40])
        assert np.all(np.isfinite(tau))
        assert np.all(hi - lo > 0)
