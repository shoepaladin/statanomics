"""
Numba-optimized causal forest with all Phase 1/2/3 improvements.
"""

import logging
import math
import os
import numpy as np
from scipy.stats import norm
from sklearn.ensemble import RandomForestRegressor
from joblib import Parallel, delayed

from .tree_numba import NumbaCausalTree
from .numba_core import (
    estimate_tau_ols_numba,
    traverse_tree_batch,
    batch_predict_from_leaves,
    accumulate_omega_tree,
    compute_delta_variance,
    compute_blb_variance,
    compute_ij_variance,
    compute_variance_from_tree_preds,
)

logger = logging.getLogger(__name__)


def _build_tree_worker(X, Y_resid, W_resid, split_idx, est_idx, seed,
                       min_leaf_size, max_depth, mtry, n_quantiles):
    """
    Module-level function so loky can pickle it for multiprocessing.

    use_parallel=False: within each worker process we run feature search
    sequentially, letting the OS schedule worker processes across cores
    instead of each tree spinning up its own Numba thread pool.
    """
    tree = NumbaCausalTree(
        min_leaf_size=min_leaf_size, max_depth=max_depth,
        mtry=mtry, n_quantiles=n_quantiles,
        use_parallel=False,
    )
    tree.fit(X, Y_resid, W_resid, split_idx, est_idx,
             rng=np.random.default_rng(seed))
    return tree


class NumbaCausalForest:
    """
    Numba-optimized causal forest.

    Parameters
    ----------
    n_trees : int
    subsample_ratio : float
        Per-tree subsample size as a fraction of n (grf `sample.fraction`,
        default 0.5).  Each tree subsamples this fraction of n from its
        group's half-sample; at 0.5 that is the entire half (the grf default).
    min_leaf_size : int
    max_depth : int
    mtry : int or None
        Features considered per split.  None → min(p, ceil(sqrt(p) + 20)),
        the R grf default (= all features for small/moderate p).
    n_quantiles : int
        Maximum candidate split thresholds per feature (default 20).
        Actual count is capped at min(n_quantiles, split_sample // 10)
        so it scales with available data.
    n_folds : int
        Cross-fitting folds for nuisance estimation (default 5, was 2).
    honesty_fraction : float
        Fraction of subsample used for determining splits (default 0.5).
    subforest_size : int
        Trees per "little bag" for BLB variance (grf `ci.group.size`,
        default 2).  Bag-mates share one half-sample; between-bag variance
        gives the calibrated CI.  n_trees is rounded down to a multiple of it.
    variance : {'blb', 'delta', 'ij'}
        Variance estimator for predict(return_std=True) / predict_interval.
        'blb' (default) replicates R grf / econml.grf (bootstrap of little
        bags + objective-Bayes debiasing).  'delta' is the delta-method SE;
        'ij' is the bias-corrected infinitesimal jackknife.
    use_parallel : bool
        Parallel feature search within each tree.
    n_jobs : int or 'auto'
        Parallel tree building jobs.
        1 = sequential (always safe, no overhead).
        -1 = use all available CPUs (loky backend).
        'auto' = automatically decide based on estimated per-tree work
                 (default). Parallelizes when n_trees × n_sub × mtry × q
                 exceeds ~1_000_000 units, the empirical break-even point
                 where loky startup overhead is justified.
    verbose : int
        0 = silent, 1 = progress.
    random_state : int or None
    """

    def __init__(self, n_trees=100, subsample_ratio=0.5,
                 min_leaf_size=10, max_depth=10,
                 mtry=None, n_quantiles=20,
                 n_folds=4, honesty_fraction=0.5,
                 subforest_size=2, variance='blb',
                 use_parallel=True, n_jobs='auto',
                 verbose=0, random_state=None):
        self.n_trees = n_trees
        self.subsample_ratio = subsample_ratio
        self.min_leaf_size = min_leaf_size
        self.max_depth = max_depth
        self.mtry = mtry
        self.n_quantiles = n_quantiles
        self.n_folds = n_folds
        self.honesty_fraction = honesty_fraction
        # Bootstrap-of-little-bags grouping for variance estimation.
        # Trees are grown in groups of `subforest_size` that share a common
        # half-sample; the between-group variance gives calibrated CIs at
        # moderate B (the estimator used by R grf / econml.grf).
        self.subforest_size = subforest_size
        self.variance = variance
        self.use_parallel = use_parallel
        self.n_jobs = n_jobs
        self.verbose = verbose
        self.random_state = random_state
        # Instance RNG — does NOT mutate global numpy state
        self._rng = np.random.default_rng(random_state)

    # ------------------------------------------------------------------
    # sklearn compatibility
    # ------------------------------------------------------------------

    def get_params(self, deep=True):
        return dict(
            n_trees=self.n_trees,
            subsample_ratio=self.subsample_ratio,
            min_leaf_size=self.min_leaf_size,
            max_depth=self.max_depth,
            mtry=self.mtry,
            n_quantiles=self.n_quantiles,
            n_folds=self.n_folds,
            honesty_fraction=self.honesty_fraction,
            subforest_size=self.subforest_size,
            variance=self.variance,
            use_parallel=self.use_parallel,
            n_jobs=self.n_jobs,
            verbose=self.verbose,
            random_state=self.random_state,
        )

    def set_params(self, **params):
        for k, v in params.items():
            setattr(self, k, v)
        if 'random_state' in params:
            self._rng = np.random.default_rng(params['random_state'])
        return self

    # ------------------------------------------------------------------
    # Auto-parallelization heuristic
    # ------------------------------------------------------------------

    def _effective_n_jobs(self):
        """
        Resolve n_jobs to an integer, applying auto-detection when n_jobs='auto'.

        Auto-detection rule (derived from tree-building-only benchmarks isolating
        loky process overhead from nuisance RF estimation):

        Per-tree work metric:  n_trees × n_sub × mtry × effective_q

        Empirical break-even (tree-build time ≥ loky startup overhead):
          work_units < 1_000_000  → sequential  (loky overhead > savings)
          work_units ≥ 1_000_000  → parallel    (1.2–2.3× speedup)

        Representative calibration points:
          n=500,  p=5,  n_trees=100:  600k  → sequential (0.88× with 2 jobs)
          n=800,  p=10, n_trees=50:  1600k  → parallel   (1.17× with 2 jobs)
          n=1000, p=10, n_trees=30:  1200k  → parallel   (1.21× with 2 jobs)
          n=1000, p=20, n_trees=50:  3500k  → parallel   (1.44× with 2 jobs)
          n=2000, p=40, n_trees=20:  5600k  → parallel   (1.66× with 2 jobs)
        """
        if self.n_jobs != 'auto':
            return self.n_jobs

        n_cpu = os.cpu_count() or 1
        if n_cpu < 2:
            return 1

        n_sub = max(2 * self.min_leaf_size * 2,
                    int(self.subsample_ratio * self.n))
        mtry_est = (self.mtry if self.mtry is not None
                    else min(self.n_features_in_,
                             math.ceil(math.sqrt(self.n_features_in_) + 20)))
        split_n = max(1, int(self.honesty_fraction * n_sub))
        q_est = max(3, min(self.n_quantiles, split_n // 10))

        work_units = self.n_trees * n_sub * mtry_est * q_est
        if work_units >= 1_000_000:
            return min(n_cpu, 4)
        return 1

    # ------------------------------------------------------------------
    # Fitting
    # ------------------------------------------------------------------

    def fit(self, X, Y, W):
        """Fit causal forest with orthogonalization."""
        self._validate_inputs(X, Y, W)
        self.X_train = np.ascontiguousarray(X, dtype=np.float64)
        self.Y_train = np.ascontiguousarray(Y, dtype=np.float64)
        self.W_train = np.ascontiguousarray(W, dtype=np.float64)
        self.n = len(X)
        self.n_features_in_ = X.shape[1]

        if self.verbose >= 1:
            logger.info("Step 1/2: Cross-fitted nuisance estimation (%d folds)...",
                        self.n_folds)

        Y_hat, W_hat = self._estimate_nuisance(
            self.X_train, self.Y_train, self.W_train
        )
        self.Y_resid = self.Y_train - Y_hat
        self.W_resid = self.W_train - W_hat

        if self.verbose >= 1:
            logger.info("Step 2/2: Growing %d trees...", self.n_trees)

        subsamples = self._draw_subsamples()
        effective_jobs = self._effective_n_jobs()

        if self.verbose >= 1 and self.n_jobs == 'auto':
            logger.info("Auto n_jobs selected: %d", effective_jobs)

        if effective_jobs != 1:
            # loky (separate processes) avoids two failure modes of threading:
            #   1. GIL: Python recursion in _build_tree holds GIL, so threading
            #      cannot truly parallelize the Python layer.
            #   2. Thread pool oversubscription: each tree uses Numba prange
            #      internally, spinning up n_cpu threads. With k joblib threads,
            #      that creates k * n_cpu threads on n_cpu cores — severe thrashing.
            # loky gives each worker its own process+address space. We disable
            # use_parallel within each worker so each process runs sequentially
            # inside and lets the OS schedule processes across cores cleanly.
            self.trees = Parallel(n_jobs=effective_jobs, backend='loky')(
                delayed(_build_tree_worker)(
                    self.X_train, self.Y_resid, self.W_resid,
                    split_idx, est_idx, seed,
                    self.min_leaf_size, self.max_depth,
                    self.mtry, self.n_quantiles
                )
                for split_idx, est_idx, seed in subsamples
            )
        else:
            self.trees = [
                self._build_single_tree(split_idx, est_idx, seed)
                for split_idx, est_idx, seed in subsamples
            ]

        # Aggregate feature importances across all trees
        self.feature_importances_ = np.mean(
            [t.feature_importances_ for t in self.trees], axis=0
        )

        if self.verbose >= 1:
            logger.info("Fit complete.")

        return self

    def _validate_inputs(self, X, Y, W):
        if self.variance not in ('blb', 'delta', 'ij'):
            raise ValueError(
                f"Unknown variance method {self.variance!r}; "
                "expected 'blb', 'delta', or 'ij'."
            )
        if X.ndim != 2:
            raise ValueError(f"X must be 2-D, got shape {X.shape}")
        n = X.shape[0]
        if len(Y) != n:
            raise ValueError(f"Y length {len(Y)} != X rows {n}")
        if len(W) != n:
            raise ValueError(f"W length {len(W)} != X rows {n}")
        if np.any(~np.isfinite(X)):
            raise ValueError("X contains NaN or Inf")
        if np.any(~np.isfinite(Y)):
            raise ValueError("Y contains NaN or Inf")
        if np.any(~np.isfinite(W)):
            raise ValueError("W contains NaN or Inf")

    def _draw_subsamples(self):
        """
        Generate (split_idx, est_idx, seed) for each tree using the
        bootstrap-of-little-bags structure required for BLB variance.

        Trees are grown in ``n_groups = n_trees // subforest_size`` groups.
        Each group draws ONE half-sample of size n//2; every tree in the
        group then subsamples ``n_sub`` points from that shared half and
        honest-splits them.  Because bag-mates share a half-sample, the
        between-group variance of bag means estimates the true sampling
        variance of tau(x) (Athey, Tibshirani & Wager 2019).

        ``n_trees`` is rounded down to a multiple of ``subforest_size`` so the
        groups are balanced; the effective count is stored back on the model.
        Uses the instance RNG so global numpy state is not mutated.
        """
        half = self.n // 2
        L = max(2, int(self.subforest_size))
        n_groups = max(1, self.n_trees // L)
        # Keep groups balanced.  We do NOT mutate the public ``n_trees`` param
        # (sklearn get_params must echo the constructor value); the effective
        # count is ``len(self.trees)`` after fitting.
        self._subforest_size = L
        self._n_trees_eff = n_groups * L

        # Subsample size: at most the half-sample (required so bag-mates can
        # differ), at least enough for an honest split with valid leaves.
        floor = 2 * self.min_leaf_size * 2
        n_sub = min(half, max(floor, int(self.subsample_ratio * self.n)))
        if n_sub > half:
            n_sub = half
        self._subsample_size = n_sub

        subsamples = []
        slices = []
        tree_id = 0
        for _ in range(n_groups):
            half_idx = self._rng.choice(self.n, half, replace=False)
            group_ids = []
            for _ in range(L):
                sub = half_idx[self._rng.choice(half, n_sub, replace=False)]
                mid = max(1, int(self.honesty_fraction * len(sub)))
                split_idx = sub[:mid]
                est_idx = sub[mid:]
                seed = int(self._rng.integers(0, 2**31))
                subsamples.append((split_idx, est_idx, seed))
                group_ids.append(tree_id)
                tree_id += 1
            slices.append(np.array(group_ids, dtype=np.int64))
        # Contiguous blocks of L trees form each bag (see compute_blb_variance).
        self.slices_ = slices
        return subsamples

    def _build_single_tree(self, split_idx, est_idx, seed):
        tree = NumbaCausalTree(
            min_leaf_size=self.min_leaf_size,
            max_depth=self.max_depth,
            mtry=self.mtry,
            n_quantiles=self.n_quantiles,
            use_parallel=self.use_parallel,
        )
        tree.fit(
            self.X_train, self.Y_resid, self.W_resid,
            split_idx, est_idx,
            rng=np.random.default_rng(seed)
        )
        return tree

    def _estimate_nuisance(self, X, Y, W):
        """
        Orthogonalization nuisances Y.hat = E[Y|X], W.hat = E[W|X], computed
        exactly as R grf's causal_forest does: out-of-bag predictions from a
        regression forest.

        grf grows ``max(50, num.trees / 4)`` trees per nuisance forest and
        takes OOB predictions (each obs predicted only by trees that did not
        include it).  We mirror that with a bagged RandomForestRegressor and
        its ``oob_prediction_``.  This is lower-variance than the previous
        k-fold cross-fit (which trained a separate model per held-out fold)
        and is what makes grf's CI calibration robust at small n.

        ``n_folds`` is retained only as a deprecated no-op fallback for any obs
        that happens to have no OOB trees (impossible in practice at >=50
        trees, but handled defensively).
        """
        n_nuisance_trees = max(50, self.n_trees // 4)
        rf_params = {
            'n_estimators': n_nuisance_trees,
            # grf regression_forest defaults: deep honest trees, min.node.size=5.
            'min_samples_leaf': 5,
            'max_features': max(1, math.ceil(self.n_features_in_ / 3)),
            'bootstrap': True,
            'oob_score': True,
            'n_jobs': -1,
            'random_state': int(self._rng.integers(0, 2**31)),
        }
        rf_Y = RandomForestRegressor(**rf_params).fit(X, Y)
        # Independent randomness for the W forest (distinct bootstraps/feature
        # draws) rather than reusing the Y forest's seed.
        rf_params['random_state'] = int(self._rng.integers(0, 2**31))
        rf_W = RandomForestRegressor(**rf_params).fit(X, W)
        Y_hat = rf_Y.oob_prediction_
        W_hat = rf_W.oob_prediction_

        # Defensive: any obs in every bootstrap has no OOB estimate (NaN).
        # Fall back to the in-bag prediction for just those points.
        bad_Y = ~np.isfinite(Y_hat)
        bad_W = ~np.isfinite(W_hat)
        if bad_Y.any():
            Y_hat[bad_Y] = rf_Y.predict(X[bad_Y])
        if bad_W.any():
            W_hat[bad_W] = rf_W.predict(X[bad_W])

        return Y_hat, W_hat

    # ------------------------------------------------------------------
    # Prediction
    # ------------------------------------------------------------------

    def predict(self, X, return_std=False):
        """
        Predict CATEs (and, if ``return_std``, standard errors).

        Uses batch JIT-compiled tree traversal instead of per-point Python
        while-loops.  When ``return_std`` is set, the standard error is computed
        by the configured ``variance`` estimator (default 'blb' — the R grf
        bootstrap-of-little-bags; see ``_compute_variance``).
        """
        if not hasattr(self, 'trees'):
            raise RuntimeError("Call fit() before predict()")
        if X.shape[1] != self.n_features_in_:
            raise ValueError(
                f"X has {X.shape[1]} features, expected {self.n_features_in_}"
            )

        X = np.ascontiguousarray(X, dtype=np.float64)
        n_test = len(X)

        # Collect per-tree predictions.  For the delta method we also
        # accumulate the forest OLS weight matrix Omega during the same loop.
        n_trees_eff = len(self.trees)
        tree_preds = np.zeros((n_trees_eff, n_test))
        need_omega = return_std and self.variance == 'delta'
        if need_omega:
            Omega = np.zeros((self.n, n_test), dtype=np.float64)

        for b, tree in enumerate(self.trees):
            feats, threshs, left_ch, right_ch, starts, sizes, flat_idx = \
                tree.to_arrays()

            max_leaf = int(sizes.max()) if sizes.max() > 0 else 1

            leaf_indices, leaf_sizes = traverse_tree_batch(
                X, feats, threshs, left_ch, right_ch,
                starts, sizes, flat_idx, max_leaf
            )

            tree_preds[b] = batch_predict_from_leaves(
                self.Y_resid, self.W_resid, leaf_indices, leaf_sizes
            )

            if need_omega:
                accumulate_omega_tree(
                    Omega, leaf_indices, leaf_sizes, self.W_resid, n_trees_eff
                )

        tau_hat = np.mean(tree_preds, axis=0)

        if not return_std:
            return tau_hat

        variances = self._compute_variance(tree_preds,
                                           Omega if need_omega else None)
        std = np.sqrt(np.maximum(variances, 0.0))
        return tau_hat, std

    def _compute_variance(self, tree_preds, Omega):
        """
        Dispatch to the configured variance estimator.

        'blb'   : bootstrap-of-little-bags (default) — calibrated at moderate B.
        'delta' : delta-method OLS-weight variance.
        'ij'    : bias-corrected infinitesimal jackknife (needs large B).
        """
        method = self.variance
        if method == 'blb':
            return compute_blb_variance(tree_preds, self._subforest_size)
        if method == 'delta':
            return compute_delta_variance(Omega, self.Y_resid)
        if method == 'ij':
            flags = np.array([t.in_subsample_ for t in self.trees], dtype=bool)
            return compute_ij_variance(
                tree_preds, flags, self._subsample_size, self.n,
                bias_correction=True,
            )
        raise ValueError(
            f"Unknown variance method {method!r}; "
            "expected 'blb', 'delta', or 'ij'."
        )

    def predict_interval(self, X, alpha=0.05):
        tau, std = self.predict(X, return_std=True)
        z = norm.ppf(1 - alpha / 2)
        return tau, tau - z * std, tau + z * std

    # ------------------------------------------------------------------
    # OOB predictions
    # ------------------------------------------------------------------

    def oob_predict(self):
        """
        Out-of-bag CATE predictions for all training points.

        Each training point is predicted using only trees that did NOT
        include it in their subsample, so there is no train/test leakage.

        Returns
        -------
        oob_tau : float array (n,)
            NaN for any point that was in-sample for every tree.
        """
        if not hasattr(self, 'trees'):
            raise RuntimeError("Call fit() before oob_predict()")

        oob_sum = np.zeros(self.n)
        oob_count = np.zeros(self.n, dtype=int)
        X = self.X_train

        for tree in self.trees:
            # Points NOT in this tree's subsample
            oob_mask = ~tree.in_subsample_
            oob_idx = np.where(oob_mask)[0]
            if len(oob_idx) == 0:
                continue

            feats, threshs, left_ch, right_ch, starts, sizes, flat_idx = \
                tree.to_arrays()
            max_leaf = int(sizes.max()) if sizes.max() > 0 else 1

            leaf_indices, leaf_sizes = traverse_tree_batch(
                X[oob_idx], feats, threshs, left_ch, right_ch,
                starts, sizes, flat_idx, max_leaf
            )
            preds = batch_predict_from_leaves(
                self.Y_resid, self.W_resid, leaf_indices, leaf_sizes
            )

            oob_sum[oob_idx] += preds
            oob_count[oob_idx] += 1

        oob_tau = np.where(oob_count > 0, oob_sum / oob_count, np.nan)
        return oob_tau

    # ------------------------------------------------------------------
    # econml-compatible API
    # ------------------------------------------------------------------

    def effect(self, X):
        return self.predict(X, return_std=False)

    def effect_interval(self, X, alpha=0.05):
        _, lower, upper = self.predict_interval(X, alpha)
        return lower.reshape(-1, 1), upper.reshape(-1, 1)
