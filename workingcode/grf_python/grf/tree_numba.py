"""
Causal tree using Numba-compiled functions.
"""

import math
import numpy as np
from dataclasses import dataclass
from typing import Optional
from .numba_core import (
    estimate_tau_ols_numba,
    find_best_split_honest_parallel,
    find_best_split_honest_numba,
)


@dataclass
class Node:
    feature: Optional[int] = None
    threshold: Optional[float] = None
    left: Optional['Node'] = None
    right: Optional['Node'] = None
    estimate_indices: Optional[np.ndarray] = None


class NumbaCausalTree:
    """
    Causal tree with Numba-accelerated splitting.

    Parameters
    ----------
    min_leaf_size : int
    max_depth : int
    mtry : int or None
        Number of features to consider at each split.
        None → ceil(sqrt(p)).  p → all features (no subsampling).
    n_quantiles : int
        Number of candidate split thresholds per feature (uniformly spaced
        percentiles from 5 to 95).  Was hardcoded to 3 previously.
    use_parallel : bool
        Use parallel feature search.
    """

    def __init__(self, min_leaf_size=10, max_depth=10,
                 mtry=None, n_quantiles=20, use_parallel=True):
        self.min_leaf_size = min_leaf_size
        self.max_depth = max_depth
        self.mtry = mtry
        self.n_quantiles = n_quantiles
        self.use_parallel = use_parallel

    def fit(self, X, Y_resid, W_resid, split_idx, est_idx, rng=None):
        """
        Fit tree.

        Parameters
        ----------
        rng : np.random.Generator or None
            Instance RNG for reproducibility without global seed mutation.
        """
        self.X = X
        self.Y_resid = Y_resid
        self.W_resid = W_resid
        self._rng = rng if rng is not None else np.random.default_rng()

        n_features = X.shape[1]
        # R grf default: mtry = min(ceil(sqrt(p) + 20), p).  For small/moderate
        # p this is ALL features; the previous ceil(p/3) under-sampled features
        # badly (e.g. 2 of 5), starving splits of the relevant covariate and
        # producing heavily attenuated CATEs (slope ~0.6 vs grf's ~0.76).
        self._mtry = (self.mtry if self.mtry is not None
                      else min(n_features, math.ceil(math.sqrt(n_features) + 20)))

        # Adaptive quantile grid: denser grids help when the split sample is
        # large, but thin out past ~1 obs/bin (which adds noise, not signal).
        # Cap at min(n_quantiles, split_sample_size // 10) so we never try
        # more thresholds than the data can support.
        n_split = len(split_idx)
        effective_q = max(3, min(self.n_quantiles, n_split // 10))
        self._percentiles = np.linspace(5.0, 95.0, effective_q)

        # Per-feature importance accumulated during tree building
        self.feature_importances_ = np.zeros(n_features)

        # Track which training points were in this tree's subsample
        n_train = len(X)
        self.in_subsample_ = np.zeros(n_train, dtype=bool)
        all_idx = np.concatenate([split_idx, est_idx])
        self.in_subsample_[all_idx] = True

        self.root = self._build_tree(split_idx, est_idx, 0)

        # Normalize importances by total samples seen across all splits
        total = self.feature_importances_.sum()
        if total > 0:
            self.feature_importances_ /= total

        # Cache flat array representation for fast batch traversal
        self._arrays = None  # lazy-built on first predict call

    def _sample_features(self, n_features):
        """Return a random mtry-sized feature subset (as int32 array)."""
        if self._mtry >= n_features:
            return np.arange(n_features, dtype=np.int32)
        indices = self._rng.choice(n_features, size=self._mtry, replace=False)
        return np.sort(indices).astype(np.int32)

    def _build_tree(self, split_idx, est_idx, depth):
        node = Node()

        if (depth >= self.max_depth
                or len(split_idx) < 2 * self.min_leaf_size
                or len(est_idx) < self.min_leaf_size):
            node.estimate_indices = est_idx
            return node

        tau_parent = estimate_tau_ols_numba(
            self.Y_resid, self.W_resid, split_idx
        )

        n_features = self.X.shape[1]
        feature_indices = self._sample_features(n_features)

        # Honest split search: enforce min_leaf on BOTH the splitting sample
        # and the estimation sample, so the estimation child of an accepted
        # split is never orphaned (issue #5, item 3b).  The score is still a
        # function of the splitting sample only.
        if self.use_parallel and len(feature_indices) > 1:
            feat, thresh, score = find_best_split_honest_parallel(
                self.X, self.Y_resid, self.W_resid,
                split_idx, est_idx, tau_parent, self.min_leaf_size,
                self._percentiles, feature_indices
            )
        else:
            feat, thresh, score = find_best_split_honest_numba(
                self.X, self.Y_resid, self.W_resid,
                split_idx, est_idx, tau_parent, self.min_leaf_size,
                self._percentiles, feature_indices
            )

        if feat == -1:
            node.estimate_indices = est_idx
            return node

        node.feature = feat
        node.threshold = thresh

        # Accumulate importance: score weighted by node sample count
        self.feature_importances_[feat] += score * len(split_idx)

        split_left = self.X[split_idx, feat] <= thresh
        est_left = self.X[est_idx, feat] <= thresh

        node.left = self._build_tree(
            split_idx[split_left], est_idx[est_left], depth + 1
        )
        node.right = self._build_tree(
            split_idx[~split_left], est_idx[~est_left], depth + 1
        )

        return node

    def get_leaf_indices(self, x):
        """Single-point traversal (Python while-loop, for compatibility)."""
        node = self.root
        while node.estimate_indices is None:
            if x[node.feature] <= node.threshold:
                node = node.left
            else:
                node = node.right
        return node.estimate_indices

    def to_arrays(self):
        """
        Convert tree to flat arrays for JIT-compiled batch traversal.

        Returns a tuple compatible with traverse_tree_batch().
        Result is cached after the first call.
        """
        if self._arrays is not None:
            return self._arrays

        # BFS to collect all nodes in a stable order
        nodes = []
        queue = [self.root]
        node_id_to_idx = {}

        while queue:
            node = queue.pop(0)
            idx = len(nodes)
            node_id_to_idx[id(node)] = idx
            nodes.append(node)
            if node.estimate_indices is None:  # internal node
                queue.append(node.left)
                queue.append(node.right)

        n_nodes = len(nodes)
        features = np.full(n_nodes, -1, dtype=np.int32)
        thresholds = np.zeros(n_nodes, dtype=np.float64)
        left_children = np.full(n_nodes, -1, dtype=np.int32)
        right_children = np.full(n_nodes, -1, dtype=np.int32)
        est_starts = np.zeros(n_nodes, dtype=np.int32)
        est_sizes = np.zeros(n_nodes, dtype=np.int32)

        flat_indices = []

        for i, node in enumerate(nodes):
            if node.estimate_indices is None:
                features[i] = node.feature
                thresholds[i] = node.threshold
                left_children[i] = node_id_to_idx[id(node.left)]
                right_children[i] = node_id_to_idx[id(node.right)]
            else:
                est_starts[i] = len(flat_indices)
                est_sizes[i] = len(node.estimate_indices)
                flat_indices.extend(node.estimate_indices.tolist())

        est_indices_flat = np.array(flat_indices, dtype=np.int32)

        self._arrays = (features, thresholds, left_children, right_children,
                        est_starts, est_sizes, est_indices_flat)
        return self._arrays
