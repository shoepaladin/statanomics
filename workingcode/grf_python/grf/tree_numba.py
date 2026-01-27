"""
Causal tree using Numba-compiled functions.
"""

import numpy as np
from dataclasses import dataclass
from typing import Optional
from .numba_core import (
    estimate_tau_ols_numba,
    find_best_split_parallel,
    find_best_split_numba
)


@dataclass
class Node:
    feature: Optional[int] = None
    threshold: Optional[float] = None
    left: Optional['Node'] = None
    right: Optional['Node'] = None
    estimate_indices: Optional[np.ndarray] = None


class NumbaCausalTree:
    """Causal tree with Numba-accelerated splitting."""
    
    def __init__(self, min_leaf_size=10, max_depth=10, use_parallel=True):
        self.min_leaf_size = min_leaf_size
        self.max_depth = max_depth
        self.use_parallel = use_parallel
        self.percentiles = np.array([25.0, 50.0, 75.0])
        
    def fit(self, X, Y_resid, W_resid, split_idx, est_idx):
        """Fit tree."""
        self.X = X
        self.Y_resid = Y_resid
        self.W_resid = W_resid
        self.root = self._build_tree(split_idx, est_idx, 0)
        
    def _build_tree(self, split_idx, est_idx, depth):
        """Recursively build tree."""
        node = Node()
        
        # Terminal conditions
        if (depth >= self.max_depth or
            len(split_idx) < 2 * self.min_leaf_size or
            len(est_idx) < self.min_leaf_size):
            node.estimate_indices = est_idx
            return node
        
        # Estimate parent treatment effect (Numba)
        tau_parent = estimate_tau_ols_numba(
            self.Y_resid, self.W_resid, split_idx
        )
        
        # Find best split (Numba, optionally parallel)
        if self.use_parallel:
            feat, thresh, score = find_best_split_parallel(
                self.X, self.Y_resid, self.W_resid,
                split_idx, tau_parent, self.min_leaf_size,
                self.percentiles
            )
        else:
            feat, thresh, score = find_best_split_numba(
                self.X, self.Y_resid, self.W_resid,
                split_idx, tau_parent, self.min_leaf_size,
                self.percentiles
            )
        
        if feat == -1:
            node.estimate_indices = est_idx
            return node
        
        # Apply split
        node.feature = feat
        node.threshold = thresh
        
        split_left = self.X[split_idx, feat] <= thresh
        est_left = self.X[est_idx, feat] <= thresh
        
        # Recurse
        node.left = self._build_tree(
            split_idx[split_left],
            est_idx[est_left],
            depth + 1
        )
        node.right = self._build_tree(
            split_idx[~split_left],
            est_idx[~est_left],
            depth + 1
        )
        
        return node
    
    def get_leaf_indices(self, x):
        """Get leaf indices for a test point."""
        node = self.root
        while node.estimate_indices is None:
            if x[node.feature] <= node.threshold:
                node = node.left
            else:
                node = node.right
        return node.estimate_indices

