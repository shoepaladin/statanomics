# ============================================================================
# File: grf/tree_cython.pyx
# Python wrapper for Cython tree
# ============================================================================
"""
Python interface to Cython-optimized tree.
"""

import numpy as np
from dataclasses import dataclass
from typing import Optional

try:
    from ._tree_cython import CythonTreeBuilder, SplitResult
    CYTHON_AVAILABLE = True
except ImportError:
    CYTHON_AVAILABLE = False
    print("Warning: Cython extension not compiled. Run: python setup.py build_ext --inplace")


@dataclass
class Node:
    feature: Optional[int] = None
    threshold: Optional[float] = None
    left: Optional['Node'] = None
    right: Optional['Node'] = None
    estimate_indices: Optional[np.ndarray] = None


class CythonCausalTree:
    """Causal tree using Cython-compiled core."""
    
    def __init__(self, min_leaf_size=10, max_depth=10):
        if not CYTHON_AVAILABLE:
            raise ImportError("Cython extension not available")
        
        self.min_leaf_size = min_leaf_size
        self.max_depth = max_depth
        
    def fit(self, X, Y_resid, W_resid, split_idx, est_idx):
        """Fit tree."""
        self.X = np.ascontiguousarray(X, dtype=np.float64)
        self.Y_resid = np.ascontiguousarray(Y_resid, dtype=np.float64)
        self.W_resid = np.ascontiguousarray(W_resid, dtype=np.float64)
        
        self.builder = CythonTreeBuilder(
            self.X, self.Y_resid, self.W_resid,
            self.min_leaf_size, self.max_depth
        )
        
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
        
        # Estimate parent tau
        split_idx_c = np.ascontiguousarray(split_idx, dtype=np.int32)
        tau_parent = self.builder.estimate_tau_ols(split_idx_c)
        
        # Find best split (Cython)
        split_result = self.builder.find_best_split(split_idx_c, tau_parent)
        
        if split_result.feature == -1:
            node.estimate_indices = est_idx
            return node
        
        # Apply split
        node.feature = split_result.feature
        node.threshold = split_result.threshold
        
        split_left = self.X[split_idx, split_result.feature] <= split_result.threshold
        est_left = self.X[est_idx, split_result.feature] <= split_result.threshold
        
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
        """Get leaf indices."""
        node = self.root
        while node.estimate_indices is None:
            if x[node.feature] <= node.threshold:
                node = node.left
            else:
                node = node.right
        return node.estimate_indices
