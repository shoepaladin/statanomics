"""Python wrapper for Cython tree."""

import numpy as np
from dataclasses import dataclass
from typing import Optional

# Try importing the compiled Cython module
CYTHON_AVAILABLE = False
TreeBuilder = None

try:
    from ._tree_cython import TreeBuilder as CythonTreeBuilder
    TreeBuilder = CythonTreeBuilder
    CYTHON_AVAILABLE = True
except ImportError as e:
    # Print the actual error for debugging
    import sys
    if '--debug' in sys.argv:
        print(f"Cython import failed: {e}")
    CYTHON_AVAILABLE = False


@dataclass
class Node:
    feature: Optional[int] = None
    threshold: Optional[float] = None
    left: Optional['Node'] = None
    right: Optional['Node'] = None
    estimate_indices: Optional[np.ndarray] = None


class CythonCausalTree:
    """Causal tree using Cython core."""
    
    def __init__(self, min_leaf_size=10, max_depth=10):
        if not CYTHON_AVAILABLE or TreeBuilder is None:
            raise ImportError(
                "Cython extension not compiled. Run:\n"
                "  python setup.py build_ext --inplace"
            )
        
        self.min_leaf_size = min_leaf_size
        self.max_depth = max_depth
        
    def fit(self, X, Y_resid, W_resid, split_idx, est_idx):
        """Fit tree."""
        self.X = np.ascontiguousarray(X, dtype=np.float64)
        self.Y_resid = np.ascontiguousarray(Y_resid, dtype=np.float64)
        self.W_resid = np.ascontiguousarray(W_resid, dtype=np.float64)
        
        self.builder = TreeBuilder(
            self.X, self.Y_resid, self.W_resid,
            self.min_leaf_size, self.max_depth
        )
        
        self.root = self._build_tree(split_idx, est_idx, 0)
        
    def _build_tree(self, split_idx, est_idx, depth):
        """Recursively build tree."""
        node = Node()
        
        if (depth >= self.max_depth or
            len(split_idx) < 2 * self.min_leaf_size or
            len(est_idx) < self.min_leaf_size):
            node.estimate_indices = est_idx
            return node
        
        split_idx_c = np.ascontiguousarray(split_idx, dtype=np.int32)
        tau_parent = self.builder.estimate_tau_ols(split_idx_c)
        
        best_feature, best_threshold, best_score = self.builder.find_best_split(
            split_idx_c, tau_parent
        )
        
        if best_feature == -1:
            node.estimate_indices = est_idx
            return node
        
        node.feature = best_feature
        node.threshold = best_threshold
        
        split_left = self.X[split_idx, best_feature] <= best_threshold
        est_left = self.X[est_idx, best_feature] <= best_threshold
        
        node.left = self._build_tree(split_idx[split_left], est_idx[est_left], depth + 1)
        node.right = self._build_tree(split_idx[~split_left], est_idx[~est_left], depth + 1)
        
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