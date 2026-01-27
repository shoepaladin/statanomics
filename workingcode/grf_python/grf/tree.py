"""
Corrected honest causal tree implementation following GRF algorithm.
"""

import numpy as np
from dataclasses import dataclass
from typing import Optional
from sklearn.ensemble import RandomForestRegressor


@dataclass
class Node:
    """Tree node with honest sample splitting."""
    feature: Optional[int] = None
    threshold: Optional[float] = None
    left: Optional['Node'] = None
    right: Optional['Node'] = None
    estimate_indices: Optional[np.ndarray] = None


class CausalTree:
    """
    Honest causal tree with correct gradient-based splitting.
    
    Key fixes:
    1. Computes pseudo-outcomes PER PARENT NODE
    2. Maximizes difference in treatment effects (not sum of variances)
    3. Uses parent-level estimates for gradient computation
    """
    
    def __init__(self, min_leaf_size: int = 10, max_depth: int = 10):
        self.min_leaf_size = min_leaf_size
        self.max_depth = max_depth
        self.root = None
        
    def fit(self, X: np.ndarray, Y_resid: np.ndarray, W_resid: np.ndarray,
            split_indices: np.ndarray, estimate_indices: np.ndarray):
        """
        Fit tree using ORTHOGONALIZED inputs.
        
        Parameters
        ----------
        X : Covariates
        Y_resid : Y - E[Y|X] (residualized outcomes)
        W_resid : W - E[W|X] (residualized treatment)
        split_indices : Indices for determining splits
        estimate_indices : Indices for leaf estimation (disjoint)
        """
        self.X = X
        self.Y_resid = Y_resid
        self.W_resid = W_resid
        self.root = self._build_tree(split_indices, estimate_indices, depth=0)
        
    def _build_tree(self, split_idx: np.ndarray, est_idx: np.ndarray, 
                    depth: int) -> Node:
        """Recursively build tree with correct gradient splitting."""
        node = Node()
        
        # Terminal conditions
        if (depth >= self.max_depth or 
            len(split_idx) < 2 * self.min_leaf_size or
            len(est_idx) < self.min_leaf_size):
            node.estimate_indices = est_idx
            return node
        
        # STEP 1: Estimate treatment effect in PARENT node (on split sample)
        tau_parent = self._estimate_tau_ols(split_idx)
        
        # STEP 2: Compute pseudo-outcomes (gradients) for split sample
        # ρ_i = (W_i - W̄)(Y_i - τ_parent × W_i)
        W_split = self.W_resid[split_idx]
        Y_split = self.Y_resid[split_idx]
        
        W_mean = np.mean(W_split)
        pseudo_outcomes = (W_split - W_mean) * (Y_split - tau_parent * W_split)
        
        # STEP 3: Find split that maximizes difference in pseudo-outcomes
        best_score = -np.inf
        best_split = None
        
        n_features = self.X.shape[1]
        for feat in range(n_features):
            vals = self.X[split_idx, feat]
            
            for pct in [25, 50, 75]:
                thresh = np.percentile(vals, pct)
                left_mask = vals <= thresh
                right_mask = ~left_mask
                
                n_left = np.sum(left_mask)
                n_right = np.sum(right_mask)
                
                if n_left < self.min_leaf_size or n_right < self.min_leaf_size:
                    continue
                
                # Correct objective: n_L × n_R × (mean_L - mean_R)²
                mean_left = np.mean(pseudo_outcomes[left_mask])
                mean_right = np.mean(pseudo_outcomes[right_mask])
                
                score = n_left * n_right * (mean_left - mean_right)**2
                
                if score > best_score:
                    best_score = score
                    best_split = (feat, thresh, left_mask, right_mask)
        
        # No valid split found
        if best_split is None:
            node.estimate_indices = est_idx
            return node
        
        # Apply split
        feat, thresh, split_left, split_right = best_split
        node.feature = feat
        node.threshold = thresh
        
        # Split estimation sample using same rule
        est_left = self.X[est_idx, feat] <= thresh
        est_right = ~est_left
        
        # Recurse
        node.left = self._build_tree(
            split_idx[split_left],
            est_idx[est_left],
            depth + 1
        )
        node.right = self._build_tree(
            split_idx[split_right],
            est_idx[est_right],
            depth + 1
        )
        
        return node
    
    def _estimate_tau_ols(self, idx: np.ndarray) -> float:
        """Estimate treatment effect via OLS on residuals."""
        if len(idx) < 2:
            return 0.0
        
        W = self.W_resid[idx]
        Y = self.Y_resid[idx]
        
        # OLS: Y_resid ~ W_resid (no intercept since both are centered)
        denominator = np.sum(W**2)
        if denominator < 1e-10:
            return 0.0
        
        tau = np.sum(W * Y) / denominator
        return tau
    
    def get_leaf_indices(self, x: np.ndarray) -> np.ndarray:
        """Get estimation sample indices in leaf containing x."""
        node = self.root
        while node.estimate_indices is None:
            if x[node.feature] <= node.threshold:
                node = node.left
            else:
                node = node.right
        return node.estimate_indices
    
    def predict_one(self, x: np.ndarray) -> float:
        """Predict using leaf's estimation sample."""
        idx = self.get_leaf_indices(x)
        return self._estimate_tau_ols(idx)
