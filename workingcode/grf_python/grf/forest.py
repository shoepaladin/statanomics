"""
Corrected causal forest with orthogonalization.
"""

import numpy as np
from scipy.stats import norm
from typing import Tuple, Optional
from sklearn.ensemble import RandomForestRegressor
from .tree import CausalTree


class CausalForest:
    """
    Causal forest with R-learner orthogonalization.
    
    Key fixes:
    1. Estimates E[Y|X] and E[W|X] first (nuisance functions)
    2. Works with residuals throughout
    3. Correct gradient-based splitting
    4. Weighted regression for prediction
    """
    
    def __init__(self, n_trees: int = 100, subsample_ratio: float = 0.5,
                 min_leaf_size: int = 10, max_depth: int = 10,
                 random_state: Optional[int] = None):
        self.n_trees = n_trees
        self.subsample_ratio = subsample_ratio
        self.min_leaf_size = min_leaf_size
        self.max_depth = max_depth
        self.random_state = random_state
        self.trees = []
        
        if random_state is not None:
            np.random.seed(random_state)
    
    def fit(self, X: np.ndarray, Y: np.ndarray, W: np.ndarray):
        """
        Fit causal forest with orthogonalization.
        
        Steps:
        1. Estimate E[Y|X] and E[W|X] using out-of-bag predictions
        2. Compute residuals
        3. Build forest on residuals
        """
        self.X_train = X
        self.Y_train = Y
        self.W_train = W
        self.n = len(X)
        
        # STEP 1: Orthogonalization (R-learner)
        print("Step 1/2: Estimating nuisance functions E[Y|X] and E[W|X]...")
        Y_hat, W_hat = self._estimate_nuisance_functions(X, Y, W)
        
        # Compute residuals
        self.Y_resid = Y - Y_hat
        self.W_resid = W - W_hat
        
        # STEP 2: Build forest on residuals
        print("Step 2/2: Growing causal forest...")
        for i in range(self.n_trees):
            # Subsample
            n_sub = int(self.subsample_ratio * self.n)
            idx = np.random.choice(self.n, n_sub, replace=False)
            np.random.shuffle(idx)
            
            # Honest split
            mid = len(idx) // 2
            split_idx = idx[:mid]
            est_idx = idx[mid:]
            
            # Grow tree
            tree = CausalTree(self.min_leaf_size, self.max_depth)
            tree.fit(X, self.Y_resid, self.W_resid, split_idx, est_idx)
            self.trees.append(tree)
        
        return self
    
    def _estimate_nuisance_functions(self, X, Y, W):
        """
        Estimate E[Y|X] and E[W|X] using out-of-bag random forests.
        
        This is the critical orthogonalization step.
        """
        n = len(X)
        Y_hat = np.zeros(n)
        W_hat = np.zeros(n)
        
        # Use k-fold cross-fitting (k=2 for simplicity)
        n_half = n // 2
        idx1 = np.arange(n_half)
        idx2 = np.arange(n_half, n)
        
        # Fit on first half, predict on second
        rf_Y_1 = RandomForestRegressor(n_estimators=50, max_depth=10, random_state=42)
        rf_W_1 = RandomForestRegressor(n_estimators=50, max_depth=10, random_state=42)
        
        rf_Y_1.fit(X[idx1], Y[idx1])
        rf_W_1.fit(X[idx1], W[idx1])
        
        Y_hat[idx2] = rf_Y_1.predict(X[idx2])
        W_hat[idx2] = rf_W_1.predict(X[idx2])
        
        # Fit on second half, predict on first
        rf_Y_2 = RandomForestRegressor(n_estimators=50, max_depth=10, random_state=43)
        rf_W_2 = RandomForestRegressor(n_estimators=50, max_depth=10, random_state=43)
        
        rf_Y_2.fit(X[idx2], Y[idx2])
        rf_W_2.fit(X[idx2], W[idx2])
        
        Y_hat[idx1] = rf_Y_2.predict(X[idx1])
        W_hat[idx1] = rf_W_2.predict(X[idx1])
        
        return Y_hat, W_hat
    
    def predict(self, X: np.ndarray, return_std: bool = False) -> Tuple:
        """Predict treatment effects."""
        n_test = len(X)
        
        if not return_std:
            # Simple averaging
            preds = np.mean([tree.predict(X) for tree in self.trees], axis=0)
            return preds
        
        # Compute forest weights for inference
        weights = np.zeros((n_test, self.n))
        tree_weights = np.zeros((self.n_trees, n_test, self.n))
        
        for b, tree in enumerate(self.trees):
            for i, x in enumerate(X):
                leaf_idx = tree.get_leaf_indices(x)
                if len(leaf_idx) > 0:
                    tree_weights[b, i, leaf_idx] = 1.0 / len(leaf_idx)
        
        weights = np.mean(tree_weights, axis=0)
        
        # Weighted regression predictions
        tau_hat = np.zeros(n_test)
        for i in range(n_test):
            # Weighted OLS: Y_resid ~ W_resid with weights α_i(x)
            w = weights[i]
            if np.sum(w) > 0:
                w = w / np.sum(w)  # Normalize
                W_weighted = self.W_resid * w
                Y_weighted = self.Y_resid * w
                
                denom = np.sum(W_weighted * self.W_resid)
                if abs(denom) > 1e-10:
                    tau_hat[i] = np.sum(Y_weighted * self.W_resid) / denom
        
        # IJ variance
        W_centered = self.W_resid - np.mean(self.W_resid)
        Y_centered = self.Y_resid - np.mean(self.Y_resid)
        psi = W_centered * Y_centered
        
        variances = np.zeros(n_test)
        for i in range(n_test):
            deviations = tree_weights[:, i, :] - weights[i]
            agg_dev = np.sum(deviations, axis=0)
            variances[i] = np.sum(agg_dev**2 * psi**2)
        
        std = np.sqrt(variances)
        
        return tau_hat, std
    
    def predict_interval(self, X: np.ndarray, alpha: float = 0.05):
        """Predict with confidence intervals."""
        tau, std = self.predict(X, return_std=True)
        z = norm.ppf(1 - alpha/2)
        return tau, tau - z*std, tau + z*std
    
    def effect(self, X: np.ndarray) -> np.ndarray:
        """econml-compatible API."""
        return self.predict(X)
    
    def effect_interval(self, X: np.ndarray, alpha: float = 0.05):
        """econml-compatible API."""
        _, lower, upper = self.predict_interval(X, alpha)
        return lower.reshape(-1, 1), upper.reshape(-1, 1)

