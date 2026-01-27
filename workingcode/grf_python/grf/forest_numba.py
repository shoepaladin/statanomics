"""
Numba-optimized causal forest (Phase 1).
"""

import numpy as np
from scipy.stats import norm
from sklearn.ensemble import RandomForestRegressor
from .tree_numba import NumbaCausalTree
from .numba_core import (
    estimate_tau_ols_numba,
    compute_tree_weights_numba,
    weighted_ols_prediction,
    compute_ij_variance_numba
)


class NumbaCausalForest:
    """
    Phase 1: Numba-optimized causal forest.
    
    Expected speedup: 2-4x over pure Python
    
    Key optimizations:
    - JIT-compiled split finding
    - Parallel feature search
    - Fast weight computation
    - Vectorized predictions
    """
    
    def __init__(self, n_trees=100, subsample_ratio=0.5,
                 min_leaf_size=10, max_depth=10,
                 use_parallel=True, random_state=None):
        self.n_trees = n_trees
        self.subsample_ratio = subsample_ratio
        self.min_leaf_size = min_leaf_size
        self.max_depth = max_depth
        self.use_parallel = use_parallel
        self.random_state = random_state
        
        if random_state is not None:
            np.random.seed(random_state)
    
    def fit(self, X, Y, W):
        """Fit forest with orthogonalization."""
        self.X_train = X
        self.Y_train = Y
        self.W_train = W
        self.n = len(X)
        
        # Step 1: Orthogonalization
        print("Step 1/2: Orthogonalization (cross-fitted nuisance estimation)...")
        Y_hat, W_hat = self._estimate_nuisance(X, Y, W)
        
        self.Y_resid = Y - Y_hat
        self.W_resid = W - W_hat
        
        # Step 2: Build forest
        print(f"Step 2/2: Growing {self.n_trees} trees (Numba-accelerated)...")
        self.trees = []
        
        for i in range(self.n_trees):
            if (i + 1) % 20 == 0:
                print(f"  Tree {i+1}/{self.n_trees}")
            
            # Subsample
            n_sub = int(self.subsample_ratio * self.n)
            idx = np.random.choice(self.n, n_sub, replace=False)
            np.random.shuffle(idx)
            
            # Honest split
            mid = len(idx) // 2
            split_idx = idx[:mid]
            est_idx = idx[mid:]
            
            tree = NumbaCausalTree(
                self.min_leaf_size,
                self.max_depth,
                self.use_parallel
            )
            tree.fit(X, self.Y_resid, self.W_resid, split_idx, est_idx)
            self.trees.append(tree)
        
        print("✓ Fit complete")
        return self
    
    def _estimate_nuisance(self, X, Y, W):
        """Cross-fitted nuisance function estimation."""
        n = len(X)
        Y_hat = np.zeros(n)
        W_hat = np.zeros(n)
        
        # 2-fold cross-fitting
        n_half = n // 2
        idx1 = np.arange(n_half)
        idx2 = np.arange(n_half, n)
        
        # Fold 1
        rf_Y_1 = RandomForestRegressor(n_estimators=50, max_depth=10, n_jobs=-1)
        rf_W_1 = RandomForestRegressor(n_estimators=50, max_depth=10, n_jobs=-1)
        rf_Y_1.fit(X[idx1], Y[idx1])
        rf_W_1.fit(X[idx1], W[idx1])
        Y_hat[idx2] = rf_Y_1.predict(X[idx2])
        W_hat[idx2] = rf_W_1.predict(X[idx2])
        
        # Fold 2
        rf_Y_2 = RandomForestRegressor(n_estimators=50, max_depth=10, n_jobs=-1)
        rf_W_2 = RandomForestRegressor(n_estimators=50, max_depth=10, n_jobs=-1)
        rf_Y_2.fit(X[idx2], Y[idx2])
        rf_W_2.fit(X[idx2], W[idx2])
        Y_hat[idx1] = rf_Y_2.predict(X[idx1])
        W_hat[idx1] = rf_W_2.predict(X[idx1])
        
        return Y_hat, W_hat
    
    def predict(self, X, return_std=False):
        """Fast prediction with optional inference."""
        n_test = len(X)
        
        if not return_std:
            # Fast path: simple averaging
            preds = np.zeros(n_test)
            for x_i in range(n_test):
                tree_preds = []
                for tree in self.trees:
                    idx = tree.get_leaf_indices(X[x_i])
                    tau = estimate_tau_ols_numba(
                        self.Y_resid, self.W_resid, idx
                    )
                    tree_preds.append(tau)
                preds[x_i] = np.mean(tree_preds)
            return preds
        
        # Inference path: compute weights
        print("Computing forest weights...")
        
        # Pack leaf indices for Numba
        max_leaf_size = self.n // 2
        tree_weights_list = []
        
        for b, tree in enumerate(self.trees):
            leaf_indices = -np.ones((n_test, max_leaf_size), dtype=np.int32)
            leaf_sizes = np.zeros(n_test, dtype=np.int32)
            
            for i in range(n_test):
                idx = tree.get_leaf_indices(X[i])
                n_leaf = min(len(idx), max_leaf_size)
                leaf_indices[i, :n_leaf] = idx[:n_leaf]
                leaf_sizes[i] = n_leaf
            
            # Compute weights for this tree (Numba)
            weights_b = compute_tree_weights_numba(
                leaf_indices, leaf_sizes, self.n
            )
            tree_weights_list.append(weights_b)
        
        # Stack and average
        tree_weights = np.array(tree_weights_list)  # (n_trees, n_test, n_train)
        avg_weights = np.mean(tree_weights, axis=0)
        
        # Weighted predictions
        print("Computing weighted predictions...")
        tau_hat = np.zeros(n_test)
        for i in range(n_test):
            tau_hat[i] = weighted_ols_prediction(
                self.Y_resid, self.W_resid, avg_weights[i]
            )
        
        # IJ variance
        print("Computing IJ variance...")
        W_c = self.W_resid - np.mean(self.W_resid)
        Y_c = self.Y_resid - np.mean(self.Y_resid)
        psi = W_c * Y_c
        
        variances = compute_ij_variance_numba(tree_weights, avg_weights, psi)
        std = np.sqrt(variances)
        
        return tau_hat, std
    
    def predict_interval(self, X, alpha=0.05):
        """Predict with confidence intervals."""
        tau, std = self.predict(X, return_std=True)
        z = norm.ppf(1 - alpha/2)
        lower = tau - z * std
        upper = tau + z * std
        return tau, lower, upper
    
    def effect(self, X):
        """econml-compatible API."""
        return self.predict(X, return_std=False)
    
    def effect_interval(self, X, alpha=0.05):
        """econml-compatible API."""
        _, lower, upper = self.predict_interval(X, alpha)
        return lower.reshape(-1, 1), upper.reshape(-1, 1)

