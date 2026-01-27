"""Cython causal forest with parallelism."""

import numpy as np
from scipy.stats import norm
from sklearn.ensemble import RandomForestRegressor
from joblib import Parallel, delayed, parallel_backend
import time

# Check if Cython modules are available
CYTHON_AVAILABLE = False
CythonCausalTree = None
compute_forest_weights_cython = None

try:
    from .tree_cython import CythonCausalTree as _CythonCausalTree
    from ._weights_cython import compute_forest_weights_cython as _compute_weights
    CythonCausalTree = _CythonCausalTree
    compute_forest_weights_cython = _compute_weights
    CYTHON_AVAILABLE = True
except ImportError as e:
    import sys
    if '--debug' in sys.argv:
        print(f"Cython import error: {e}")
    CYTHON_AVAILABLE = False


class CausalForestCython:
    """Production causal forest with Cython."""
    
    def __init__(self, n_trees=100, subsample_ratio=0.5,
                 min_leaf_size=10, max_depth=10,
                 n_jobs=-1, random_state=None):
        if not CYTHON_AVAILABLE:
            raise ImportError(
                "Cython extensions not compiled. Run:\n"
                "  python setup.py build_ext --inplace"
            )
        
        self.n_trees = n_trees
        self.subsample_ratio = subsample_ratio
        self.min_leaf_size = min_leaf_size
        self.max_depth = max_depth
        self.n_jobs = n_jobs if n_jobs > 0 else -1
        self.random_state = random_state
        
        if random_state is not None:
            np.random.seed(random_state)
    
    def fit(self, X, Y, W):
        """Fit with parallel tree building using threading (Cython-safe)."""
        self.X_train = np.ascontiguousarray(X, dtype=np.float64)
        self.Y_train = np.ascontiguousarray(Y, dtype=np.float64)
        self.W_train = np.ascontiguousarray(W, dtype=np.float64)
        self.n = len(X)
        
        print("Step 1/2: Orthogonalization...")
        t0 = time.time()
        Y_hat, W_hat = self._estimate_nuisance(X, Y, W)
        self.Y_resid = Y - Y_hat
        self.W_resid = W - W_hat
        print(f"  Completed in {time.time() - t0:.2f}s")
        
        print(f"Step 2/2: Growing {self.n_trees} trees (Cython + parallel with threading)...")
        t0 = time.time()
        
        # Generate subsamples
        subsamples = []
        for i in range(self.n_trees):
            n_sub = int(self.subsample_ratio * self.n)
            idx = np.random.choice(self.n, n_sub, replace=False)
            np.random.shuffle(idx)
            mid = len(idx) // 2
            subsamples.append((idx[:mid], idx[mid:]))
        
        # Use threading backend (Cython objects can't be pickled)
        # Threading works because Python's GIL is released in Cython nogil blocks
        with parallel_backend('threading', n_jobs=self.n_jobs):
            self.trees = Parallel(verbose=0)(
                delayed(self._build_single_tree)(split_idx, est_idx)
                for split_idx, est_idx in subsamples
            )
        
        print(f"  Completed in {time.time() - t0:.2f}s")
        return self
    
    def _build_single_tree(self, split_idx, est_idx):
        """Build a single tree."""
        tree = CythonCausalTree(self.min_leaf_size, self.max_depth)
        tree.fit(self.X_train, self.Y_resid, self.W_resid, split_idx, est_idx)
        return tree
    
    def _estimate_nuisance(self, X, Y, W):
        """Cross-fitted nuisance estimation."""
        n = len(X)
        Y_hat = np.zeros(n)
        W_hat = np.zeros(n)
        
        n_half = n // 2
        idx1 = np.arange(n_half)
        idx2 = np.arange(n_half, n)
        
        rf_params = {'n_estimators': 50, 'max_depth': 10, 'n_jobs': -1}
        
        rf_Y_1 = RandomForestRegressor(**rf_params)
        rf_W_1 = RandomForestRegressor(**rf_params)
        rf_Y_1.fit(X[idx1], Y[idx1])
        rf_W_1.fit(X[idx1], W[idx1])
        Y_hat[idx2] = rf_Y_1.predict(X[idx2])
        W_hat[idx2] = rf_W_1.predict(X[idx2])
        
        rf_Y_2 = RandomForestRegressor(**rf_params)
        rf_W_2 = RandomForestRegressor(**rf_params)
        rf_Y_2.fit(X[idx2], Y[idx2])
        rf_W_2.fit(X[idx2], W[idx2])
        Y_hat[idx1] = rf_Y_2.predict(X[idx1])
        W_hat[idx1] = rf_W_2.predict(X[idx1])
        
        return Y_hat, W_hat
    
    def predict(self, X, return_std=False):
        """Fast prediction."""
        n_test = len(X)
        X = np.ascontiguousarray(X, dtype=np.float64)
        
        if not return_std:
            preds = np.zeros(n_test)
            for i in range(n_test):
                tree_preds = []
                for tree in self.trees:
                    idx = tree.get_leaf_indices(X[i])
                    tau = tree.builder.estimate_tau_ols(
                        np.ascontiguousarray(idx, dtype=np.int32)
                    )
                    tree_preds.append(tau)
                preds[i] = np.mean(tree_preds)
            return preds
        
        print("Computing forest weights...")
        max_leaf_size = self.n // 2
        all_leaf_indices = -np.ones((self.n_trees, n_test, max_leaf_size), dtype=np.int32)
        all_leaf_sizes = np.zeros((self.n_trees, n_test), dtype=np.int32)
        
        for b, tree in enumerate(self.trees):
            for i in range(n_test):
                idx = tree.get_leaf_indices(X[i])
                n_leaf = min(len(idx), max_leaf_size)
                all_leaf_indices[b, i, :n_leaf] = idx[:n_leaf]
                all_leaf_sizes[b, i] = n_leaf
        
        weights = compute_forest_weights_cython(
            all_leaf_indices, all_leaf_sizes, self.n, self.n_trees
        )
        
        tau_hat = np.zeros(n_test)
        for i in range(n_test):
            w = weights[i]
            if np.sum(w) > 0:
                w = w / np.sum(w)
                W_w = self.W_resid * w
                num = np.sum(W_w * self.Y_resid)
                den = np.sum(W_w * self.W_resid)
                if abs(den) > 1e-10:
                    tau_hat[i] = num / den
        
        W_c = self.W_resid - np.mean(self.W_resid)
        Y_c = self.Y_resid - np.mean(self.Y_resid)
        psi = W_c * Y_c
        
        tree_weights = np.zeros((self.n_trees, n_test, self.n))
        for b in range(self.n_trees):
            for i in range(n_test):
                n_leaf = all_leaf_sizes[b, i]
                if n_leaf > 0:
                    for j in range(n_leaf):
                        idx = all_leaf_indices[b, i, j]
                        if 0 <= idx < self.n:
                            tree_weights[b, i, idx] = 1.0 / n_leaf
        
        variances = np.zeros(n_test)
        for i in range(n_test):
            deviations = tree_weights[:, i, :] - weights[i]
            agg_dev = np.sum(deviations, axis=0)
            variances[i] = np.sum(agg_dev**2 * psi**2)
        
        return tau_hat, np.sqrt(variances)
    
    def predict_interval(self, X, alpha=0.05):
        """Predict with CIs."""
        tau, std = self.predict(X, return_std=True)
        z = norm.ppf(1 - alpha/2)
        return tau, tau - z*std, tau + z*std
    
    def effect(self, X):
        """econml API."""
        return self.predict(X, return_std=False)
    
    def effect_interval(self, X, alpha=0.05):
        """econml API."""
        _, lower, upper = self.predict_interval(X, alpha)
        return lower.reshape(-1, 1), upper.reshape(-1, 1)