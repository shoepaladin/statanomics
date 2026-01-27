"""
Numba-compiled core functions for causal trees.
All performance-critical operations are JIT-compiled.
"""

import numpy as np
from numba import jit, prange


@jit(nopython=True, cache=True)
def estimate_tau_ols_numba(Y_resid, W_resid, indices):
    """
    Fast OLS estimation: τ = (W'Y) / (W'W)
    
    Parameters
    ----------
    Y_resid : array (n,)
    W_resid : array (n,)
    indices : array of indices
    
    Returns
    -------
    tau : float
    """
    if len(indices) < 2:
        return 0.0
    
    W = W_resid[indices]
    Y = Y_resid[indices]
    
    denom = np.sum(W * W)
    if denom < 1e-10:
        return 0.0
    
    return np.sum(W * Y) / denom


@jit(nopython=True, cache=True)
def compute_pseudo_outcomes(Y_resid, W_resid, indices, tau_parent):
    """
    Compute gradient pseudo-outcomes: ρ_i = (W_i - W̄)(Y_i - τ × W_i)
    
    Returns
    -------
    pseudo_outcomes : array
    """
    W = W_resid[indices]
    Y = Y_resid[indices]
    
    W_mean = np.mean(W)
    pseudo = (W - W_mean) * (Y - tau_parent * W)
    
    return pseudo


@jit(nopython=True, cache=True)
def compute_split_score_numba(pseudo_outcomes, left_mask):
    """
    GRF splitting objective: n_L × n_R × (mean_L - mean_R)²
    
    Returns
    -------
    score : float
    """
    n_left = np.sum(left_mask)
    n_total = len(left_mask)
    n_right = n_total - n_left
    
    if n_left == 0 or n_right == 0:
        return -np.inf
    
    mean_left = np.mean(pseudo_outcomes[left_mask])
    mean_right = np.mean(pseudo_outcomes[~left_mask])
    
    score = float(n_left) * float(n_right) * (mean_left - mean_right) ** 2
    
    return score


@jit(nopython=True, cache=True)
def find_best_split_numba(X, Y_resid, W_resid, split_idx, tau_parent, 
                          min_leaf, percentiles):
    """
    Find best split across all features (parallelized over features).
    
    Returns
    -------
    best_feature : int
    best_threshold : float
    best_score : float
    """
    n_features = X.shape[1]
    best_score = -np.inf
    best_feature = -1
    best_threshold = 0.0
    
    # Compute pseudo-outcomes once
    pseudo = compute_pseudo_outcomes(Y_resid, W_resid, split_idx, tau_parent)
    
    # Search over features
    for feat in range(n_features):
        feature_vals = X[split_idx, feat]
        
        # Try each percentile threshold
        for pct in percentiles:
            # Compute threshold
            sorted_vals = np.sort(feature_vals)
            idx = int(len(sorted_vals) * pct / 100.0)
            threshold = sorted_vals[min(idx, len(sorted_vals) - 1)]
            
            # Create split mask
            left_mask = feature_vals <= threshold
            n_left = np.sum(left_mask)
            n_right = len(feature_vals) - n_left
            
            # Check minimum leaf size
            if n_left < min_leaf or n_right < min_leaf:
                continue
            
            # Compute score
            score = compute_split_score_numba(pseudo, left_mask)
            
            if score > best_score:
                best_score = score
                best_feature = feat
                best_threshold = threshold
    
    return best_feature, best_threshold, best_score


@jit(nopython=True, parallel=True, cache=True)
def find_best_split_parallel(X, Y_resid, W_resid, split_idx, tau_parent,
                             min_leaf, percentiles):
    """
    Parallel version of split finding (search features in parallel).
    """
    n_features = X.shape[1]
    
    # Pre-allocate arrays for each feature's best split
    feature_scores = np.full(n_features, -np.inf)
    feature_thresholds = np.zeros(n_features)
    
    # Compute pseudo-outcomes once
    pseudo = compute_pseudo_outcomes(Y_resid, W_resid, split_idx, tau_parent)
    
    # Parallel loop over features
    for feat in prange(n_features):
        feature_vals = X[split_idx, feat]
        best_score = -np.inf
        best_thresh = 0.0
        
        for pct in percentiles:
            sorted_vals = np.sort(feature_vals)
            idx = int(len(sorted_vals) * pct / 100.0)
            threshold = sorted_vals[min(idx, len(sorted_vals) - 1)]
            
            left_mask = feature_vals <= threshold
            n_left = np.sum(left_mask)
            n_right = len(feature_vals) - n_left
            
            if n_left < min_leaf or n_right < min_leaf:
                continue
            
            score = compute_split_score_numba(pseudo, left_mask)
            
            if score > best_score:
                best_score = score
                best_thresh = threshold
        
        feature_scores[feat] = best_score
        feature_thresholds[feat] = best_thresh
    
    # Find overall best
    best_feature = np.argmax(feature_scores)
    best_score = feature_scores[best_feature]
    best_threshold = feature_thresholds[best_feature]
    
    if best_score == -np.inf:
        return -1, 0.0, -np.inf
    
    return best_feature, best_threshold, best_score


@jit(nopython=True, parallel=True, cache=True)
def compute_tree_weights_numba(leaf_indices, leaf_sizes, n_train):
    """
    Compute weights for a single tree.
    
    Parameters
    ----------
    leaf_indices : array (n_test, max_leaf_size)
        Packed leaf indices (-1 for padding)
    leaf_sizes : array (n_test,)
        Actual size of each leaf
    n_train : int
        Number of training samples
    
    Returns
    -------
    weights : array (n_test, n_train)
    """
    n_test = leaf_indices.shape[0]
    weights = np.zeros((n_test, n_train))
    
    for i in prange(n_test):
        n_leaf = leaf_sizes[i]
        if n_leaf > 0:
            for j in range(n_leaf):
                idx = leaf_indices[i, j]
                if idx >= 0 and idx < n_train:
                    weights[i, idx] = 1.0 / n_leaf
    
    return weights


@jit(nopython=True, cache=True)
def weighted_ols_prediction(Y_resid, W_resid, weights):
    """
    Compute weighted OLS prediction for a single test point.
    
    τ̂ = (W'WY) / (W'WW) where W is weighted
    
    Returns
    -------
    tau : float
    """
    if np.sum(weights) < 1e-10:
        return 0.0
    
    # Normalize weights
    w = weights / np.sum(weights)
    
    # Weighted regression
    W_weighted = W_resid * w
    numerator = np.sum(W_weighted * Y_resid)
    denominator = np.sum(W_weighted * W_resid)
    
    if abs(denominator) < 1e-10:
        return 0.0
    
    return numerator / denominator


@jit(nopython=True, cache=True)
def compute_ij_variance_numba(tree_weights, avg_weights, psi):
    """
    Compute Infinitesimal Jackknife variance.
    
    V(x) = Σ_i [Σ_b (α_ib(x) - ᾱ_i(x))]² × ψ_i²
    
    Parameters
    ----------
    tree_weights : array (n_trees, n_test, n_train)
    avg_weights : array (n_test, n_train)
    psi : array (n_train,)
        Pseudo-outcomes
    
    Returns
    -------
    variances : array (n_test,)
    """
    n_test = avg_weights.shape[0]
    n_train = avg_weights.shape[1]
    n_trees = tree_weights.shape[0]
    
    variances = np.zeros(n_test)
    
    for i in range(n_test):
        # Compute deviation sum for each training point
        agg_dev = np.zeros(n_train)
        for b in range(n_trees):
            for j in range(n_train):
                agg_dev[j] += tree_weights[b, i, j] - avg_weights[i, j]
        
        # IJ variance
        var_i = 0.0
        for j in range(n_train):
            var_i += agg_dev[j]**2 * psi[j]**2
        
        variances[i] = var_i
    
    return variances
