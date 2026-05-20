"""
Numba-compiled core functions for causal trees.
All performance-critical operations are JIT-compiled.
"""

import numpy as np
from numba import jit, prange


@jit(nopython=True, cache=True)
def estimate_tau_ols_numba(Y_resid, W_resid, indices):
    """Fast OLS: tau = (W'Y) / (W'W)."""
    if len(indices) < 2:
        return 0.0
    W = W_resid[indices]
    Y = Y_resid[indices]
    denom = np.dot(W, W)
    if denom < 1e-10:
        return 0.0
    return np.dot(W, Y) / denom


@jit(nopython=True, cache=True)
def compute_pseudo_outcomes(Y_resid, W_resid, indices, tau_parent):
    """Gradient pseudo-outcomes: rho_i = (W_i - W_bar)(Y_i - tau * W_i)."""
    W = W_resid[indices]
    Y = Y_resid[indices]
    W_mean = np.mean(W)
    return (W - W_mean) * (Y - tau_parent * W)


@jit(nopython=True, cache=True)
def compute_split_score_numba(pseudo_outcomes, left_mask):
    """GRF objective: n_L * n_R * (mean_L - mean_R)^2."""
    n_left = np.sum(left_mask)
    n_total = len(left_mask)
    n_right = n_total - n_left
    if n_left == 0 or n_right == 0:
        return -np.inf
    mean_left = np.mean(pseudo_outcomes[left_mask])
    mean_right = np.mean(pseudo_outcomes[~left_mask])
    return float(n_left) * float(n_right) * (mean_left - mean_right) ** 2


@jit(nopython=True, cache=True)
def find_best_split_numba(X, Y_resid, W_resid, split_idx, tau_parent,
                          min_leaf, percentiles, feature_indices):
    """
    Find best split over a subset of features (sequential).

    Parameters
    ----------
    feature_indices : int array
        Indices of features to search (mtry subset).
    percentiles : float array
        Candidate split quantiles (e.g. [5, 10, ..., 95]).
    """
    n_cand = len(feature_indices)
    best_score = -np.inf
    best_feature = -1
    best_threshold = 0.0

    pseudo = compute_pseudo_outcomes(Y_resid, W_resid, split_idx, tau_parent)

    for fi in range(n_cand):
        feat = feature_indices[fi]
        feature_vals = X[split_idx, feat]

        # FIX: sort once per feature, not once per percentile candidate
        sorted_vals = np.sort(feature_vals)
        n_vals = len(sorted_vals)

        for p_idx in range(len(percentiles)):
            pct = percentiles[p_idx]
            idx = int(n_vals * pct / 100.0)
            threshold = sorted_vals[min(idx, n_vals - 1)]

            left_mask = feature_vals <= threshold
            n_left = np.sum(left_mask)
            n_right = n_vals - n_left

            if n_left < min_leaf or n_right < min_leaf:
                continue

            score = compute_split_score_numba(pseudo, left_mask)
            if score > best_score:
                best_score = score
                best_feature = feat
                best_threshold = threshold

    return best_feature, best_threshold, best_score


@jit(nopython=True, parallel=True, cache=True)
def find_best_split_parallel(X, Y_resid, W_resid, split_idx, tau_parent,
                              min_leaf, percentiles, feature_indices):
    """Parallel version: each feature searched on a separate thread."""
    n_cand = len(feature_indices)
    feature_scores = np.full(n_cand, -np.inf)
    feature_thresholds = np.zeros(n_cand)

    pseudo = compute_pseudo_outcomes(Y_resid, W_resid, split_idx, tau_parent)

    for fi in prange(n_cand):
        feat = feature_indices[fi]
        feature_vals = X[split_idx, feat]

        # FIX: sort once per feature
        sorted_vals = np.sort(feature_vals)
        n_vals = len(sorted_vals)

        best_score = -np.inf
        best_thresh = 0.0

        for p_idx in range(len(percentiles)):
            pct = percentiles[p_idx]
            idx = int(n_vals * pct / 100.0)
            threshold = sorted_vals[min(idx, n_vals - 1)]

            left_mask = feature_vals <= threshold
            n_left = np.sum(left_mask)
            n_right = n_vals - n_left

            if n_left < min_leaf or n_right < min_leaf:
                continue

            score = compute_split_score_numba(pseudo, left_mask)
            if score > best_score:
                best_score = score
                best_thresh = threshold

        feature_scores[fi] = best_score
        feature_thresholds[fi] = best_thresh

    best_fi = np.argmax(feature_scores)
    best_score = feature_scores[best_fi]
    best_threshold = feature_thresholds[best_fi]

    if best_score == -np.inf:
        return -1, 0.0, -np.inf

    return feature_indices[best_fi], best_threshold, best_score


@jit(nopython=True, parallel=True, cache=True)
def traverse_tree_batch(X_test, features, thresholds, left_children,
                         right_children, est_starts, est_sizes,
                         est_indices_flat, max_leaf_size):
    """
    JIT-compiled batch tree traversal.

    Traverse all test points through the tree in parallel, returning packed
    leaf membership arrays instead of looping in Python.

    Parameters
    ----------
    features : int array (n_nodes,)  -1 marks a leaf node
    thresholds : float array (n_nodes,)
    left_children, right_children : int array (n_nodes,)
    est_starts : int array (n_nodes,)  offset into est_indices_flat
    est_sizes : int array (n_nodes,)
    est_indices_flat : int array  all estimation indices concatenated
    max_leaf_size : int  column width of result array

    Returns
    -------
    result_indices : int array (n_test, max_leaf_size)  -1 padded
    result_sizes : int array (n_test,)
    """
    n_test = X_test.shape[0]
    result_indices = np.full((n_test, max_leaf_size), -1, dtype=np.int32)
    result_sizes = np.zeros(n_test, dtype=np.int32)

    for i in prange(n_test):
        node = np.int32(0)
        while features[node] != -1:
            if X_test[i, features[node]] <= thresholds[node]:
                node = left_children[node]
            else:
                node = right_children[node]
        size = est_sizes[node]
        if size > max_leaf_size:
            size = max_leaf_size
        start = est_starts[node]
        for j in range(size):
            result_indices[i, j] = est_indices_flat[start + j]
        result_sizes[i] = size

    return result_indices, result_sizes


@jit(nopython=True, parallel=True, cache=True)
def batch_predict_from_leaves(Y_resid, W_resid, leaf_indices, leaf_sizes):
    """
    Compute per-test-point tau estimates from packed leaf membership arrays.

    Parameters
    ----------
    leaf_indices : int array (n_test, max_leaf_size)
    leaf_sizes : int array (n_test,)

    Returns
    -------
    preds : float array (n_test,)
    """
    n_test = leaf_indices.shape[0]
    preds = np.zeros(n_test)

    for i in prange(n_test):
        size = leaf_sizes[i]
        if size < 2:
            continue
        idx = leaf_indices[i, :size]
        W = W_resid[idx]
        Y = Y_resid[idx]
        denom = np.dot(W, W)
        if denom < 1e-10:
            continue
        preds[i] = np.dot(W, Y) / denom

    return preds


@jit(nopython=True, parallel=True, cache=True)
def compute_tree_weights_numba(leaf_indices, leaf_sizes, n_train):
    """Compute (n_test, n_train) weight matrix for a single tree."""
    n_test = leaf_indices.shape[0]
    weights = np.zeros((n_test, n_train))
    for i in prange(n_test):
        n_leaf = leaf_sizes[i]
        if n_leaf > 0:
            w = 1.0 / n_leaf
            for j in range(n_leaf):
                idx = leaf_indices[i, j]
                if 0 <= idx < n_train:
                    weights[i, idx] = w
    return weights


def compute_ij_variance(tree_preds, subsample_flags, subsample_size, n_train):
    """
    Proper Infinitesimal Jackknife variance for a subsampling forest.

    The previous implementation summed deviations from their own mean
    (algebraically zero), giving V=0 always.  This replaces it with the
    correct formula from Wager, Hastie & Efron (2014) / Athey & Wager (2019):

        cov_bj(x) = (1/B) * sum_b [ T_b(x) * I_bj ] - T_bar(x) * p_j
        V_IJ(x)   = (n / s) * sum_j  cov_bj(x)^2

    where I_bj = 1 if training obs j was in subsample b, p_j = empirical
    inclusion rate, n = total training size, s = subsample size.

    Parameters
    ----------
    tree_preds : float array (n_trees, n_test)
        Per-tree predictions.
    subsample_flags : bool array (n_trees, n_train)
        Whether each training obs was in each tree's subsample.
    subsample_size : int
        Number of training obs per subsample.
    n_train : int

    Returns
    -------
    variances : float array (n_test,)
    """
    n_trees, n_test = tree_preds.shape
    if n_trees < 2:
        return np.zeros(n_test)

    flags = subsample_flags.astype(np.float64)  # (n_trees, n_train)

    # (n_test, n_train):  sum_b T_b(x_i) * I_bj for every (i, j) pair
    T_sum = tree_preds.T @ flags            # (n_test, n_trees) @ (n_trees, n_train)
    T_mean_ij = T_sum / n_trees             # average over trees

    T_bar = tree_preds.mean(axis=0)        # (n_test,)
    p_j = flags.mean(axis=0)              # (n_train,) empirical inclusion rate

    # cov_bj: (n_test, n_train)
    cov_bj = T_mean_ij - T_bar[:, np.newaxis] * p_j[np.newaxis, :]

    # V_IJ(x) = (n / s) * sum_j cov_bj(x)^2
    scale = n_train / subsample_size
    variances = scale * np.sum(cov_bj ** 2, axis=1)
    return variances


@jit(nopython=True, cache=True)
def compute_variance_from_tree_preds(tree_preds):
    """
    Fallback: unbiased variance-of-the-mean from per-tree predictions.
    Used when subsample flags are not available.
    V(x) = Var_b[T_b(x)] / B  (conservative — prefers compute_ij_variance).

    Parameters
    ----------
    tree_preds : float array (n_trees, n_test)

    Returns
    -------
    variances : float array (n_test,)
    """
    n_trees = tree_preds.shape[0]
    n_test = tree_preds.shape[1]
    variances = np.zeros(n_test)

    if n_trees < 2:
        return variances

    for i in range(n_test):
        mean_val = 0.0
        for b in range(n_trees):
            mean_val += tree_preds[b, i]
        mean_val /= n_trees

        sq_sum = 0.0
        for b in range(n_trees):
            diff = tree_preds[b, i] - mean_val
            sq_sum += diff * diff

        variances[i] = sq_sum / (n_trees * (n_trees - 1))

    return variances
