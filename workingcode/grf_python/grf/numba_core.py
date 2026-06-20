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


@jit(nopython=True, cache=True)
def accumulate_omega_tree(Omega, leaf_indices, leaf_sizes, W_resid, n_trees):
    """
    Accumulate one tree's contribution to the forest OLS weight matrix Omega.

    For each test point i, adds w_{j,b}(i) / B to Omega[j, i] for every
    estimation-sample obs j in the leaf:

        w_{j,b}(i) = W_resid[j] / sum_{k in leaf} W_resid[k]^2

    Call once per tree; after B trees Omega[j,i] = (1/B) sum_b w_{j,b}(i).

    Parameters
    ----------
    Omega : float64 array (n_train, n_test)   accumulated in-place
    leaf_indices : int32 array (n_test, max_leaf)
    leaf_sizes   : int32 array (n_test,)
    W_resid      : float64 array (n_train,)
    n_trees      : int  (B — the forest size, used to normalise)
    """
    n_test = leaf_indices.shape[0]
    inv_B = 1.0 / n_trees
    for i in range(n_test):
        sz = leaf_sizes[i]
        if sz < 2:
            continue
        denom = 0.0
        for k in range(sz):
            j = leaf_indices[i, k]
            denom += W_resid[j] * W_resid[j]
        if denom < 1e-10:
            continue
        scale = inv_B / denom
        for k in range(sz):
            j = leaf_indices[i, k]
            Omega[j, i] += W_resid[j] * scale


def compute_delta_variance(Omega, Y_resid):
    """
    Delta method variance: V(x) = sigma^2 * sum_j Omega_j(x)^2.

    sigma^2 is estimated from the training residuals mean(Y_resid^2).

    Parameters
    ----------
    Omega   : float64 array (n_train, n_test)
    Y_resid : float64 array (n_train,)

    Returns
    -------
    variances : float64 array (n_test,)
    """
    sigma2 = np.mean(Y_resid ** 2)
    return sigma2 * np.sum(Omega ** 2, axis=0)


def compute_ij_variance(tree_preds, subsample_flags, subsample_size, n_train,
                        bias_correction=True):
    """
    Infinitesimal Jackknife variance for a subsampling forest, with the
    finite-B Monte-Carlo bias correction (IJ-U).

    Raw IJ covariance (Wager, Hastie & Efron 2014; Athey & Wager 2019):

        cov_j(x) = (1/B) * sum_b [ T_b(x) * I_bj ] - T_bar(x) * p_j
                 = (1/B) * sum_b (I_bj - p_j)(T_b(x) - T_bar(x))
        V_raw(x) = (n / s) * sum_j cov_j(x)^2

    With a *finite* number of trees B, V_raw is the sum of n squared sample
    covariances, each estimated from only B trees.  Each cov_j carries
    Monte-Carlo sampling noise of order 1/B, so E[cov_j^2] = Cov_true^2 +
    Var(cov_j), and summing n of these inflates the variance estimate by a
    positive O(n/B) bias.  This is exactly why the *uncorrected* estimator
    needed an impractically huge B to calibrate (the "0% FPR / B>>9000"
    symptom): it was not a theoretical limitation of IJ, just a missing
    correction term.

    The Wager-Hastie-Efron bias correction subtracts an estimate of that
    Monte-Carlo noise.  Under independence of inclusion and prediction
    (the relevant regime under the null),

        Var(cov_j) ~= (1/B) * p_j (1 - p_j) * sigma_T^2(x),

    where sigma_T^2(x) = (1/B) sum_b (T_b(x) - T_bar(x))^2 is the empirical
    variance of the tree predictions at x.  Hence

        V_IJ-U(x) = (n/s) * [ sum_j cov_j(x)^2
                              - (sigma_T^2(x) / B) * sum_j p_j (1 - p_j) ].

    Sanity check vs. the WHE bootstrap form: with sampling-with-replacement
    (scale n/s -> 1, Var(N_j) -> 1) this reduces to V_raw - (n/B) sigma_T^2,
    i.e. their (n / B^2) sum_b (T_b - T_bar)^2 correction.

    Parameters
    ----------
    tree_preds : float array (n_trees, n_test)
        Per-tree predictions.
    subsample_flags : bool array (n_trees, n_train)
        Whether each training obs was in each tree's subsample.
    subsample_size : int
        Number of training obs per subsample (s).
    n_train : int
        Total training size (n).
    bias_correction : bool
        If True (default) return the bias-corrected IJ-U estimate; if False
        return the raw (uncorrected, positively biased) IJ estimate.

    Returns
    -------
    variances : float array (n_test,)
        Non-negative; the correction is floored at 0 per test point.
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

    # cov_j: (n_test, n_train)
    cov_j = T_mean_ij - T_bar[:, np.newaxis] * p_j[np.newaxis, :]

    scale = n_train / subsample_size
    raw = np.sum(cov_j ** 2, axis=1)        # (n_test,)

    if not bias_correction:
        return scale * raw

    # Monte-Carlo bias term per test point.
    sigma_T2 = tree_preds.var(axis=0)       # (n_test,)  population var over trees
    sum_pq = np.sum(p_j * (1.0 - p_j))      # scalar; ~ s(n-s)/n for p_j ~ s/n
    bias = (sigma_T2 / n_trees) * sum_pq    # (n_test,)

    variances = scale * np.maximum(raw - bias, 0.0)
    return variances


def compute_blb_variance(tree_preds, group_size):
    """
    Bootstrap-of-little-bags (BLB) variance for a subsampling forest.

    This is the variance estimator actually used by the R `grf` package and
    `econml.grf` (Athey, Tibshirani & Wager 2019, sec. 4), and the reason
    those packages get calibrated confidence intervals with only B = 100-400
    trees.  The raw infinitesimal jackknife needs an impractically large B
    because every one of its n squared covariances carries O(1/B) Monte-Carlo
    noise; BLB instead estimates the variance from the *between-group*
    spread of bag means, whose Monte-Carlo error is governed by the number of
    groups G = B / L rather than by n.

    Pre-condition (enforced by the forest's grouped subsampling): the trees
    are ordered so that every consecutive block of `group_size` trees forms a
    "little bag" that shares one common half-sample of the training data.
    Each tree subsamples from that shared half.  Consequently:

      * between-bag variance of the bag means reflects the half-sample
        (data-resampling) variability == the true sampling variance of tau(x);
      * within-bag variance reflects only the extra Monte-Carlo noise from
        sub-subsampling within a fixed half, which we subtract off.

        bag_mean_g(x) = mean_{b in g} T_b(x)
        V_between(x)  = mean_g (bag_mean_g(x) - T_bar(x))^2
        V_within(x)   = mean_g mean_{b in g} (T_b(x) - bag_mean_g(x))^2
        V_BLB(x)      = V_between(x) - V_within(x) / (L - 1)

    The subtraction de-biases V_between for the within-bag Monte-Carlo term
    (E[V_between] = V_true + V_within/(L*(L-1)) ... cancelled exactly here).
    The naive estimate can be negative when the true variance is tiny; rather
    than hard-clipping to 0 (which biases CIs and inflates the null FPR), we
    apply the same objective-Bayes positive-part correction econml uses.

    Parameters
    ----------
    tree_preds : float array (n_trees, n_test)
        Per-tree predictions, grouped in contiguous blocks of `group_size`.
    group_size : int
        Number of trees per little bag (L >= 2).

    Returns
    -------
    variances : float array (n_test,)  non-negative.
    """
    from scipy.special import erfc

    B, n_test = tree_preds.shape
    L = group_size
    G = B // L
    if G < 2 or L < 2:
        # Fall back to the plain variance of the mean (conservative).
        return compute_variance_from_tree_preds(np.ascontiguousarray(tree_preds))

    tp = tree_preds[:G * L].reshape(G, L, n_test)
    bag_means = tp.mean(axis=1)                 # (G, n_test)
    overall = bag_means.mean(axis=0)            # (n_test,)

    v_between = np.mean((bag_means - overall) ** 2, axis=0)            # (n_test,)
    v_within = np.mean(np.mean((tp - bag_means[:, None, :]) ** 2,
                               axis=1), axis=0)                        # (n_test,)
    correction = v_within / (L - 1)
    naive = v_between - correction

    # Objective-Bayes debiasing of the positive-part (matches econml.grf):
    # smoothly maps the (possibly negative) naive estimate to a positive value
    # instead of truncating at 0, which keeps CI coverage calibrated.
    se = np.maximum(v_between, correction) * np.sqrt(2.0 / G)
    zstat = naive / np.clip(se, 1e-10, np.inf)
    numerator = np.exp(-(zstat ** 2) / 2.0) / np.sqrt(2.0 * np.pi)
    denominator = 0.5 * erfc(-zstat / np.sqrt(2.0))
    variances = naive + se * numerator / np.clip(denominator, 1e-10, np.inf)
    return np.maximum(variances, 0.0)


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
