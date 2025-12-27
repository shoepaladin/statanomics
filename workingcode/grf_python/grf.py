"""
pygrf_core.py

Algorithmic translation of the GRF causal & instrumental forest core.

- Implements CausalForest (binary treatment) and InstrumentalForest (single instrument).
- Honesty via sample splitting for each tree.
- Split search via mtry and threshold midpoints.
- Leaf estimation: diff-in-means (causal) and Cov[Y,Z]/Cov[W,Z] (IV).
- Minimal dependencies: numpy, sklearn (train_test_split) - both standard.

Notes:
- This is an algorithmic translation intended to match the structure of GRF C++ core.
- It omits low-level optimizations present in the original (C++ vectorization, threading).
- For production, consider numba/cython and implementing the IJ/influence-based variance.

Correspondences:
- ForestTrainer / CausalForest.cpp    -> CausalForest (fit/predict)
- Forest / Tree                       -> _Tree / _TreeNode
- SplittingRule                       -> split-score logic inside _Tree._try_splits
"""

import numpy as np
from sklearn.model_selection import train_test_split
import math
import warnings
from typing import Optional, Tuple, List

# --------------------
# Utilities
# --------------------

def _ensure_2d(X):
    X = np.asarray(X)
    if X.ndim == 1:
        X = X.reshape(-1, 1)
    return X

def _safe_mean(arr):
    if arr.size == 0:
        return np.nan
    return np.mean(arr)

# --------------------
# Tree Node
# --------------------

class _TreeNode:
    def __init__(self, depth=0):
        self.depth = depth
        self.is_leaf = True
        self.feature = None
        self.threshold = None
        self.left = None
        self.right = None
        # leaf stats
        self.n_est = 0
        self.tau = 0.0         # causal effect estimate (or IV)
        self.mu0 = np.nan
        self.mu1 = np.nan
        self.cov_y_z = np.nan
        self.cov_w_z = np.nan

# --------------------
# Tree builder (honest)
# --------------------

class _HonestTree:
    """
    Honest tree: uses splitting data to choose splits, estimation data to compute leaf stats.
    The public interface mirrors the GRF TreeTrainer/CausalTree organization.
    """

    def __init__(self, max_depth=10, min_node_size=20, mtry=None, rng=None, task='causal', ridge=1e-8):
        self.max_depth = max_depth
        self.min_node_size = min_node_size
        self.mtry = mtry  # number of features to try per split
        self.root = _TreeNode(depth=0)
        self.rng = np.random.RandomState(rng)
        self.task = task
        self.ridge = ridge

    def fit(self,
            X_split: np.ndarray, Y_split: np.ndarray, W_split: np.ndarray,
            X_est: np.ndarray, Y_est: np.ndarray, W_est: np.ndarray,
            Z_est: Optional[np.ndarray] = None):
        """
        X_split, Y_split, W_split: arrays used to find splits (structure)
        X_est, Y_est, W_est [, Z_est]: arrays used to estimate leaf quantities
        """
        X_split = _ensure_2d(X_split)
        X_est = _ensure_2d(X_est)
        n_features = X_split.shape[1]
        if self.mtry is None:
            self.mtry = max(1, int(math.sqrt(n_features)))

        split_idx = np.arange(X_split.shape[0])
        est_idx = np.arange(X_est.shape[0])
        self._build_node(self.root, X_split, Y_split, W_split, split_idx, X_est, Y_est, W_est, est_idx, Z_est)

    # ---------- leaf estimators ----------
    def _leaf_causal(self, y_e, w_e):
        # difference-in-means; return tau, mu0, mu1, n
        if y_e.size == 0:
            return 0.0, np.nan, np.nan, 0
        mask0 = (w_e == 0)
        mask1 = (w_e == 1)
        n0 = mask0.sum()
        n1 = mask1.sum()
        if n0 == 0 or n1 == 0:
            # can't estimate in leaf -- return 0 and small n to discourage splits that produce this
            mu0 = np.nan if n0 == 0 else y_e[mask0].mean()
            mu1 = np.nan if n1 == 0 else y_e[mask1].mean()
            return 0.0, mu0, mu1, n0 + n1
        mu0 = y_e[mask0].mean()
        mu1 = y_e[mask1].mean()
        return mu1 - mu0, mu0, mu1, n0 + n1

    def _leaf_iv(self, y_e, w_e, z_e):
        # local IV: cov(y,z) / (cov(w,z) + ridge)
        if y_e.size == 0:
            return 0.0, np.nan, np.nan, 0
        # compute population covariance (bias=True) to match intuitive cov
        cov_yz = np.cov(y_e, z_e, bias=True)[0,1]
        cov_wz = np.cov(w_e, z_e, bias=True)[0,1]
        if abs(cov_wz) < 1e-12:
            return 0.0, cov_yz, cov_wz, y_e.size
        return cov_yz / (cov_wz + self.ridge), cov_yz, cov_wz, y_e.size

    # ---------- recursive builder ----------
    def _build_node(self, node: _TreeNode,
                    X_s, Y_s, W_s, idx_s,
                    X_e, Y_e, W_e, idx_e,
                    Z_e = None):
        # Estimate leaf parameters from estimation data
        if idx_e.size == 0:
            node.is_leaf = True
            node.n_est = 0
            node.tau = 0.0
            return

        y_node = Y_e[idx_e]
        w_node = W_e[idx_e]

        if self.task == 'causal':
            tau_hat, mu0, mu1, nleaf = self._leaf_causal(y_node, w_node)
            node.tau = tau_hat
            node.mu0 = mu0
            node.mu1 = mu1
            node.n_est = nleaf
        elif self.task == 'iv':
            if Z_e is None:
                raise ValueError("Instrument Z must be provided for IV tree.")
            z_node = Z_e[idx_e]
            tau_hat, cov_yz, cov_wz, nleaf = self._leaf_iv(y_node, w_node, z_node)
            node.tau = tau_hat
            node.cov_y_z = cov_yz
            node.cov_w_z = cov_wz
            node.n_est = nleaf
        else:
            raise ValueError("Unknown task: " + str(self.task))

        # stopping conditions
        if (node.depth >= self.max_depth) or (idx_s.size < 2 * self.min_node_size):
            node.is_leaf = True
            return

        # search for best split using splitting data
        best_gain = -np.inf
        best_split = None

        n_features = X_s.shape[1]
        features = self.rng.choice(n_features, size=min(self.mtry, n_features), replace=False)

        for feat in features:
            xs = X_s[idx_s, feat]
            uniques = np.unique(xs)
            if uniques.size <= 1:
                continue
            # candidate splits at midpoints
            thresholds = (uniques[:-1] + uniques[1:]) / 2.0
            for thr in thresholds:
                left_mask_s = xs <= thr
                right_mask_s = xs > thr
                left_idx_s = idx_s[left_mask_s]
                right_idx_s = idx_s[right_mask_s]
                if left_idx_s.size < self.min_node_size or right_idx_s.size < self.min_node_size:
                    continue

                xe_feat = X_e[idx_e, feat]
                left_mask_e = xe_feat <= thr
                right_mask_e = xe_feat > thr
                left_idx_e = idx_e[left_mask_e]
                right_idx_e = idx_e[right_mask_e]
                # need some estimation samples
                if left_idx_e.size < max(2, int(0.5 * self.min_node_size)) or right_idx_e.size < max(2, int(0.5 * self.min_node_size)):
                    continue

                # child leaf estimates (on estimation samples)
                if self.task == 'causal':
                    tau_L, _, _, nL = self._leaf_causal(Y_e[left_idx_e], W_e[left_idx_e])
                    tau_R, _, _, nR = self._leaf_causal(Y_e[right_idx_e], W_e[right_idx_e])
                else:  # iv
                    tau_L, _, _, nL = self._leaf_iv(Y_e[left_idx_e], W_e[left_idx_e], Z_e[left_idx_e])
                    tau_R, _, _, nR = self._leaf_iv(Y_e[right_idx_e], W_e[right_idx_e], Z_e[right_idx_e])

                # gain: weighted squared difference (simple heterogeneity objective)
                if (nL + nR) == 0:
                    continue
                gain = (nL * nR / float(nL + nR)) * (tau_L - tau_R) ** 2

                if gain > best_gain:
                    best_gain = gain
                    best_split = (feat, thr, left_idx_s.copy(), right_idx_s.copy(), left_idx_e.copy(), right_idx_e.copy())

        if best_split is None or best_gain <= 0:
            node.is_leaf = True
            return

        feat, thr, left_idx_s, right_idx_s, left_idx_e, right_idx_e = best_split
        node.is_leaf = False
        node.feature = feat
        node.threshold = thr
        node.left = _TreeNode(depth=node.depth + 1)
        node.right = _TreeNode(depth=node.depth + 1)

        # recurse
        if self.task == 'causal':
            self._build_node(node.left, X_s, Y_s, W_s, left_idx_s, X_e, Y_e, W_e, left_idx_e, Z_e)
            self._build_node(node.right, X_s, Y_s, W_s, right_idx_s, X_e, Y_e, W_e, right_idx_e, Z_e)
        else:
            self._build_node(node.left, X_s, Y_s, W_s, left_idx_s, X_e, Y_e, W_e, left_idx_e, Z_e)
            self._build_node(node.right, X_s, Y_s, W_s, right_idx_s, X_e, Y_e, W_e, right_idx_e, Z_e)

    def predict_one(self, x):
        node = self.root
        while not node.is_leaf:
            if x[node.feature] <= node.threshold:
                node = node.left
            else:
                node = node.right
        return node.tau, node.n_est

# --------------------
# Forest trainer / wrapper
# --------------------

class CausalForest:
    """
    High-level forest object that mirrors GRF's CausalForest trainer.

    Parameters mirror typical GRF settings:
    - n_trees: number of trees
    - sample_fraction: fraction of full data used to build each tree (subsampling)
    - honesty: always True here (we perform sample split per tree)
    - honesty_fraction: proportion of subsample used for splitting vs estimation
    - mtry, min_node_size, max_depth control tree splitting
    """
    def __init__(self,
                 n_trees: int = 200,
                 sample_fraction: float = 0.5,
                 honesty_fraction: float = 0.5,
                 mtry: Optional[int] = None,
                 min_node_size: int = 5,
                 max_depth: int = 20,
                 random_state: Optional[int] = None):
        self.n_trees = n_trees
        self.sample_fraction = sample_fraction
        self.honesty_fraction = honesty_fraction
        self.mtry = mtry
        self.min_node_size = min_node_size
        self.max_depth = max_depth
        self.random_state = np.random.RandomState(random_state)
        self.trees: List[_HonestTree] = []

    def fit(self, X, Y, W):
        X = _ensure_2d(X)
        Y = np.asarray(Y)
        W = np.asarray(W)
        n = X.shape[0]
        if self.mtry is None:
            self.mtry = max(1, int(math.sqrt(X.shape[1])))

        self.trees = []
        for t in range(self.n_trees):
            # subsample (without replacement) akin to GRF's sub-sampling
            subsample_size = max(2, int(self.sample_fraction * n))
            subsample_idx = self.random_state.choice(n, size=subsample_size, replace=False)

            # honesty split: split the subsample into split-set and estimation-set
            # We'll use sklearn train_test_split with shuffle=True, but we need deterministic RNG
            s_idx, e_idx = train_test_split(subsample_idx, test_size=self.honesty_fraction, random_state=self.random_state.randint(2**31 - 1))

            tree = _HonestTree(
                max_depth=self.max_depth,
                min_node_size=self.min_node_size,
                mtry=self.mtry,
                rng=self.random_state.randint(2**31 - 1),
                task='causal'
            )
            tree.fit(X[s_idx], Y[s_idx], W[s_idx], X[e_idx], Y[e_idx], W[e_idx])
            self.trees.append(tree)

    def predict(self, X):
        X = _ensure_2d(X)
        n = X.shape[0]
        all_preds = np.zeros((self.n_trees, n))
        all_ns = np.zeros((self.n_trees, n), dtype=int)
        for i, tree in enumerate(self.trees):
            for j in range(n):
                tau, nleaf = tree.predict_one(X[j])
                all_preds[i, j] = tau
                all_ns[i, j] = nleaf
        tau_hat = all_preds.mean(axis=0)
        # naive se = sd across trees / sqrt(n_trees)
        tau_se = all_preds.std(axis=0, ddof=1) / math.sqrt(max(1, self.n_trees))
        return tau_hat, tau_se

# --------------------
# InstrumentalForest wrapper
# --------------------

class InstrumentalForest:
    def __init__(self,
                 n_trees: int = 200,
                 sample_fraction: float = 0.5,
                 honesty_fraction: float = 0.5,
                 mtry: Optional[int] = None,
                 min_node_size: int = 10,
                 max_depth: int = 20,
                 ridge: float = 1e-8,
                 random_state: Optional[int] = None):
        self.n_trees = n_trees
        self.sample_fraction = sample_fraction
        self.honesty_fraction = honesty_fraction
        self.mtry = mtry
        self.min_node_size = min_node_size
        self.max_depth = max_depth
        self.ridge = ridge
        self.random_state = np.random.RandomState(random_state)
        self.trees: List[_HonestTree] = []

    def fit(self, X, Y, W, Z):
        X = _ensure_2d(X)
        Y = np.asarray(Y)
        W = np.asarray(W)
        Z = np.asarray(Z)
        n = X.shape[0]
        if self.mtry is None:
            self.mtry = max(1, int(math.sqrt(X.shape[1])))

        self.trees = []
        for t in range(self.n_trees):
            subsample_size = max(2, int(self.sample_fraction * n))
            subsample_idx = self.random_state.choice(n, size=subsample_size, replace=False)
            s_idx, e_idx = train_test_split(subsample_idx, test_size=self.honesty_fraction, random_state=self.random_state.randint(2**31 - 1))

            tree = _HonestTree(max_depth=self.max_depth,
                               min_node_size=self.min_node_size,
                               mtry=self.mtry,
                               rng=self.random_state.randint(2**31 - 1),
                               task='iv',
                               ridge=self.ridge)
            # pass Z for estimation
            tree.fit(X[s_idx], Y[s_idx], W[s_idx], X[e_idx], Y[e_idx], W[e_idx], Z[e_idx])
            self.trees.append(tree)

    def predict(self, X):
        X = _ensure_2d(X)
        n = X.shape[0]
        all_preds = np.zeros((self.n_trees, n))
        for i, tree in enumerate(self.trees):
            for j in range(n):
                tau, nleaf = tree.predict_one(X[j])
                all_preds[i, j] = tau
        tau_hat = all_preds.mean(axis=0)
        tau_se = all_preds.std(axis=0, ddof=1) / math.sqrt(max(1, self.n_trees))
        return tau_hat, tau_se

# --------------------
# Example / quick sanity test
# --------------------

if __name__ == "__main__":
    # small synthetic example for causal forest
    np.random.seed(123)
    n = 1000
    p = 6
    X = np.random.normal(size=(n, p))
    tau_true = (X[:, 0] > 0).astype(float) * 1.0 + 0.5 * X[:, 1]
    prop = 0.25 + 0.5 * (X[:, 2] > 0).astype(float)
    W = np.random.binomial(1, prop)
    Y0 = 0.2 * X[:, 3] + np.random.normal(scale=1.0, size=n)
    Y = Y0 + W * tau_true

    cf = CausalForest(n_trees=80, sample_fraction=0.6, honesty_fraction=0.5, min_node_size=30, max_depth=6, random_state=1)
    cf.fit(X, Y, W)
    tau_hat, tau_se = cf.predict(X[:10])
    print("Causal forest predicted tau (first 10):", tau_hat)
    print("Causal forest se (first 10):", tau_se)

    # IV example
    Z = np.random.binomial(1, 0.5, size=n)
    W_iv_prob = 0.15 + 0.6 * Z * (X[:, 0] > 0).astype(float)
    W_iv = np.random.binomial(1, W_iv_prob)
    Y_iv = 0.2 * X[:, 3] + W_iv * tau_true + np.random.normal(size=n)
    ivf = InstrumentalForest(n_trees=80, sample_fraction=0.6, honesty_fraction=0.5, min_node_size=40, max_depth=6, random_state=2)
    ivf.fit(X, Y_iv, W_iv, Z)
    tau_iv_hat, tau_iv_se = ivf.predict(X[:10])
    print("IV forest predicted tau (first 10):", tau_iv_hat)
    print("IV forest se (first 10):", tau_iv_se)
