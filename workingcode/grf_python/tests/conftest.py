"""
Shared pytest fixtures for GRF tests.
"""

import numpy as np
import pytest


def make_data(n=400, p=5, noise=0.3, seed=42):
    """
    Synthetic causal data with known heterogeneous treatment effect.
    tau(x) = x[:,0]  (linear in first feature, zero in others)
    """
    rng = np.random.default_rng(seed)
    X = rng.uniform(-1, 1, (n, p))
    W = rng.binomial(1, 0.5, n).astype(float)
    tau_true = X[:, 0]
    Y = tau_true * W + rng.normal(0, noise, n)
    return X, Y, W, tau_true


@pytest.fixture(scope="session")
def small_data():
    """n=300, p=4 — fast enough for unit tests."""
    return make_data(n=300, p=4, noise=0.3, seed=0)


@pytest.fixture(scope="session")
def medium_data():
    """n=800, p=6 — used for accuracy / coverage tests."""
    return make_data(n=800, p=6, noise=0.3, seed=1)


@pytest.fixture(scope="session")
def fitted_forest(small_data):
    """Pre-fitted NumbaCausalForest on small_data (session-scoped for speed)."""
    from grf.forest_numba import NumbaCausalForest
    X, Y, W, _ = small_data
    forest = NumbaCausalForest(
        n_trees=30, max_depth=6, min_leaf_size=5,
        n_quantiles=10, n_folds=2,
        verbose=0, random_state=7
    )
    forest.fit(X, Y, W)
    return forest
