"""
Utility functions for GRF package.
"""

import numpy as np


def generate_causal_data(n: int, p: int = 1, 
                         tau_func=lambda x: x[:, 0],
                         noise_std: float = 0.1,
                         random_state: Optional[int] = None) -> Tuple:
    """
    Generate synthetic data for causal inference.
    
    Parameters
    ----------
    n : int
        Number of samples
    p : int, default=1
        Number of features
    tau_func : callable, default=lambda x: x[:, 0]
        Function mapping covariates to treatment effects
    noise_std : float, default=0.1
        Standard deviation of outcome noise
    random_state : int, optional
        Random seed
        
    Returns
    -------
    X : array-like, shape (n, p)
        Covariates
    Y : array-like, shape (n,)
        Outcomes
    W : array-like, shape (n,)
        Treatments
    tau : array-like, shape (n,)
        True treatment effects
    """
    if random_state is not None:
        np.random.seed(random_state)
    
    X = np.random.uniform(-1, 1, (n, p))
    W = np.random.binomial(1, 0.5, n)
    tau = tau_func(X)
    Y = tau * W + np.random.normal(0, noise_std, n)
    
    return X, Y, W, tau


def compute_metrics(tau_true: np.ndarray, tau_hat: np.ndarray,
                   lower: np.ndarray, upper: np.ndarray) -> dict:
    """
    Compute evaluation metrics.
    
    Parameters
    ----------
    tau_true : array-like
        True treatment effects
    tau_hat : array-like
        Predicted treatment effects
    lower : array-like
        Lower confidence bounds
    upper : array-like
        Upper confidence bounds
        
    Returns
    -------
    metrics : dict
        Dictionary containing MSE, coverage, and CI width
    """
    mse = np.mean((tau_hat - tau_true)**2)
    coverage = np.mean((tau_true >= lower) & (tau_true <= upper))
    ci_width = np.mean(upper - lower)
    
    return {
        'mse': mse,
        'coverage': coverage,
        'ci_width': ci_width
    }
