"""
GRF: Generalized Random Forest for Causal Inference

Supports multiple performance tiers:
  - NumbaCausalForest: 2-4x faster than pure Python (always available)
  - CausalForestCython: 10-15x faster, matches econml (requires compilation)

Examples
--------
>>> from grf import CausalForest
>>> import numpy as np
>>> 
>>> # Generate data
>>> X = np.random.randn(1000, 5)
>>> W = np.random.binomial(1, 0.5, 1000)
>>> Y = X[:, 0] * W + np.random.randn(1000) * 0.1
>>> 
>>> # Fit forest (automatically uses best available version)
>>> forest = CausalForest(n_trees=100)
>>> forest.fit(X, Y, W)
>>> 
>>> # Predict
>>> tau, lower, upper = forest.predict_interval(X)
"""

from .forest_numba import NumbaCausalForest

# Try to import Cython version
try:
    from .forest_cython import CausalForestCython
    CYTHON_AVAILABLE = True
except ImportError:
    CYTHON_AVAILABLE = False
    CausalForestCython = None

# Default to best available implementation
if CYTHON_AVAILABLE:
    CausalForest = CausalForestCython
    _default_backend = "Cython"
else:
    CausalForest = NumbaCausalForest
    _default_backend = "Numba"

__version__ = "0.4.0"
__all__ = ["CausalForest", "NumbaCausalForest"]

if CYTHON_AVAILABLE:
    __all__.append("CausalForestCython")


def get_backend():
    """Return the currently active backend."""
    return _default_backend


def print_info():
    """Print information about available backends."""
    print("="*70)
    print("GRF Package Information")
    print("="*70)
    print(f"Version: {__version__}")
    print(f"Default backend: {_default_backend}")
    print(f"\nAvailable implementations:")
    print(f"  - NumbaCausalForest: [OK] (always available)")
    if CYTHON_AVAILABLE:
        print(f"  - CausalForestCython: [OK]")
    else:
        print(f"  - CausalForestCython: [NOT COMPILED]")
    
    if not CYTHON_AVAILABLE:
        print(f"\nTo enable Cython backend (10-15x speedup):")
        print(f"  1. Install dependencies: pip install cython")
        print(f"  2. Compile: python setup.py build_ext --inplace")
    
    print("="*70)


# Auto-print info on first import (optional - can be removed if annoying)
import sys
if 'pytest' not in sys.modules:  # Don't print during testing
    _first_import = True
    if _first_import and __name__ == 'grf':
        print(f"\nGRF loaded with {_default_backend} backend")
        if not CYTHON_AVAILABLE:
            print("Tip: Compile Cython extensions for 10-15x speedup")
        print()