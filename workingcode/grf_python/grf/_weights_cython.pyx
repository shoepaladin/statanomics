# cython: language_level=3
# cython: boundscheck=False
# cython: wraparound=False
# cython: cdivision=True

import numpy as np
cimport numpy as np

ctypedef np.float64_t DTYPE_t
ctypedef np.int32_t INT_t

def compute_forest_weights_cython(int[:, :, :] all_leaf_indices,
                                   int[:, :] all_leaf_sizes,
                                   int n_train,
                                   int n_trees):
    """
    Compute forest weights efficiently.
    
    Parameters
    ----------
    all_leaf_indices : (n_trees, n_test, max_leaf_size) int array
    all_leaf_sizes : (n_trees, n_test) int array
    n_train : int
    n_trees : int
    
    Returns
    -------
    weights : (n_test, n_train) float array
    """
    cdef int n_test = all_leaf_indices.shape[1]
    cdef int b, i, j, idx, n_leaf
    cdef double w, scale
    
    # Create output array
    cdef np.ndarray[DTYPE_t, ndim=2] weights = np.zeros((n_test, n_train), dtype=np.float64)
    
    # Compute weights
    for b in range(n_trees):
        for i in range(n_test):
            n_leaf = all_leaf_sizes[b, i]
            if n_leaf > 0:
                w = 1.0 / n_leaf
                for j in range(n_leaf):
                    idx = all_leaf_indices[b, i, j]
                    if 0 <= idx < n_train:
                        weights[i, idx] += w
    
    # Average over trees
    scale = 1.0 / n_trees
    for i in range(n_test):
        for j in range(n_train):
            weights[i, j] *= scale
    
    return weights
    