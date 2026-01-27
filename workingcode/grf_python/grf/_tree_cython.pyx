# cython: language_level=3
# cython: boundscheck=False
# cython: wraparound=False
# cython: cdivision=True

import numpy as np
cimport numpy as cnp
from libc.math cimport fabs

cnp.import_array()

ctypedef cnp.float64_t DOUBLE
ctypedef cnp.int32_t INT32


cdef class TreeBuilder:
    """Fast tree builder using Cython."""
    
    cdef:
        DOUBLE[:, :] X
        DOUBLE[:] Y_resid
        DOUBLE[:] W_resid
        int min_leaf_size
        int max_depth
        
    def __init__(self, DOUBLE[:, :] X, DOUBLE[:] Y_resid, DOUBLE[:] W_resid,
                 int min_leaf_size, int max_depth):
        self.X = X
        self.Y_resid = Y_resid
        self.W_resid = W_resid
        self.min_leaf_size = min_leaf_size
        self.max_depth = max_depth
    
    cpdef DOUBLE estimate_tau_ols(self, INT32[:] indices):
        """Fast OLS estimation."""
        cdef:
            int n = indices.shape[0]
            DOUBLE numerator = 0.0
            DOUBLE denominator = 0.0
            int i, idx
        
        if n < 2:
            return 0.0
        
        for i in range(n):
            idx = indices[i]
            numerator += self.W_resid[idx] * self.Y_resid[idx]
            denominator += self.W_resid[idx] * self.W_resid[idx]
        
        if fabs(denominator) < 1e-10:
            return 0.0
        
        return numerator / denominator
    
    cpdef tuple find_best_split(self, INT32[:] split_idx, DOUBLE tau_parent):
        """Find best split across all features."""
        cdef:
            int n_samples = split_idx.shape[0]
            int n_features = self.X.shape[1]
            int feat, i, pct_idx, n_left, n_right
            DOUBLE threshold, score, best_score = -1e308
            int best_feature = -1
            DOUBLE best_threshold = 0.0
            DOUBLE W_mean, sum_left, sum_right, mean_left, mean_right
            
            DOUBLE[:] feature_vals = np.empty(n_samples, dtype=np.float64)
            DOUBLE[:] sorted_vals = np.empty(n_samples, dtype=np.float64)
            DOUBLE[:] pseudo = np.empty(n_samples, dtype=np.float64)
        
        # Compute pseudo-outcomes
        W_mean = 0.0
        for i in range(n_samples):
            W_mean += self.W_resid[split_idx[i]]
        W_mean /= n_samples
        
        for i in range(n_samples):
            idx = split_idx[i]
            w_val = self.W_resid[idx]
            y_val = self.Y_resid[idx]
            pseudo[i] = (w_val - W_mean) * (y_val - tau_parent * w_val)
        
        # Search features
        for feat in range(n_features):
            # Extract and sort feature values
            for i in range(n_samples):
                feature_vals[i] = self.X[split_idx[i], feat]
                sorted_vals[i] = feature_vals[i]
            
            # Simple insertion sort
            self._sort_array(sorted_vals)
            
            # Try quartile thresholds
            for pct_idx in range(3):
                if pct_idx == 0:
                    threshold = sorted_vals[n_samples // 4]
                elif pct_idx == 1:
                    threshold = sorted_vals[n_samples // 2]
                else:
                    threshold = sorted_vals[3 * n_samples // 4]
                
                # Compute split score
                n_left = 0
                sum_left = 0.0
                sum_right = 0.0
                
                for i in range(n_samples):
                    if feature_vals[i] <= threshold:
                        sum_left += pseudo[i]
                        n_left += 1
                    else:
                        sum_right += pseudo[i]
                
                n_right = n_samples - n_left
                
                if n_left < self.min_leaf_size or n_right < self.min_leaf_size:
                    continue
                
                mean_left = sum_left / n_left
                mean_right = sum_right / n_right
                score = n_left * n_right * (mean_left - mean_right) * (mean_left - mean_right)
                
                if score > best_score:
                    best_score = score
                    best_feature = feat
                    best_threshold = threshold
        
        return (best_feature, best_threshold, best_score)
    
    cdef void _sort_array(self, DOUBLE[:] arr) nogil:
        """Simple insertion sort."""
        cdef:
            int i, j, n = arr.shape[0]
            DOUBLE key
        
        for i in range(1, n):
            key = arr[i]
            j = i - 1
            while j >= 0 and arr[j] > key:
                arr[j + 1] = arr[j]
                j -= 1
            arr[j + 1] = key
            