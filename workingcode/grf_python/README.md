# GRF: Generalized Random Forest for Causal Inference

A Python implementation of Generalized Random Forests (GRF) for estimating
heterogeneous treatment effects (CATEs), with valid confidence intervals via
the Infinitesimal Jackknife variance estimator.

## Installation

```bash
git clone https://github.com/yourusername/grf-python.git
cd grf-python
pip install -e .
```

## Quick Start

```python
from grf.forest_numba import NumbaCausalForest
import numpy as np

rng = np.random.default_rng(42)
n, p = 1000, 10
X = rng.standard_normal((n, p))
W = rng.binomial(1, 0.5, n).astype(float)
tau_true = X[:, 0] + 0.5 * X[:, 1]          # true CATE
Y = tau_true * W + rng.standard_normal(n)

forest = NumbaCausalForest(n_trees=200, random_state=42)
forest.fit(X, Y, W)

tau_hat = forest.predict(X)
tau_hat, lower, upper = forest.predict_interval(X, alpha=0.05)
```

## API Reference

### `NumbaCausalForest`

```python
NumbaCausalForest(
    n_trees         = 100,    # trees in the forest
    subsample_ratio = 0.5,    # fraction of n drawn per tree (without replacement)
    min_leaf_size   = 10,     # minimum estimation-sample observations per leaf
    max_depth       = 10,     # maximum tree depth
    mtry            = None,   # features considered per split; None → ceil(p/3)
    n_quantiles     = 20,     # max candidate split thresholds per feature
    n_folds         = 4,      # cross-fitting folds for nuisance estimation
    honesty_fraction= 0.5,    # share of subsample used for structure (vs estimation)
    use_parallel    = True,   # Numba prange within a single tree's feature search
    n_jobs          = 'auto', # loky processes for tree building (see below)
    verbose         = 0,      # 0 = silent, 1 = progress
    random_state    = None,
)
```

#### Key methods

| Method | Returns |
|--------|---------|
| `fit(X, Y, W)` | self |
| `predict(X)` | `tau_hat` array |
| `predict(X, return_std=True)` | `(tau_hat, std_errors)` |
| `predict_interval(X, alpha=0.05)` | `(tau_hat, lower, upper)` |
| `oob_predict()` | out-of-bag CATE estimates for training points |
| `effect(X)` | econml-compatible alias for `predict` |
| `effect_interval(X, alpha)` | econml-compatible alias for `predict_interval` |

---

## Parallelization Guide (`n_jobs`)

Tree building can run sequentially or in parallel (loky processes).
The default `n_jobs='auto'` decides automatically.

### How auto-detection works

Auto mode estimates per-tree work as:

```
work_units = n_trees × n_sub × mtry × effective_q
```

where:
- `n_sub = subsample_ratio × n`
- `mtry = ceil(p / 3)`
- `effective_q = min(n_quantiles, split_n // 10)`

If `work_units ≥ 1,000,000`, it uses `min(cpu_count, 4)` loky workers.
Otherwise it stays sequential.

### Why 1,000,000 units?

Loky spawns separate OS processes, which has a cold-start overhead of ~0.3s.
Below the threshold the overhead exceeds the parallelism benefit.
The threshold was calibrated from isolated tree-building benchmarks
(nuisance RF excluded, best-of-4 runs):

| n | p | n_trees | work_units | seq (ms) | loky-2 (ms) | speedup |
|---|---|---------|-----------|----------|-------------|---------|
| 500 | 5 | 100 | 600k | 40 | 46 | 0.88× ❌ |
| 800 | 10 | 50 | 1,600k | 63 | 54 | **1.17×** ✓ |
| 1,000 | 10 | 30 | 1,200k | 51 | 42 | **1.21×** ✓ |
| 1,000 | 20 | 50 | 3,500k | 123 | 85 | **1.44×** ✓ |
| 2,000 | 40 | 20 | 5,600k | 322 | 194 | **1.66×** ✓ |

The break-even sits between 960k and 1,200k units.

### Rules of thumb

| Scenario | Typical work_units | `n_jobs='auto'` decision |
|----------|--------------------|--------------------------|
| n < 500, p < 10, any n_trees | < 600k | Sequential |
| n = 500, p = 5 | ~6k × n_trees; parallelize at n_trees ≥ 170 | Depends on n_trees |
| n = 1000, p = 10 | ~40k × n_trees; parallelize at n_trees ≥ 25 | Parallel for typical forests |
| n ≥ 2000, p ≥ 20 | > 1M even at n_trees = 20 | Always parallel |

**Note:** nuisance RF estimation (the `_estimate_nuisance` step) already uses
`n_jobs=-1` internally via scikit-learn and dominates total wall time for small
forests (~0.65s baseline on a single machine regardless of n_trees). The
parallelization gain from `n_jobs` only applies to the tree-building phase.

### Overriding auto-detection

```python
# Always sequential — safest, no overhead
forest = NumbaCausalForest(n_jobs=1)

# Always use all CPUs — best for large forests
forest = NumbaCausalForest(n_jobs=-1)

# Use exactly 2 workers
forest = NumbaCausalForest(n_jobs=2)

# Let the heuristic decide (default)
forest = NumbaCausalForest(n_jobs='auto')
```

### Why loky instead of threading?

Two failure modes plague threading for this workload:

1. **GIL contention**: `_build_tree` is recursive Python; the GIL is held
   throughout, so threading achieves no real concurrency on the Python layer.
2. **Thread pool oversubscription**: each tree uses Numba `prange` internally,
   spawning `n_cpu` threads. With `k` joblib threads that creates `k × n_cpu`
   threads on `n_cpu` cores — measured as a **3.5× slowdown** vs sequential.

loky gives each worker its own process and address space. Workers run with
`use_parallel=False` so there is no nested thread pool.

---

## Algorithm Details

### 1. Cross-fitted nuisance estimation

```
For each fold:
    fit E[Y|X] and E[W|X] on training fold → predict on validation fold
Y_resid = Y - Ŷ,   W_resid = W - Ŵ
```

Uses `n_folds`-fold cross-fitting with shuffled KFold to avoid order-dependent
bias. Random forests (100 estimators, depth 10) are used as flexible nuisance
learners.

### 2. Honest tree construction

Each tree subsample is split `honesty_fraction` / `(1 - honesty_fraction)` into:
- **Structure sample**: determines splits
- **Estimation sample**: estimates leaf treatment effects (never seen during splitting)

This prevents the overconfident predictions that arise when the same data both
chooses splits and estimates effects.

### 3. Gradient-based splitting

At each node, pseudo-outcomes are computed relative to the node's local effect:

```
τ_parent = OLS estimate in parent node
ρᵢ = (Wᵢ - W̄)(Yᵢ - τ_parent × Wᵢ)    ← pseudo-outcome
```

The split maximises `n_L × n_R × (mean(ρ_L) - mean(ρ_R))²`,
directly targeting treatment effect heterogeneity.

Feature subsampling uses `mtry = ceil(p/3)` per split, matching R `grf`'s
default. This outperforms `ceil(sqrt(p))` when ≥ 3 of p features are relevant
to treatment heterogeneity (the common case with moderately-sized p).

Candidate thresholds per feature are capped at:

```
effective_q = max(3, min(n_quantiles, split_n // 10))
```

This prevents thin quantile bins when the split sample is small.

### 4. Leaf estimation

Within each leaf, the CATE is estimated by OLS regressing `Y_resid` on
`W_resid` using the estimation-sample indices that fell into that leaf.

### 5. Infinitesimal Jackknife variance

Standard errors use the Wager–Athey (2018) subsampling IJ formula:

```
V̂(x) = (n/s) × Σⱼ Cov_b[T_b(x), 1{j ∈ Sₐ}]²
```

where `s` is the subsample size and `T_b(x)` is tree `b`'s prediction.
Implemented as a single matrix multiplication:

```python
T_sum = tree_preds.T @ subsample_flags   # (n_test, n_train)
cov_bj = T_sum / n_trees - T_bar[:, None] * p_j[None, :]
variances = (n / s) * np.sum(cov_bj ** 2, axis=1)
```

---

## What changed vs. the original implementation

The following bugs and limitations were fixed across three phases.

### Phase 1 — Correctness fixes

| Fix | Problem | Impact |
|-----|---------|--------|
| IJ variance | Old code computed `Σ(α_bj − ᾱ_j)` which is identically zero → always-zero std errors | Valid confidence intervals |
| KFold shuffle | 2-fold unshuffled KFold was order-dependent and biased for sorted data | Unbiased nuisance estimates |
| `n_folds` default | Hardcoded 2-fold → increased to 4 | Better nuisance estimation |
| `mtry` default | `ceil(sqrt(p))` → `ceil(p/3)` matching R `grf` | Lower CATE MSE, especially for moderate p |
| Instance RNG | `np.random.seed()` mutated global numpy state → not reproducible | True reproducibility |
| Adaptive quantiles | Fixed grid of 3 thresholds regardless of sample size | Finer splits on large nodes |

### Phase 2 — Performance fixes

| Fix | Problem | Impact |
|-----|---------|--------|
| Batch JIT traversal | Per-point Python while-loop O(n_test × n_trees) → `traverse_tree_batch` with Numba `prange` | 2–5× faster prediction |
| loky backend | Threading caused 3.5× slowdown from thread pool oversubscription | Parallel tree building is now net positive |
| Leaf buffer | `max_leaf_size = n // 2` over-allocated buffers by orders of magnitude | Lower memory, faster allocation |
| Sort outside loop | `np.sort` called inside percentile loop → called once per feature | Correct threshold selection |

### Phase 3 — New features

| Feature | Description |
|---------|-------------|
| Feature importances | Split-gain weighted, normalized; available as `forest.feature_importances_` |
| OOB predictions | `oob_predict()` uses only trees that excluded each point from their subsample |
| `honesty_fraction` | Tunable via constructor (default 0.5) |
| Input validation | Checks shape, NaN/Inf, feature count mismatch |
| sklearn compat | `get_params()`, `set_params()`, `n_features_in_` |

---

## Testing

```bash
cd workingcode/grf_python
pytest tests/ -v
```

59 tests across three modules:

| Module | What it tests |
|--------|---------------|
| `test_phase1_correctness.py` | IJ coverage, RNG isolation, mtry default, adaptive quantiles, nuisance fold count |
| `test_phase2_performance.py` | Sort fix, vectorised variance, leaf buffer, loky parallel, auto n_jobs heuristic, batch traversal |
| `test_phase3_features.py` | Feature importances, OOB predictions, input validation, honesty fraction, sklearn compat |

---

## References

- Wager & Athey (2018). "Estimation and Inference of Heterogeneous Treatment Effects using Random Forests." *JASA* 113(523).
- Athey, Tibshirani & Wager (2019). "Generalized Random Forests." *Annals of Statistics* 47(2).
