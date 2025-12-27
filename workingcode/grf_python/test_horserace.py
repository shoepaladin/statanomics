#!/usr/bin/env python
"""
Test runner for the GRF vs OLS horserace notebook.
Executes the notebook code directly to check for errors.
"""

import sys
import numpy as np
import matplotlib
matplotlib.use('Agg')  # non-interactive backend
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import importlib

# Add current directory to path
if "." not in sys.path:
    sys.path.insert(0, ".")

# Import and reload grf
import grf
importlib.reload(grf)
from grf import CausalForest

print("=" * 60)
print("GRF vs OLS Horserace - Test Run")
print("=" * 60)

# Seed
np.random.seed(123)

# === Data Generation ===
print("\n[1] Generating synthetic data...")
n = 2000
p = 6
X = np.random.normal(size=(n, p))
tau_true = (X[:, 0] > 0).astype(float) * 1.0 + 0.5 * X[:, 1]
prop = 0.25 + 0.5 * (X[:, 2] > 0).astype(float)
W = np.random.binomial(1, prop)
Y0 = 0.2 * X[:, 3] + np.random.normal(scale=1.0, size=n)
Y = Y0 + W * tau_true

X_train, X_test, Y_train, Y_test, W_train, W_test, tau_train, tau_test = train_test_split(
    X, Y, W, tau_true, test_size=0.5, random_state=1
)
print(f"   Data shape: n={n}, p={p}")
print(f"   Train: {X_train.shape[0]}, Test: {X_test.shape[0]}")

# === Fit CausalForest ===
print("\n[2] Fitting CausalForest...")
cf = CausalForest(
    n_trees=80,
    sample_fraction=0.6,
    honesty_fraction=0.5,
    min_node_size=30,
    max_depth=6,
    random_state=1
)
cf.fit(X_train, Y_train, W_train)
print("   Predicting with CausalForest on test set...")
tau_cf_test, tau_cf_se = cf.predict(X_test)
print("   CausalForest: first 5 tau estimates:", np.round(tau_cf_test[:5], 3))

# === Fit OLS with interactions ===
print("\n[3] Fitting OLS with W*X interactions...")
def build_interaction_design(X, W):
    n, p = X.shape
    scaler = StandardScaler()
    Xs = scaler.fit_transform(X)
    intercept = np.ones((n, 1))
    Wcol = W.reshape(-1, 1)
    WX = (Wcol * Xs)
    return np.hstack([intercept, Wcol, Xs, WX]), scaler

X_design_train, scaler = build_interaction_design(X_train, W_train)
print(f"   OLS design matrix shape: {X_design_train.shape}")
lr = LinearRegression()
lr.fit(X_design_train, Y_train)
coef = lr.coef_
coef_W = coef[1]
coef_WX = coef[2 + X.shape[1]: 2 + 2 * X.shape[1]]
Xs_test = scaler.transform(X_test)
tau_ols_test = coef_W + Xs_test.dot(coef_WX)
print("   OLS: first 5 tau estimates:", np.round(tau_ols_test[:5], 3))

# === Evaluation ===
print("\n[4] Evaluating performance...")
def mse(a, b):
    return np.mean((a - b) ** 2)

mse_cf = mse(tau_cf_test, tau_test)
mse_ols = mse(tau_ols_test, tau_test)

print(f"\n   === RESULTS ===")
print(f"   MSE (CausalForest)     = {mse_cf:.4f}")
print(f"   MSE (OLS interactions) = {mse_ols:.4f}")
print(f"   Winner: {'CausalForest' if mse_cf < mse_ols else 'OLS interactions'}")

# === Plotting ===
print("\n[5] Creating plots...")
fig, axes = plt.subplots(1, 2, figsize=(12, 5))

axes[0].scatter(tau_test, tau_cf_test, alpha=0.4, s=10)
axes[0].plot([tau_test.min(), tau_test.max()], [tau_test.min(), tau_test.max()], "r--", linewidth=2)
axes[0].set_xlabel("True tau")
axes[0].set_ylabel("CF estimated tau")
axes[0].set_title(f"CausalForest (MSE={mse_cf:.4f})")
axes[0].grid(True, alpha=0.3)

axes[1].scatter(tau_test, tau_ols_test, alpha=0.4, s=10, color="orange")
axes[1].plot([tau_test.min(), tau_test.max()], [tau_test.min(), tau_test.max()], "r--", linewidth=2)
axes[1].set_xlabel("True tau")
axes[1].set_ylabel("OLS estimated tau")
axes[1].set_title(f"OLS with interactions (MSE={mse_ols:.4f})")
axes[1].grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig("grf_vs_ols_horserace.png", dpi=150, bbox_inches="tight")
print("   Plot saved as grf_vs_ols_horserace.png")

print("\n" + "=" * 60)
print("✓ Test run completed successfully!")
print("=" * 60)
