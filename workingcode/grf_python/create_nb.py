import json

cells = [
    {
        'cell_type': 'markdown',
        'metadata': {},
        'source': ['# GRF vs OLS Horserace\nThis notebook compares CausalForest from `grf.py` against OLS with interactions.']
    },
]

code_sources = [
    'import sys\nimport numpy as np\nimport matplotlib.pyplot as plt\nfrom sklearn.linear_model import LinearRegression\nfrom sklearn.model_selection import train_test_split\nfrom sklearn.preprocessing import StandardScaler\nimport importlib\n\nif "." not in sys.path:\n    sys.path.insert(0, ".")\n\nimport grf\nimportlib.reload(grf)\nfrom grf import CausalForest\n\nnp.random.seed(123)',
    'n = 2000\np = 6\nX = np.random.normal(size=(n, p))\ntau_true = (X[:, 0] > 0).astype(float) * 1.0 + 0.5 * X[:, 1]\nprop = 0.25 + 0.5 * (X[:, 2] > 0).astype(float)\nW = np.random.binomial(1, prop)\nY0 = 0.2 * X[:, 3] + np.random.normal(scale=1.0, size=n)\nY = Y0 + W * tau_true\nX_train, X_test, Y_train, Y_test, W_train, W_test, tau_train, tau_test = train_test_split(X, Y, W, tau_true, test_size=0.5, random_state=1)\nprint(f"Data shape: n={n}, p={p}")\nprint(f"Train: {X_train.shape[0]}, Test: {X_test.shape[0]}")',
    'cf = CausalForest(n_trees=80, sample_fraction=0.6, honesty_fraction=0.5, min_node_size=30, max_depth=6, random_state=1)\nprint("Fitting CausalForest...")\ncf.fit(X_train, Y_train, W_train)\nprint("Predicting with CausalForest on test set...")\ntau_cf_test, tau_cf_se = cf.predict(X_test)\nprint("CausalForest: first 5 tau estimates (test):", np.round(tau_cf_test[:5], 3))',
    'def build_interaction_design(X, W):\n    n, p = X.shape\n    scaler = StandardScaler()\n    Xs = scaler.fit_transform(X)\n    intercept = np.ones((n, 1))\n    Wcol = W.reshape(-1, 1)\n    WX = (Wcol * Xs)\n    return np.hstack([intercept, Wcol, Xs, WX]), scaler\n\nX_design_train, scaler = build_interaction_design(X_train, W_train)\nprint(f"OLS design matrix shape: {X_design_train.shape}")\nlr = LinearRegression()\nprint("Fitting OLS with interactions...")\nlr.fit(X_design_train, Y_train)\ncoef = lr.coef_\ncoef_W = coef[1]\ncoef_WX = coef[2 + X.shape[1]: 2 + 2 * X.shape[1]]\nXs_test = scaler.transform(X_test)\ntau_ols_test = coef_W + Xs_test.dot(coef_WX)\nprint("OLS interactions: first 5 tau estimates (test):", np.round(tau_ols_test[:5], 3))',
    'def mse(a, b):\n    return np.mean((a - b) ** 2)\n\nmse_cf = mse(tau_cf_test, tau_test)\nmse_ols = mse(tau_ols_test, tau_test)\n\nprint(f"\\n=== RESULTS ===")\nprint(f"MSE (CausalForest)     = {mse_cf:.4f}")\nprint(f"MSE (OLS interactions) = {mse_ols:.4f}")\nprint(f"Winner: {\'CausalForest\' if mse_cf < mse_ols else \'OLS interactions\'}")',
    'fig, axes = plt.subplots(1, 2, figsize=(12, 5))\naxes[0].scatter(tau_test, tau_cf_test, alpha=0.4, s=10)\naxes[0].plot([tau_test.min(), tau_test.max()], [tau_test.min(), tau_test.max()], "r--", linewidth=2)\naxes[0].set_xlabel("True tau")\naxes[0].set_ylabel("CF estimated tau")\naxes[0].set_title(f"CausalForest (MSE={mse_cf:.4f})")\naxes[0].grid(True, alpha=0.3)\naxes[1].scatter(tau_test, tau_ols_test, alpha=0.4, s=10, color="orange")\naxes[1].plot([tau_test.min(), tau_test.max()], [tau_test.min(), tau_test.max()], "r--", linewidth=2)\naxes[1].set_xlabel("True tau")\naxes[1].set_ylabel("OLS estimated tau")\naxes[1].set_title(f"OLS with interactions (MSE={mse_ols:.4f})")\naxes[1].grid(True, alpha=0.3)\nplt.tight_layout()\nplt.savefig("grf_vs_ols_horserace.png", dpi=150, bbox_inches="tight")\nplt.show()\nprint("Plot saved as grf_vs_ols_horserace.png")',
]

for src in code_sources:
    cells.append({
        'cell_type': 'code',
        'execution_count': None,
        'metadata': {},
        'outputs': [],
        'source': [src]
    })

notebook = {
    'cells': cells,
    'metadata': {
        'kernelspec': {'display_name': 'Python 3', 'language': 'python', 'name': 'python3'},
        'language_info': {'name': 'python', 'version': '3.12.7'}
    },
    'nbformat': 4,
    'nbformat_minor': 5
}

with open('grf_vs_ols_horserace.ipynb', 'w') as f:
    json.dump(notebook, f, indent=1)

print('Notebook created successfully')
