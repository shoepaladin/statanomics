"""Repro 7: (i) alpha_lambda tuning objective is IN-SAMPLE MSE -> monotone
decreasing in the penalty, so 'optimal' regularization is always ~0;
(ii) di.estimate_mu_omega labels control-unit weights with TREATMENT column names."""
import sys, warnings
warnings.filterwarnings('ignore')
sys.path.insert(0, '/home/user/statanomics/workingcode/panelmodels')
import numpy as np, pandas as pd
from panelib import alpha_lambda, di

rng = np.random.default_rng(1)
Cpre = pd.DataFrame(rng.normal(0, 1, (10, 8)), columns=[f'C{j}' for j in range(8)])

print("== (i) tuning objective vs alpha (l1_ratio fixed at 0.5) ==")
for raw_a in [2.0, 0.0, -2.0, -5.0, -10.0]:
    val = alpha_lambda.alpha_lambda_diff(np.array([raw_a, 0.0]), Cpre)
    print(f"  alpha = exp({raw_a:>5}) = {np.exp(raw_a):.5f}   objective = {val:.6f}")

print("\n== (ii) weight labels ==")
Tpre = pd.DataFrame(rng.normal(0, 1, (10, 2)), columns=['TREAT_A', 'TREAT_B'])
out = di.estimate_mu_omega(Tpre, Cpre, (0.1, 0.5))
print("coef matrix shape (should be 2 treated x 8 controls):", np.shape(out['omega']))
print("weights table returned (labels come from TREATED columns, truncated to 2 rows):")
print(out['weights'])
