"""Repro 5: (i) malformed permutations in collect_sc_outputs_individual;
(ii) empty theta grid fallback (panelib.py:966-968);
(iii) sibling MC scripts' alpha_lambda_transform(1000*log(alpha)) = alpha^1000;
(iv) staggered adoption: event-study pre dummies mislabel treated periods."""
import sys, warnings
warnings.filterwarnings('ignore')
sys.path.insert(0, '/home/user/statanomics/workingcode/panelmodels')
import numpy as np
from panelib import alpha_lambda

print("== (i) permutations built in collect_sc_outputs_individual (lines 896-902) ==")
pre_T, train = 20, 15          # pre_train_test_lengths = [15, 5]
individual_time_list = np.arange(train + 1)   # line 897
perms = []
for i in range(pre_T + 1):                    # line 898
    half_A = individual_time_list[-1*(pre_T - i):]
    half_B = individual_time_list[0:i]
    perms.append(list(np.concatenate([half_A, half_B])))
lens = sorted({len(p) for p in perms})
dup = [k for k, p in enumerate(perms) if len(set(p)) != len(p)]
needed = pre_T + 1  # arrays passed to pvalue_calc have pre_T+1 elements
print(f"required permutation length = {needed}; observed lengths = {lens}")
print(f"permutations with duplicated indices: {len(dup)} of {len(perms)}")
print(f"max index referenced = {max(max(p) for p in perms)} (array has indices 0..{needed-1};"
      f" indices {train+1}..{pre_T} are never sampled)")

print("\n== (ii) theta_grid=None fallback ==")
atet = 3.0; i = atet/100
g = np.arange(-500*i + atet, -500*i + atet, i*5)   # lines 966-968 verbatim
print(f"len(theta_grid) = {len(g)}  -> CI inversion over an EMPTY grid -> [nan, nan]")

print("\n== (iii) MC scripts' hyperparameter transform ==")
best_alpha = 0.05
raw = np.array([1000.0*np.log(best_alpha), 0.0])   # run_2x2_mc.py:72, monte_carlo_100.py:103
a_used, l1_used = alpha_lambda.alpha_lambda_transform(raw)
print(f"ElasticNetCV chose alpha={best_alpha}; alpha actually used = {a_used}"
      f"  (= alpha^1000 -> unpenalized OLS fit)")

print("\n== (iv) staggered adoption: global hold_out_time ==")
# unit A treated at t=5, unit B treated at t=8 (post is unit-specific)
# hold_out_time = max date with post==0 among treated = 7 (unit B's last pre)
# -> unit A's periods 5,6 are BEFORE hold_out_time and get 'pre_treat' dummies
# although they are post-treatment for A. Shown by reconstruction:
import pandas as pd
dates = np.arange(10)
dfA = pd.DataFrame({'unit':'A', 'date':dates, 'post':(dates>=5).astype(int), 'treated':1})
dfB = pd.DataFrame({'unit':'B', 'date':dates, 'post':(dates>=8).astype(int), 'treated':1})
data = pd.concat([dfA, dfB])
hold_out_time = data.loc[(data['treated']==1)&(data['post']==0), 'date'].max()  # line 180-181
mislabeled = dfA[(dfA['post']==1) & (dfA['date'] < hold_out_time)]['date'].tolist()
print(f"hold_out_time = {hold_out_time}; unit A's treated periods coded as 'pre_treat': {mislabeled}")
