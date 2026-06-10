"""
Monte Carlo simulation: 100 seeds, DGP matching panel_model_comparison.ipynb
Reports mean bias, variance, and IQR for DiD, SC (DI), and SDID.
SC: point estimate only (bypasses conformal inference for speed).
SDID: point estimate only (no permutation inference).
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import numpy as np
import pandas as pd
import warnings
warnings.filterwarnings('ignore')

import panelib
from panelib import did, sc, sdid, dgp, alpha_lambda
import panelib as _pb

# Access the internal di class directly
_di = _pb.di

# ── DGP parameters ────────────────────────────────────────────────────────────
N_CONTROL    = 100
N_TREATED    = 1
N_TOTAL      = N_CONTROL + N_TREATED
T_PRE        = 10
T_POST       = 3
T_TOTAL      = T_PRE + T_POST
NOISE_SD     = 0.1
TIME_TREND   = 0.3
N_SEEDS      = 100

DATA_DICT = {
    'treatment': 'treated',
    'date':      'time',
    'post':      'post',
    'unitid':    'unit_id',
    'outcome':   'y'
}

control_ids  = [f'C{i:03d}' for i in range(N_CONTROL)]
treated_id   = 'T000'
unit_ids     = [treated_id] + control_ids
time_periods = np.arange(T_TOTAL)


def make_dgp(seed):
    np.random.seed(seed)
    unit_fe = np.random.normal(5, 2, N_TOTAL)
    rows = []
    y_cf_t0 = None
    for i, uid in enumerate(unit_ids):
        is_treated = (uid == treated_id)
        for t_idx in time_periods:
            y = unit_fe[i] + TIME_TREND * t_idx + np.random.normal(0, NOISE_SD)
            post = 1 if t_idx >= T_PRE else 0
            if is_treated and t_idx == T_PRE:
                y_cf_t0 = y
            rows.append({'unit_id': uid, 'time': t_idx,
                         'y_base': y,
                         'treated': 1 if is_treated else 0,
                         'post': post})
    true_att = 0.02 * y_cf_t0
    df = pd.DataFrame(rows)
    df['y'] = df['y_base'].copy()
    df.loc[(df['treated'] == 1) & (df['post'] == 1), 'y'] += true_att
    return df, true_att


def sc_point_estimate(df):
    """Run SC (DI elastic net) and return ATT without conformal inference.

    Speed optimisation for MC: find alpha/l1_ratio with a SINGLE ElasticNetCV
    on the treated unit vs. all controls (instead of LOO over all 100 controls).
    This is ~100× faster while still choosing reasonable regularisation.
    """
    from sklearn.linear_model import ElasticNetCV
    pre_data = dgp.clean_and_input_data(
        dataset=df,
        treatment=DATA_DICT['treatment'],
        unit_id=DATA_DICT['unitid'],
        date=DATA_DICT['date'],
        post=DATA_DICT['post'],
        outcome=DATA_DICT['outcome']
    )
    pre_train_test = dgp.determine_pre_train_test_lengths(
        ci_data_output=pre_data, pre_train_test_lengths=None
    )

    # ── Fast hyperparameter search: CV on treated unit only ───────────────────
    y = pre_data['T_pre'].values.flatten()       # shape (T_pre,)
    X = pre_data['C_pre'].values                 # shape (T_pre, N_control)
    cv = ElasticNetCV(
        l1_ratio=[0.1, 0.5, 0.9, 1.0],
        cv=3, max_iter=10000, random_state=42
    )
    cv.fit(X, y)
    best_alpha = float(cv.alpha_)
    best_l1    = float(np.clip(cv.l1_ratio_, 1e-6, 1 - 1e-6))
    raw0 = 1000.0 * np.log(best_alpha)
    raw1 = np.log(best_l1 / (1.0 - best_l1))

    class _Result:
        x = np.array([raw0, raw1])
    al = alpha_lambda.alpha_lambda_transform(_Result().x)

    sc_est = _di.predict_mu_omega(pre_data['T_pre'], pre_data['C_pre'],
                                  al, pre_train_test)
    sc_out = _di.sc_style_results(
        pre_data['T_pre'], pre_data['T_pst'],
        pre_data['C_pre'], pre_data['C_pst'],
        sc_est['mu'], sc_est['omega']
    )
    # ATT = mean over post-periods (aggregate_pst_periods=True logic)
    att = float(sc_out['atet'].mean(axis=0).mean())
    return att


results = []
for seed in range(N_SEEDS):
    if seed % 10 == 0:
        print(f"  seed {seed:3d} / {N_SEEDS}", flush=True)

    df, true_att = make_dgp(seed)
    row = {'seed': seed, 'true_att': true_att}

    # ── DiD ──────────────────────────────────────────────────────────────────
    try:
        r = did.twfe(data=df, data_dict=DATA_DICT)
        att = float(r['twfe']['coef_'].iloc[0])
        row['did_att']  = att
        row['did_bias'] = att - true_att
    except Exception as e:
        print(f"    DiD failed seed={seed}: {e}")
        row['did_att'] = row['did_bias'] = np.nan

    # ── SC (DI) — point estimate only, no conformal ───────────────────────────
    try:
        att = sc_point_estimate(df)
        row['sc_att']  = att
        row['sc_bias'] = att - true_att
    except Exception as e:
        print(f"    SC  failed seed={seed}: {e}")
        row['sc_att'] = row['sc_bias'] = np.nan

    # ── SDID — point estimate only ────────────────────────────────────────────
    try:
        r = sdid.twfe_sdid(data=df, data_dict=DATA_DICT)
        att = float(r['sdid'].loc['post_SDiD', 'coef_'])
        row['sdid_att']  = att
        row['sdid_bias'] = att - true_att
    except Exception as e:
        print(f"    SDID failed seed={seed}: {e}")
        row['sdid_att'] = row['sdid_bias'] = np.nan

    results.append(row)

res = pd.DataFrame(results)

print("\n" + "=" * 65)
print("  MONTE CARLO RESULTS — 100 seeds, NOISE_SD=0.1, N_CONTROL=100")
print("=" * 65)
true_mean = res['true_att'].mean()
print(f"  True ATT (mean across seeds): {true_mean:.4f}\n")

for model, col in [('DiD', 'did'), ('SC (DI)', 'sc'), ('SDID', 'sdid')]:
    bias = res[f'{col}_bias'].dropna()
    att  = res[f'{col}_att'].dropna()
    q25, q75 = np.percentile(bias, [25, 75])
    print(f"{model}  (n={len(bias)})")
    print(f"  Mean ATT:       {att.mean():.4f}")
    print(f"  Mean bias:      {bias.mean():.4f}")
    print(f"  Bias std dev:   {bias.std():.4f}")
    print(f"  Bias variance:  {bias.var():.6f}")
    print(f"  IQR of bias:    [{q25:.4f}, {q75:.4f}]  (width={q75-q25:.4f})")
    print(f"  RMSE:           {np.sqrt((bias**2).mean()):.4f}")
    print()

out_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'mc_results.csv')
res.to_csv(out_path, index=False)
print(f"Full results saved to {out_path}")
