"""
200-seed Monte Carlo: DiD vs SC (DI) vs SDID
Uses panelib's public API: did.twfe, sc.sc_att, sdid.twfe_sdid
"""
import warnings, time, sys, os
warnings.filterwarnings('ignore')
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import numpy as np
import pandas as pd
from panelib import did, sc, sdid

# ── Config ─────────────────────────────────────────────────────────────────
DGP_CONFIGS = {
    'Baseline':            dict(sigma_lambda=0.0, T_pre=10, N_control=100),
    'A: T-rich, additive': dict(sigma_lambda=0.0, T_pre=50, N_control=20),
    'B: T-poor, factor':   dict(sigma_lambda=0.5, T_pre=10, N_control=100),
    'C: T-rich, factor':   dict(sigma_lambda=0.5, T_pre=50, N_control=20),
}

T_POST   = 3
NOISE_SD = 0.1
ATT_PCT  = 0.02
N_SEEDS  = 200

DATA_DICT = {'treatment': 'treated', 'date': 'time',
             'post': 'post', 'unitid': 'unit_id', 'outcome': 'y'}


def make_panel_df(seed, T_pre, N_control, sigma_lambda):
    np.random.seed(seed)
    N_total  = N_control + 1
    T_total  = T_pre + T_POST
    unit_ids = ['T000'] + [f'C{i:03d}' for i in range(N_control)]
    alpha_i  = np.random.normal(5, 2, N_total)
    lambda_i = np.random.normal(1, sigma_lambda, N_total)
    F_t      = np.random.normal(0, 1, T_total)
    rows, y_cf = [], None
    for i, uid in enumerate(unit_ids):
        is_treated = (uid == 'T000')
        for t in range(T_total):
            y = (alpha_i[i] + 0.3 * t + lambda_i[i] * F_t[t]
                 + np.random.normal(0, NOISE_SD))
            post = 1 if t >= T_pre else 0
            if is_treated and t == T_pre:
                y_cf = y
            rows.append({'unit_id': uid, 'time': t, 'y': y,
                         'treated': int(is_treated), 'post': post})
    true_att = ATT_PCT * y_cf
    df = pd.DataFrame(rows)
    df.loc[(df['treated'] == 1) & (df['post'] == 1), 'y'] += true_att
    return df, true_att


def run_mc(T_pre, N_control, sigma_lambda, label):
    rows = []
    for seed in range(N_SEEDS):
        if seed % 40 == 0:
            print(f"    seed {seed:3d}/{N_SEEDS}", flush=True)
        df, true_att = make_panel_df(seed, T_pre, N_control, sigma_lambda)
        row = {'seed': seed, 'true_att': true_att, 'dgp': label}

        try:
            r = did.twfe(data=df, data_dict=DATA_DICT)
            att = float(r['twfe']['coef_'].iloc[0])
            row['did_att'] = att; row['did_bias'] = att - true_att
        except Exception as e:
            row['did_att'] = row['did_bias'] = np.nan

        try:
            att = sc.sc_att(model_name='di', data=df, data_dict=DATA_DICT, fast=True)
            row['sc_att'] = att; row['sc_bias'] = att - true_att
        except Exception as e:
            row['sc_att'] = row['sc_bias'] = np.nan

        try:
            r = sdid.twfe_sdid(data=df, data_dict=DATA_DICT)
            att = float(r['sdid'].loc['post_SDiD', 'coef_'])
            row['sdid_att'] = att; row['sdid_bias'] = att - true_att
        except Exception as e:
            row['sdid_att'] = row['sdid_bias'] = np.nan

        rows.append(row)
    return pd.DataFrame(rows)


def summarise(res, label):
    records = []
    for model, col in [('DiD', 'did'), ('SC (DI)', 'sc'), ('SDID', 'sdid')]:
        b = res[f'{col}_bias'].dropna()
        q25, q75 = np.percentile(b, [25, 75])
        records.append({
            'DGP': label, 'Model': model, 'n': len(b),
            'True_ATT_mean': round(res['true_att'].mean(), 4),
            'Mean_ATT':      round(res[f'{col}_att'].dropna().mean(), 4),
            'Mean_bias':     round(b.mean(), 4),
            'Std':           round(b.std(),  4),
            'Variance':      round(b.var(),  6),
            'IQR_lo':        round(q25,      4),
            'IQR_hi':        round(q75,      4),
            'IQR_width':     round(q75-q25,  4),
            'RMSE':          round(np.sqrt((b**2).mean()), 4),
        })
    return pd.DataFrame(records)


# ── Run ─────────────────────────────────────────────────────────────────────
all_raw, summaries = [], []
t_total = time.time()

for name, cfg in DGP_CONFIGS.items():
    print(f"\n{'='*55}", flush=True)
    print(f"  {name}", flush=True)
    print(f"  sigma_lambda={cfg['sigma_lambda']}  "
          f"T_pre={cfg['T_pre']}  N_control={cfg['N_control']}", flush=True)
    print(f"{'='*55}", flush=True)
    t0 = time.time()
    res = run_mc(**cfg, label=name)
    elapsed = time.time() - t0
    all_raw.append(res)
    summaries.append(summarise(res, name))
    print(f"  Done in {elapsed:.0f}s", flush=True)
    for m, c in [('DiD', 'did'), ('SC (DI)', 'sc'), ('SDID', 'sdid')]:
        b = res[f'{c}_bias'].dropna()
        print(f"  {m:8s}: mean_bias={b.mean():+.4f}  "
              f"std={b.std():.4f}  rmse={np.sqrt((b**2).mean()):.4f}", flush=True)

summary_df = pd.concat(summaries, ignore_index=True)
raw_df     = pd.concat(all_raw,   ignore_index=True)

print(f"\n\nTotal runtime: {(time.time()-t_total)/60:.1f} min")

print("\n" + "="*70)
print("  MEAN BIAS  (rows=DGP, cols=Model)")
print("="*70)
pivot_bias = summary_df.pivot_table(
    index='DGP', columns='Model', values='Mean_bias'
)[['DiD', 'SC (DI)', 'SDID']].reindex(list(DGP_CONFIGS.keys()))
print(pivot_bias.round(4).to_string())

print("\n" + "="*70)
print("  RMSE")
print("="*70)
pivot_rmse = summary_df.pivot_table(
    index='DGP', columns='Model', values='RMSE'
)[['DiD', 'SC (DI)', 'SDID']].reindex(list(DGP_CONFIGS.keys()))
print(pivot_rmse.round(4).to_string())

print("\n" + "="*70)
print("  FULL DETAIL")
print("="*70)
cols = ['DGP', 'Model', 'True_ATT_mean', 'Mean_ATT', 'Mean_bias',
        'Std', 'Variance', 'IQR_lo', 'IQR_hi', 'IQR_width', 'RMSE']
print(summary_df[cols].round(4).to_string(index=False))

out = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'mc_200_results.csv')
summary_df.to_csv(out, index=False)
raw_df.to_csv(out.replace('results', 'raw'), index=False)
print(f"\nSaved to {out}")
