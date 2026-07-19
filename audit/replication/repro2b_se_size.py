"""Repro 2b: per-coefficient size of the library's t-test on the ATET dummy,
zero true effect, AR(1) errors within unit. Also size of an aggregated
2x2 DiD with cluster-robust SEs as the correct benchmark."""
import sys, warnings
warnings.filterwarnings('ignore')
sys.path.insert(0, '/home/user/statanomics/workingcode/panelmodels')
import numpy as np, pandas as pd
import statsmodels.formula.api as smf
from panelib import did

dd = {'treatment':'treated','date':'time','post':'post','unitid':'unit_id','outcome':'y'}
N, T, T0, n_tr, rho, reps = 30, 20, 15, 5, 0.8, 150
rng = np.random.default_rng(7)

rej_lib, n_lib = 0, 0
rej_agg, n_agg = 0, 0
for r in range(reps):
    a = rng.normal(0, 1, N)
    e = np.empty((N, T))
    e[:, 0] = rng.normal(0, 1/np.sqrt(1-rho**2), N)
    for t in range(1, T):
        e[:, t] = rho*e[:, t-1] + rng.normal(0, 1, N)
    gt = rng.normal(0, .3, T)
    rows = []
    for i in range(N):
        for t in range(T):
            rows.append(dict(unit_id=f'u{i:02d}', time=t,
                             treated=int(i < n_tr), post=int(t >= T0),
                             y=a[i] + gt[t] + 0.3*e[i, t]))
    df = pd.DataFrame(rows)
    res = did.twfe(data=df, data_dict=dd)
    rej_lib += int((res['twfe']['pvalue'] < 0.05).sum()); n_lib += n_tr

    # correct benchmark: pooled DiD, single ATT, cluster by unit (30 clusters)
    df['tp'] = df['treated']*df['post']
    m = smf.ols('y ~ tp + C(unit_id) + C(time)', data=df).fit(
        cov_type='cluster', cov_kwds={'groups': df['unit_id']})
    rej_agg += int(m.pvalues['tp'] < 0.05); n_agg += 1

print(f"AR(1) rho={rho}, zero effect, {reps} reps")
print(f"per-coefficient rejection at 5%, library iid OLS SE : {rej_lib/n_lib:.3f}")
print(f"pooled-ATT rejection at 5%, cluster(unit) SE        : {rej_agg/n_agg:.3f}")
