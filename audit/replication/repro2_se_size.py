"""Repro 2: did.twfe reports classical iid OLS SEs. Under AR(1)-within-unit
errors and zero true effect, the nominal 5% t-test on the ATET dummy should
reject ~5%; BDM (2004) predicts large over-rejection without unit clustering."""
import sys, warnings
warnings.filterwarnings('ignore')
sys.path.insert(0, '/home/user/statanomics/workingcode/panelmodels')
import numpy as np, pandas as pd
import statsmodels.api as sm
from panelib import did

dd = {'treatment':'treated','date':'time','post':'post','unitid':'unit_id','outcome':'y'}
N, T, T0, n_tr, rho, reps = 30, 20, 15, 5, 0.8, 150
rng = np.random.default_rng(7)

rej_iid, rej_clu = 0, 0
for r in range(reps):
    a = rng.normal(0, 1, N)
    # AR(1) errors within unit, stationary init
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
    # library's own inference: per-unit p-values (classical OLS)
    rej_iid += int((res['twfe']['pvalue'] < 0.05).any())

    # correct: same regression, cluster-robust by unit
    m = res['twfe_model']
    Xcols = m.model.exog_names
    X = pd.DataFrame(m.model.exog, columns=Xcols)
    groups = df['unit_id'].values
    mc = sm.OLS(df['y'], X).fit(cov_type='cluster', cov_kwds={'groups': groups})
    pv = mc.pvalues.iloc[1:1+n_tr]
    rej_clu += int((pv < 0.05).any())

# familywise over 5 unit dummies -> compare like with like (any-rejection)
print(f"AR(1) rho={rho}, zero effect, {reps} reps, {n_tr} treated units")
print(f"any-unit rejection at 5% -- library iid OLS SEs : {rej_iid/reps:.3f}")
print(f"any-unit rejection at 5% -- cluster(unit) SEs   : {rej_clu/reps:.3f}")
print(f"(nominal familywise ~{1-(0.95**n_tr):.3f} under independence of tests)")
