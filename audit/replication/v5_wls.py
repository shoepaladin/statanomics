import sys
sys.path.insert(0, '/tmp/claude-0/-home-user-statanomics/28ff6dc2-2743-5832-a59f-2f25cf36520b/scratchpad/stubs')
sys.path.insert(0, '/home/user/statanomics')
import numpy as np, pandas as pd
from sklearn.linear_model import LogisticRegression, LinearRegression
import stnomics as st

tm = LogisticRegression(C=1e6, max_iter=2000); ym = LinearRegression()
def dgp(n, rng):
    X = rng.normal(size=(n,2))
    e = 1/(1+np.exp(-(0.8*X[:,0])))
    D = rng.binomial(1,e)
    tau = 1.0 + 0.5*X[:,0]
    Y = X[:,0]+X[:,1]+tau*D+rng.normal(size=n)
    return pd.DataFrame({'x1':X[:,0],'x2':X[:,1],'D':D,'Y':Y}), tau

a_ate, a_att, t_att, ols_e = [], [], [], []
for s in range(150):
    df, tau = dgp(3000, np.random.default_rng(s))
    r = st.ate.ipw_wls(df, 'splits', ['x1','x2'], 'Y', 'D', ym, tm, 4, {'lower':0.01,'upper':0.99})
    a_ate.append(r['ATE TE']); a_att.append(r['ATT TE']); t_att.append(tau[df['D']==1].mean())
    o = st.ate.ols_vanilla(df, 'splits', ['x1','x2'], 'Y', 'D', ym, tm, 4, {})
    ols_e.append(o['ATE TE'])
print("ipw_wls: ATE mean=%.3f (true 1.0)   ATT mean=%.3f (true %.3f)" %
      (np.mean(a_ate), np.mean(a_att), np.mean(t_att)))
print("ols_vanilla 'ATE': mean=%.3f (true ATE 1.0, true ATT %.3f) -> variance-weighted, neither" %
      (np.mean(ols_e), np.mean(t_att)))

# dml_irm trimmed: SE denominator uses full n even though estimate uses trimmed subsample
df, tau = dgp(3000, np.random.default_rng(7))
r1 = st.ate.dml.dml_irm(df.copy(), 'splits', ['x1','x2'], 'Y', 'D', ym, tm, 4, {'lower':0.35,'upper':0.65})
kept = ((r1['PScore']>=0.35)&(r1['PScore']<=0.65)).sum()
print("dml_irm heavy trim: kept %d of %d; SE divides by sqrt(full n) -> understated by factor %.2f"
      % (kept, len(df), np.sqrt(len(df)/kept)))
