import sys
sys.path.insert(0, '/tmp/claude-0/-home-user-statanomics/28ff6dc2-2743-5832-a59f-2f25cf36520b/scratchpad/stubs')
sys.path.insert(0, '/home/user/statanomics')
import numpy as np, pandas as pd
from sklearn.linear_model import LogisticRegression, LinearRegression
import stnomics as st

rng = np.random.default_rng(0)

def dgp(n, rng):
    X = rng.normal(size=(n, 2))
    e = 1/(1+np.exp(-(0.5*X[:,0] - 0.5*X[:,1])))
    D = rng.binomial(1, e)
    tau = 1.0 + 0.5*X[:,0]              # heterogeneous effect -> ATT != ATE
    Y = X[:,0] + X[:,1] + tau*D + rng.normal(size=n)
    df = pd.DataFrame({'x1':X[:,0],'x2':X[:,1],'D':D,'Y':Y})
    return df, tau

aux = {'lower':0.0,'upper':1.0}
tm = LogisticRegression(C=1e6, max_iter=2000)
ym = LinearRegression()

ates, atts, ate_ses, true_atts = [], [], [], []
for s in range(200):
    df, tau = dgp(2000, np.random.default_rng(s))
    r = st.ate.ipw_main(df, 'splits', ['x1','x2'], 'Y', 'D', ym, tm, 4, aux)
    ates.append(r['ATE TE']); atts.append(r['ATT TE']); ate_ses.append(r['ATE SE'])
    true_atts.append(tau[df['D']==1].mean())

print("true ATE = 1.0 ; true ATT (avg over sims) = %.3f" % np.mean(true_atts))
print("mean ipw ATE      = %.4f  (MC sd of ATE estimator = %.4f)" % (np.mean(ates), np.std(ates)))
print("mean reported ATE SE = %.4f  -> ratio SE/MCsd = %.1f  (should be ~1; n=2000, sqrt(n)=%.1f)"
      % (np.mean(ate_ses), np.mean(ate_ses)/np.std(ates), np.sqrt(2000)))
print("mean ipw 'ATT'    = %.4f  vs true ATT %.3f ; P(D=1)*ATT = %.4f"
      % (np.mean(atts), np.mean(true_atts), 0.5*np.mean(true_atts)))
