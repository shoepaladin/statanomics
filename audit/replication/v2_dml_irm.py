import sys
sys.path.insert(0, '/tmp/claude-0/-home-user-statanomics/28ff6dc2-2743-5832-a59f-2f25cf36520b/scratchpad/stubs')
sys.path.insert(0, '/home/user/statanomics')
import numpy as np, pandas as pd
from sklearn.linear_model import LogisticRegression, LinearRegression
import stnomics as st

## DGP: m0 nonlinear (so LinearRegression ymodel is misspecified), pscore logistic (tmodel correct)
def dgp(n, rng):
    X = rng.normal(size=(n,2))
    e = 1/(1+np.exp(-(1.0*X[:,0])))
    D = rng.binomial(1, e)
    m0 = np.sin(2*X[:,0]) + X[:,1]**2          # nonlinear baseline
    tau = 1.0 + 1.0*X[:,0]                     # heterogeneous; ATT>ATE since e increasing in x1
    Y = m0 + tau*D + rng.normal(size=n)
    return pd.DataFrame({'x1':X[:,0],'x2':X[:,1],'D':D,'Y':Y}), tau

aux = {'lower':0.0,'upper':1.0}
tm = LogisticRegression(C=1e6, max_iter=2000)
ym = LinearRegression()

code_att, corr_att, true_att, code_ate = [], [], [], []
for s in range(200):
    rng = np.random.default_rng(s); df, tau = dgp(4000, rng)
    r = st.ate.dml.dml_irm(df, 'splits', ['x1','x2'], 'Y', 'D', ym, tm, 4, aux)
    code_att.append(r['ATT TE']); code_ate.append(r['ATE TE'])
    true_att.append(tau[df['D']==1].mean())
    ## Correct CCDDHNR (2018) IRM ATT score with the SAME cross-fitted nuisances:
    ## psi = [ D*(Y-m0hat) - e*(1-D)*(Y-m0hat)/(1-e) ] / phat  ,  theta = mean(psi)
    that = r['PScore']; D = df['D'].to_numpy(); Y = df['Y'].to_numpy()
    ## reconstruct yhat_control identically to the code
    yt, yc = st.predict_counterfactual_outcomes(df, 'splits', 4, ['x1','x2'], 'D', 'Y', ym)
    p = D.mean()
    psi = (D*(Y-yc) - that*(1-D)*(Y-yc)/(1-that))/p
    corr_att.append(psi.mean())

print("true ATT = %.3f" % np.mean(true_att))
print("dml_irm  ATT  : mean=%.3f  bias=%+.3f  (MC sd %.3f)" % (np.mean(code_att), np.mean(code_att)-np.mean(true_att), np.std(code_att)))
print("CCDDHNR ATT (same nuisances): mean=%.3f  bias=%+.3f  (MC sd %.3f)" % (np.mean(corr_att), np.mean(corr_att)-np.mean(true_att), np.std(corr_att)))
print("dml_irm  ATE  : mean=%.3f  (true 1.0)" % np.mean(code_ate))

## Now: trimming applied -> SE loop index alignment
df, tau = dgp(4000, np.random.default_rng(999))
aux2 = {'lower':0.10,'upper':0.90}
try:
    r = st.ate.dml.dml_irm(df, 'splits', ['x1','x2'], 'Y', 'D', ym, tm, 4, aux2)
    print("trimmed dml_irm ran; ATE SE =", r['ATE SE'])
except Exception as ex:
    print("trimmed dml_irm CRASH:", type(ex).__name__, str(ex)[:150])

## dml_plm with aux['yhat']=None
try:
    aux3 = {'lower':0.0,'upper':1.0,'yhat':None}
    r = st.ate.dml.dml_plm(df, 'splits', ['x1','x2'], 'Y', 'D', ym, tm, 4, aux3)
    print("dml_plm yhat=None ran:", r['ATE TE'])
except Exception as ex:
    print("dml_plm yhat=None CRASH:", type(ex).__name__, str(ex)[:120])
