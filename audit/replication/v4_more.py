import sys
sys.path.insert(0, '/tmp/claude-0/-home-user-statanomics/28ff6dc2-2743-5832-a59f-2f25cf36520b/scratchpad/stubs')
sys.path.insert(0, '/home/user/statanomics')
import numpy as np
np.int = int   # restore removed alias so we can see the NEXT bug in HR/SGCT
import pandas as pd, statsmodels.api as sm
from sklearn.linear_model import LogisticRegression, LinearRegression
import stnomics as st

rng = np.random.default_rng(3)
def mkdf(n, rng):
    X = rng.normal(size=(n,2)); e = 1/(1+np.exp(-X[:,0])); D = rng.binomial(1,e)
    Y = X[:,0]+X[:,1]+(1+0.5*X[:,0])*D+rng.normal(size=n)
    return pd.DataFrame({'x1':X[:,0],'x2':X[:,1],'D':D,'Y':Y})
tm = LogisticRegression(C=1e6, max_iter=2000); ym = LinearRegression()

print("== A. HR / SGCT with np.int patched -> next failure ==")
df = mkdf(1000, rng)
for name, fn in [('HR', st.hte.het_dml_approaches.HR), ('SGCT', st.hte.het_dml_approaches.SGCT)]:
    try:
        out = fn(df.copy(), ['x1','x2'], 'Y', 'D', ['x1'], ym, tm, 4, {'force_second_stage':'OLS'})
        print("   %s ran" % name)
    except Exception as ex:
        print("   %s CRASH: %s: %s" % (name, type(ex).__name__, str(ex)[:100]))

print("== B. DR second stage (unpacks 4 correctly) with np.int patched ==")
try:
    out = st.hte.other.DR(df.copy(), ['x1','x2'], 'Y', 'D', ['x1'], ym, tm, 4, {'force_second_stage':'OLS'})
    te = np.array(out[0]); print("   DR OLS 2nd stage ran; corr(TE_hat, 1+0.5*x1) across units = %.3f" %
          np.corrcoef(te, 1+0.5*df.sort_index()['x1'])[0,1])
except Exception as ex:
    print("   DR CRASH:", type(ex).__name__, str(ex)[:100])

print("== C. predict_cv misalignment when splits are NOT block-contiguous ==")
n = 2000
X = rng.normal(size=(n,3))
target = X @ np.array([1.0, -2.0, 0.5])          # deterministic, perfectly learnable
d = pd.DataFrame(X, columns=['a','b','c']); d['tgt'] = target
d['splits_random'] = rng.integers(0, 4, n)        # like diagnostics.balance.feature_balance line 626
xhat = st.predict_cv(d, 'splits_random', 4, ['a','b','c'], 'tgt', LinearRegression())
print("   random splits:  corr(xhat, target) = %.3f ; RMSE = %.3f" %
      (np.corrcoef(xhat, target)[0,1], np.sqrt(np.mean((xhat-target)**2))))
st.block_splits(d, 'splits_block', 4)
xhat2 = st.predict_cv(d, 'splits_block', 4, ['a','b','c'], 'tgt', LinearRegression())
print("   block  splits:  corr(xhat, target) = %.3f ; RMSE = %.3e" %
      (np.corrcoef(xhat2, target)[0,1], np.sqrt(np.mean((xhat2-target)**2))))

print("== D. propbinning_main 'ATE SE' does not shrink with n ==")
for n in [2000, 8000]:
    df = mkdf(n, np.random.default_rng(11))
    r = st.ate.propbinning_main(df, 'splits', ['x1','x2'], 'Y', 'D', ym, tm, 4, {'n_bins':5})
    print("   n=%5d: ATE TE=%.3f  reported ATE SE=%.3f  (sd of tau(x)=1+0.5*x1 is %.3f)"
          % (n, r['ATE TE'], r['ATE SE'], np.std(1+0.5*df['x1'])))

print("== E. dml_plm: point est + SE calibration under heteroskedasticity ==")
def mkdf_het(n, rng):
    X = rng.normal(size=(n,2)); e = 1/(1+np.exp(-X[:,0])); D = rng.binomial(1,e)
    Y = X[:,0]+X[:,1]+1.0*D + np.abs(X[:,0])*1.5*rng.normal(size=n)   # heteroskedastic
    return pd.DataFrame({'x1':X[:,0],'x2':X[:,1],'D':D,'Y':Y})
est, ses = [], []
for s in range(200):
    df = mkdf_het(2000, np.random.default_rng(1000+s))
    r = st.ate.dml.dml_plm(df, 'splits', ['x1','x2'], 'Y', 'D', ym, tm, 4, {'lower':0.0,'upper':1.0})
    est.append(r['ATE TE']); ses.append(r['ATE SE'])
cover = np.mean(np.abs((np.array(est)-1.0)/np.array(ses)) <= 1.96)
print("   mean est=%.3f  MC sd=%.4f  mean reported SE=%.4f  95%% coverage=%.2f" %
      (np.mean(est), np.std(est), np.mean(ses), cover))

print("== F. second_stage OLS: intercept leaks into CATE prediction ==")
m = 3000
a = rng.normal(size=m); t = rng.normal(size=m)
y = 0.7 + 2.0*t + 1.0*t*a + 0.1*rng.normal(size=m)     # residualized-style: CATE(a) = 2 + a
d2 = pd.DataFrame({'y':y,'t':t,'t_xa':t*a,'a':a,'cons':1,'ones':1})
te, se, coefs, fitres = st.secondstage.second_stage('OLS', d2, d2, ['t','t_xa'], ['a'])
te = np.array(te)
print("   fitted params:", dict(np.round(fitres.params,3)))
print("   mean(TE_hat - (2 + a)) = %.3f   (should be 0; equals const=0.7 if intercept leaks)" %
      np.mean(te - (2+1.0*a)))

print("== G. block_splits with non-default index ==")
d3 = pd.DataFrame({'v':np.arange(10)}, index=np.arange(100,110))
try:
    st.block_splits(d3, 's', 3); print("   assigned:", d3['s'].to_list())
except Exception as ex:
    print("   CRASH:", type(ex).__name__, str(ex)[:100])
