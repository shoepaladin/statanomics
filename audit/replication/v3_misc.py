import sys
sys.path.insert(0, '/tmp/claude-0/-home-user-statanomics/28ff6dc2-2743-5832-a59f-2f25cf36520b/scratchpad/stubs')
sys.path.insert(0, '/home/user/statanomics')
import numpy as np, pandas as pd, statsmodels.api as sm
from sklearn import metrics
from sklearn.linear_model import LogisticRegression, LinearRegression
import stnomics as st

print("== A. bootstrap.go_reps ATT copy-paste ==")
def fake_model(df, *a):  # returns distinct ATE vs ATT
    return {'ATE TE':1.0, 'ATE SE':0.1, 'ATT TE':2.0, 'ATT SE':0.2}
df = pd.DataFrame({'z':np.arange(50)})
out = st.bootstrap.go_reps(3, fake_model, df, 0,0,0,0,0,0,0,0)
print("   est_model returns ATE=1, ATT=2 ->  go_reps says ATT mean =", out['ATT mean'], ", ATT SE mean =", out['ATT SE mean'])

print("== B. predQC.r2 and battery ==")
rng = np.random.default_rng(1)
t = rng.normal(size=500); e = 0.8*t + 0.6*rng.normal(size=500)
print("   sqrt(corr) =", round(st.predQC.r2(t,e),4), " vs r2_score =", round(metrics.r2_score(t,e),4),
      " vs corr^2 =", round(np.corrcoef(t,e)[0,1]**2,4))
print("   negative corr ->", st.predQC.r2(t, -e))
try:
    st.predQC.battery(t, e, (t>0).astype(int), (t>0).astype(int), (e>0).astype(int),
                      metrics.recall_score, metrics.r2_score)
    print("   battery ran")
except Exception as ex:
    print("   battery CRASH:", type(ex).__name__, str(ex)[:80])

print("== C. secondstage SE formula: sqrt(|x'Vb|) vs correct sqrt(x'Vx) ==")
n=400; X = rng.normal(size=(n,3))
y = 1 + X@np.array([2.,-1.,0.5]) + rng.normal(size=n)
Xc = sm.add_constant(pd.DataFrame(X, columns=['a','b','c']))
fit = sm.OLS(y, Xc).fit()
V = np.array(fit.cov_params())[1:,1:]
Xt = np.array(Xc)[:5]
code_se = np.sqrt(np.abs(np.dot(np.dot(Xt[:,1:], V), np.array(fit.params[1:]))))
correct_se = np.sqrt(np.einsum('ij,jk,ik->i', Xt[:,1:], V, Xt[:,1:]))
print("   code SE (first 5):   ", np.round(code_se,4))
print("   correct SE(first 5): ", np.round(correct_se,4))
print("   ratio:", np.round(code_se/correct_se,2))

print("== D. np.int crash in DR / HR / SGCT second stage (numpy %s) ==" % np.__version__)
def mkdf(n, rng):
    X = rng.normal(size=(n,2)); e = 1/(1+np.exp(-X[:,0])); D = rng.binomial(1,e)
    Y = X[:,0]+X[:,1]+(1+0.5*X[:,0])*D+rng.normal(size=n)
    return pd.DataFrame({'x1':X[:,0],'x2':X[:,1],'D':D,'Y':Y})
df = mkdf(1000, rng)
tm = LogisticRegression(C=1e6, max_iter=2000); ym = LinearRegression()
for name, fn in [('DR', st.hte.other.DR), ('HR', st.hte.het_dml_approaches.HR), ('SGCT', st.hte.het_dml_approaches.SGCT)]:
    try:
        out = fn(df.copy(), ['x1','x2'], 'Y', 'D', ['x1'], ym, tm, 4, {'force_second_stage':'OLS','lower':0.0,'upper':1.0})
        print("   %s ran, first TE: %.3f" % (name, out[0][0]))
    except Exception as ex:
        print("   %s CRASH: %s: %s" % (name, type(ex).__name__, str(ex)[:90]))

print("== E. DR with force_second_stage=None (pseudo-outcome path) ==")
try:
    out = st.hte.other.DR(df.copy(), ['x1','x2'], 'Y', 'D', ['x1'], ym, tm, 4, {'force_second_stage':None})
    print("   DR pseudo-outcome ran; mean pseudo-outcome = %.3f (ATE approx 1.0); SE sentinel:" % np.mean(out[0]), np.unique(out[1]))
except Exception as ex:
    print("   DR CRASH:", type(ex).__name__, str(ex)[:90])

print("== F. secondstage.second_stage returns 4 items; HR/SGCT unpack 3 ==")
import inspect
src = inspect.getsource(st.secondstage.second_stage)
print("   return stmt:", src.strip().splitlines()[-1].strip())

print("== G. Lasso branch: lasso_alpha undefined ==")
tr = pd.DataFrame({'y':rng.normal(size=100),'t':rng.normal(size=100),'t_xa':rng.normal(size=100),
                   'cons':1,'ones':1,'a':rng.normal(size=100)})
try:
    st.secondstage.second_stage('Lasso', tr, tr, ['t','t_xa'], ['a'])
except Exception as ex:
    print("   CRASH:", type(ex).__name__, str(ex)[:80])
