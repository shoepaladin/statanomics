import sys
sys.path.insert(0, '/tmp/claude-0/-home-user-statanomics/28ff6dc2-2743-5832-a59f-2f25cf36520b/scratchpad/stubs')
sys.path.insert(0, '/home/user/statanomics')
import numpy as np, pandas as pd, statsmodels.api as sm
from sklearn.linear_model import LogisticRegression, LinearRegression
import stnomics as st

print("pandas", pd.__version__)
print("== A. params[1] on labeled Series raises (pandas>=3) ==")
X = sm.add_constant(pd.DataFrame({'D':[0,1,0,1],'x':[1.,2.,3.,4.]}))
fit = sm.OLS(pd.Series([1.,2.,3.,4.]), X).fit()
try:
    fit.params[1]
except Exception as ex:
    print("   fit.params[1] ->", type(ex).__name__, "| index:", list(fit.params.index))

print("== B. ipw_wls weights logic (replicated exactly, .iloc for coef) ==")
tm = LogisticRegression(C=1e6, max_iter=2000)
def dgp(n, rng):
    Xm = rng.normal(size=(n,2))
    e = 1/(1+np.exp(-(0.8*Xm[:,0]))); D = rng.binomial(1,e)
    tau = 1.0 + 0.5*Xm[:,0]
    Y = Xm[:,0]+Xm[:,1]+tau*D+rng.normal(size=n)
    return pd.DataFrame({'x1':Xm[:,0],'x2':Xm[:,1],'D':D,'Y':Y}), tau
a_ate,a_att,t_att,ols_e = [],[],[],[]
for s in range(150):
    df, tau = dgp(3000, np.random.default_rng(s))
    st.block_splits(df,'splits',4)
    that = st.predict_treatment_indicator(df,'splits',4,['x1','x2'],'D',tm)
    keep = (that>=0.01)&(that<=0.99)
    w_ate = df['D']/that + (1-df['D'])/(1-that)
    w_att = df['D'] + (1-df['D'])*that/(1-that)
    f1 = sm.WLS(df['Y'][keep], sm.add_constant(df['D'])[keep], weights=w_ate[keep]).fit()
    f2 = sm.WLS(df['Y'][keep], sm.add_constant(df['D'])[keep], weights=w_att[keep]).fit()
    a_ate.append(f1.params.iloc[1]); a_att.append(f2.params.iloc[1]); t_att.append(tau[df['D']==1].mean())
    o = sm.OLS(df['Y'], sm.add_constant(df[['D','x1','x2']])).fit()
    ols_e.append(o.params.iloc[1])
print("   ipw_wls-logic ATE=%.3f (true 1.0)  ATT=%.3f (true %.3f)" % (np.mean(a_ate), np.mean(a_att), np.mean(t_att)))
print("   ols_vanilla-logic coef=%.3f (true ATE 1.0, true ATT %.3f)" % (np.mean(ols_e), np.mean(t_att)))

print("== C. dml_irm heavy trim: SE denominator uses full n ==")
ym = LinearRegression()
df, tau = dgp(3000, np.random.default_rng(7))
r1 = st.ate.dml.dml_irm(df.copy(),'splits',['x1','x2'],'Y','D',ym,tm,4,{'lower':0.35,'upper':0.65})
kept = ((r1['PScore']>=0.35)&(r1['PScore']<=0.65)).sum()
print("   kept %d of %d; SE uses sqrt(3000) -> SE understated by ~%.2fx" % (kept, len(df), np.sqrt(len(df)/kept)))

print("== D. delta_ols R-measure + formula: sqrt(corr) vs R2 on same fit ==")
rng = np.random.default_rng(2)
n=4000; w = rng.normal(size=(n,2)); W = 0.5*w[:,0]+rng.normal(size=n)
y = 1.0*W + w[:,0] + 0.5*w[:,1] + rng.normal(size=n)
dfo = pd.DataFrame({'W':W,'y':y,'w1':w[:,0],'w2':w[:,1]})
f = sm.OLS(dfo['y'], sm.add_constant(dfo[['W','w1','w2']])).fit()
pred = f.predict(sm.add_constant(dfo[['W','w1','w2']]))
print("   true R2 = %.3f ; code's sqrt(corr) = %.3f" % (f.rsquared, np.sqrt(np.corrcoef(pred,dfo['y'])[0,1])))
