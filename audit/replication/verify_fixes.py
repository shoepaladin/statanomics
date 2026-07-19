import numpy as np, pandas as pd
from sklearn.linear_model import LinearRegression, LogisticRegression
rng = np.random.default_rng(0)

print("="*70)
print("S-2/3  IPW Hajek ATT + influence-function SE (oracle propensity)")
print("="*70)
def dgp(n, rng):
    x1 = rng.normal(size=n); x2 = rng.normal(size=n)
    lin = 0.5*x1 - 0.5*x2
    e = 1/(1+np.exp(-lin))          # true propensity
    D = (rng.uniform(size=n) < e).astype(float)
    tau = 1.0 + 0.5*x1             # heterogeneous effect
    y0 = x1 + x2 + rng.normal(size=n)
    y1 = y0 + tau
    Y = D*y1 + (1-D)*y0
    # true ATT = E[tau|D=1]
    return pd.DataFrame({'x1':x1,'x2':x2,'D':D,'Y':Y,'e':e,'tau':tau,'y1':y1,'y0':y0})

def ipw_hajek(D,Y,e):
    w1,w0 = D/e,(1-D)/(1-e)
    mu1=np.sum(w1*Y)/np.sum(w1); mu0=np.sum(w0*Y)/np.sum(w0)
    ate=mu1-mu0
    psi_ate=w1*(Y-mu1)/np.mean(w1)-w0*(Y-mu0)/np.mean(w0)
    se_ate=np.std(psi_ate)/np.sqrt(len(Y))
    wt,wc=D,(1-D)*e/(1-e)
    m1=np.sum(wt*Y)/np.sum(wt); m0=np.sum(wc*Y)/np.sum(wc)
    att=m1-m0
    psi_att=wt*(Y-m1)/np.mean(wt)-wc*(Y-m0)/np.mean(wc)
    se_att=np.std(psi_att)/np.sqrt(len(Y))
    return ate,se_ate,att,se_att

ates,atts,se_ates,se_atts,true_atts=[],[],[],[],[]
for s in range(400):
    r=np.random.default_rng(s); d=dgp(3000,r)
    a,sa,t,st=ipw_hajek(d['D'].values,d['Y'].values,d['e'].values)
    ates.append(a); atts.append(t); se_ates.append(sa); se_atts.append(st)
    true_atts.append(d.loc[d.D==1,'tau'].mean())
print(f"true ATE=1.0    est ATE mean={np.mean(ates):.4f}  MCsd={np.std(ates):.4f}  meanSE={np.mean(se_ates):.4f}  ratio={np.mean(se_ates)/np.std(ates):.2f}")
print(f"true ATT~{np.mean(true_atts):.4f} est ATT mean={np.mean(atts):.4f}  MCsd={np.std(atts):.4f}  meanSE={np.mean(se_atts):.4f}  ratio={np.mean(se_atts)/np.std(atts):.2f}")

print()
print("="*70)
print("S-5  CCDDHNR ATT score: point estimate is DR (correct m0 vs misspecified m0)")
print("="*70)
def att_dml(D,Y,e,m0):
    p=D.mean(); ycr=Y-m0
    term=(D*ycr)/p - (e*(1-D)*ycr)/((1-e)*p)
    theta=np.mean(term)
    score=term - theta*D/p
    se=np.std(score)/np.sqrt(len(Y))
    return theta,se
res_correct,res_wrong,truth=[],[],[]
for s in range(400):
    r=np.random.default_rng(1000+s); d=dgp(3000,r)
    D,Y,e=d['D'].values,d['Y'].values,d['e'].values
    m0_correct=d['x1'].values+d['x2'].values       # true E[Y0|X]
    m0_wrong=np.zeros_like(Y)                       # badly misspecified (constant 0)
    tc,_=att_dml(D,Y,e,m0_correct)
    tw,_=att_dml(D,Y,e,m0_wrong)
    res_correct.append(tc); res_wrong.append(tw); truth.append(d.loc[d.D==1,'tau'].mean())
print(f"true ATT~{np.mean(truth):.4f}")
print(f"  correct m0 : ATT mean={np.mean(res_correct):.4f}  bias={np.mean(res_correct)-np.mean(truth):+.4f}")
print(f"  wrong  m0 : ATT mean={np.mean(res_wrong):.4f}  bias={np.mean(res_wrong)-np.mean(truth):+.4f}  (DR via correct e -> still ~unbiased)")

print()
print("="*70)
print("S-6/7  einsum prediction SE == sqrt(diag(Xd V Xd')) and column alignment")
print("="*70)
import statsmodels.api as sm
n=2000
a=rng.normal(size=n); tresid=rng.normal(size=n)
y=0.7+2*tresid+1.0*tresid*a+rng.normal(size=n,scale=0.5)
df=pd.DataFrame({'t':tresid,'t_xa':tresid*a,'a':a,'y':y,'ones':1.0,'cons':1.0})
covar=['t','t_xa']; het=['a']
X=sm.add_constant(df[covar]); fit=sm.OLS(df['y'],X).fit()
print("params index:",list(fit.params.index))       # expect [const, t, t_xa]
Xt=pd.concat([df[['ones']],df[het]],axis=1)          # [ones, a]
Xd=np.asarray(Xt,float)
V=np.asarray(fit.cov_params())[1:,1:]
se_einsum=np.sqrt(np.einsum('ij,jk,ik->i',Xd,V,Xd))
se_diag=np.sqrt(np.diag(Xd@V@Xd.T))
print("einsum==diag(XVX'):",np.allclose(se_einsum,se_diag))
cate=Xd@np.asarray(fit.params[1:],float)
true_cate=2.0+1.0*a
print(f"mean(CATE_hat - true_cate)={np.mean(cate-true_cate):+.4f} (should be ~0; intercept NOT leaking)")
print(f"fitted const={fit.params.iloc[0]:.4f} (old code added this to every CATE)")

print()
print("="*70)
print("P-5 / P-6  cyclic-shift permutation construction")
print("="*70)
def cyclic(L):
    tl=np.arange(L); out=[]
    for i in range(L):
        out.append(list(np.concatenate([tl[-1*(L-i):], tl[0:i]])))
    return out
for L in [5,21]:
    P=cyclic(L)
    allperm=all(sorted(p)==list(range(L)) for p in P)
    lastmoves=len(set(p[-1] for p in P))    # post index rotates through all positions
    print(f"L={L}: n_perms={len(P)} all_valid_perms={allperm} distinct_last_positions={lastmoves} (want {L})")
    if L==5: 
        for p in P: print("   ",p)
