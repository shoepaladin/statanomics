"""Repro 4b: quantify SDID ATT bias from the constrained intercept / wrong
penalty by re-running the same pipeline with a corrected omega program
(free intercept, penalty zeta^2 * T_pre * ||omega||^2 excluding intercept),
Monte Carlo over 40 seeds."""
import sys, warnings
warnings.filterwarnings('ignore')
sys.path.insert(0, '/home/user/statanomics/workingcode/panelmodels')
import numpy as np, pandas as pd
from scipy.optimize import fmin_slsqp
from panelib import sdid, dgp

T_pre, T_post, Nc = 12, 4, 20
T = T_pre + T_post
dd = {'treatment':'treated','date':'time','post':'post','unitid':'unit_id','outcome':'y'}

def make(seed):
    rng = np.random.default_rng(seed)
    ctrl = rng.normal(10, 1, (T, Nc)) + np.linspace(0, 2, T)[:, None]
    treat = ctrl.mean(axis=1) - 5.0 + rng.normal(0, .05, T)
    rows = []
    for j in range(Nc):
        for t in range(T):
            rows.append(dict(unit_id=f'C{j:02d}', time=t, treated=0, post=int(t>=T_pre), y=ctrl[t,j]))
    for t in range(T):
        rows.append(dict(unit_id='T0', time=t, treated=1, post=int(t>=T_pre), y=treat[t]+2.0*int(t>=T_pre)))
    return pd.DataFrame(rows)

def att_corrected(df):
    cl = dgp.clean_and_input_data(dataset=df, treatment='treated', unit_id='unit_id',
                                  date='time', post='post', outcome='y')
    Cpre, Cpst, Tpre = cl['C_pre'].values, cl['C_pst'].values, cl['T_pre'].values.flatten()
    zeta = sdid.comp_reg_parameter(cl['C_pre'], cl['C_pst'], cl['T_pre'], cl['T_pst'])
    def obj_o(w):
        return np.sum((w[0] + Cpre @ w[1:] - Tpre)**2) + zeta**2 * T_pre * np.sum(w[1:]**2)
    w0 = np.r_[0., np.ones(Nc)/Nc]
    w = fmin_slsqp(obj_o, w0, f_eqcons=lambda x: np.sum(x[1:])-1,
                   bounds=[(-np.inf, np.inf)]+[(0., np.inf)]*Nc, iter=50000, disp=False)
    def obj_l(la):
        pre_y = la[0] + np.dot(la[1:], Cpre)
        return np.sum((pre_y - Cpst.mean(axis=0))**2) + 1e-12*np.sum(la[1:]**2)
    l0 = np.r_[0., np.ones(T_pre)/T_pre]
    la = fmin_slsqp(obj_l, l0, f_eqcons=lambda x: np.sum(x[1:])-1,
                    bounds=[(-np.inf, np.inf)]+[(0., np.inf)]*T_pre, iter=50000, disp=False)
    sc_all = np.vstack([Cpre, Cpst]) @ w[1:]
    y_tr = np.r_[Tpre, cl['T_pst'].values.flatten()]
    delta = np.dot(la[1:], Tpre - sc_all[:T_pre])
    return float(np.mean(y_tr[T_pre:] - (sc_all[T_pre:] + delta)))

lib, fix = [], []
for s in range(40):
    df = make(s)
    r = sdid.twfe_sdid(data=df, data_dict=dd)
    lib.append(float(r['sdid']['coef_'].iloc[0]))
    fix.append(att_corrected(df))
lib, fix = np.array(lib), np.array(fix)
print(f"true ATT = 2.0, 40 seeds")
print(f"library twfe_sdid : mean={lib.mean():.4f}  bias={lib.mean()-2:.4f}  sd={lib.std():.4f}")
print(f"corrected weights : mean={fix.mean():.4f}  bias={fix.mean()-2:.4f}  sd={fix.std():.4f}")
