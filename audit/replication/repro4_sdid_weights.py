"""Repro 4: SDID weight estimation defects.
(a) intercept omega_0 / lambda_0 is bounded >= 0 (panelib.py:636,682) although
    Arkhangelsky et al. (2021) leave it free -> binds whenever the treated unit
    lies below the omega-weighted controls.
(b) lambda regularization uses zeta^2 * N_co * ||(lambda_0,lambda)||^2 instead of
    the paper's tiny uniqueness ridge (footnote: zeta = 1e-6 * sigma_hat) ->
    lambda shrunk toward zero/uniform; delta_pre miscomputed when the correct
    lambda should concentrate on late pre-periods."""
import sys, warnings
warnings.filterwarnings('ignore')
sys.path.insert(0, '/home/user/statanomics/workingcode/panelmodels')
import numpy as np, pandas as pd
from scipy.optimize import fmin_slsqp
from functools import partial
from panelib import sdid, dgp

rng = np.random.default_rng(3)
# panel: 20 controls, treated = mean(controls) - 5 (needs negative intercept)
T_pre, T_post, Nc = 12, 4, 20
T = T_pre + T_post
ctrl = rng.normal(10, 1, (T, Nc)) + np.linspace(0, 2, T)[:, None]
treat = ctrl.mean(axis=1) - 5.0 + rng.normal(0, .05, T)
rows = []
for j in range(Nc):
    for t in range(T):
        rows.append(dict(unit_id=f'C{j:02d}', time=t, treated=0, post=int(t>=T_pre), y=ctrl[t, j]))
for t in range(T):
    rows.append(dict(unit_id='T0', time=t, treated=1, post=int(t>=T_pre), y=treat[t] + 2.0*int(t>=T_pre)))
df = pd.DataFrame(rows)
cl = dgp.clean_and_input_data(dataset=df, treatment='treated', unit_id='unit_id',
                              date='time', post='post', outcome='y')

om = sdid.estimate_omega(cl['C_pre'], cl['C_pst'], cl['T_pre'], cl['T_pst'])
print(f"(a) omega_0 (intercept) = {om[0]:.6f}  -> bound at 0 binds; "
      f"true required intercept = -5")
resid = om[0] + cl['C_pre'].values @ om[1:] - cl['T_pre'].values.flatten()
print(f"    pre-fit RMSE of omega program = {np.sqrt(np.mean(resid**2)):.3f} (should be ~0.05)")

# (b) lambda: DGP where late pre-periods predict post (common AR trend jump)
lam = sdid.estimate_lambda(cl['C_pre'], cl['C_pst'], cl['T_pre'], cl['T_pst'])
lam_impl = lam[1:]

# same program with paper's tiny ridge and free intercept
def t_unit_paper(la, Cpre, Cpst, zeta_tiny):
    pre_y = la[0] + Cpre.values @ la[1:]
    pst_y = Cpst.values.mean(axis=0)
    # broadcast: pre_y is (Nc,) after transpose? paper: fit each control's post mean
    return np.sum((pre_y - pst_y)**2) + zeta_tiny**2 * np.sum(la[1:]**2)

Cpre_T = cl['C_pre'].T  # units x time -> match code's np.dot(lambda, data_control_pre)
def obj(la):
    pre_y = la[0] + np.dot(la[1:], cl['C_pre'].values)
    pst_y = cl['C_pst'].values.mean(axis=0)
    return np.sum((pre_y - pst_y)**2) + (1e-6)**2 * np.sum(la[1:]**2)
init = np.ones(T_pre+1)/T_pre
lam_paper = fmin_slsqp(obj, init, f_eqcons=lambda x: np.sum(x[1:])-1,
                       bounds=[(-np.inf, np.inf)] + [(0., np.inf)]*T_pre,
                       iter=50000, disp=False)
print("\n(b) lambda weights over pre-periods (should concentrate on periods "
      "resembling post; DGP has upward trend -> late periods)")
print("    implemented :", np.round(lam_impl, 3))
print("    paper ridge :", np.round(lam_paper[1:], 3))
print(f"    implied delta_pre gap, implemented vs paper: "
      f"{np.dot(lam_impl, treat[:T_pre]):.3f} vs {np.dot(lam_paper[1:], treat[:T_pre]):.3f}")

# ATT consequence
r = sdid.twfe_sdid(data=df, data_dict={'treatment':'treated','date':'time',
                                       'post':'post','unitid':'unit_id','outcome':'y'})
print(f"\n    SDID ATT (true = 2.0): {float(r['sdid']['coef_'].iloc[0]):.4f}")
