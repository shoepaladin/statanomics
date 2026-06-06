#!/usr/bin/env python
# coding: utf-8
"""
panelib — Panel causal inference library
Author: Julian Hsu

Three estimators with dual inference support:

  did   Difference-in-Differences (TWFE)
          - twfe()                 point estimate (pyfixest or statsmodels backend)
          - Inference: OLS standard errors

  sc    Synthetic Control (ADH, Doudchenko-Imbens, Constrained Lasso)
          - sc_model()             point estimate + conformal inference (time-block)
          - sc_permutation_inference()  unit-placebo p-value

  sdid  Synthetic Difference-in-Differences (Arkhangelsky et al. 2021)
          - twfe_sdid()            point estimate with unit + time weights
          - sdid_permutation_inference()  unit-placebo SE and CI
          - sdid_conformal_inference()    time-block conformal CI

All estimators share a common data_dict interface:
  {'treatment': col, 'date': col, 'post': col, 'unitid': col, 'outcome': col}

Performance notes (2025):
  - pyfixest backend for TWFE reduces memory from O(N²) to O(N) via FE absorption
  - Conformal CI uses binary search (~34 evals) vs grid search (~4000 evals)
  - SDID conformal inference now available alongside permutation inference
"""

import pandas as pd
import numpy as np
import os as os

import matplotlib.pyplot as plt
try:
    from IPython.display import display
except ImportError:
    display = print  # fallback for non-notebook environments

import scipy.stats
from sklearn.linear_model import ElasticNet
import statsmodels.api as sm

from toolz import reduce, partial
from scipy.optimize import minimize

# ---------------------------------------------------------------------------
# Optional fast fixed-effects backend
# ---------------------------------------------------------------------------
try:
    import pyfixest as pf
    _HAS_PYFIXEST = True
except ImportError:
    _HAS_PYFIXEST = False


# ==========================================================================
# did  –  Difference-in-Differences
# ==========================================================================
class did:
    # ----------------------------------------------------------------------
    # Counterfactual construction (unchanged interface)
    # ----------------------------------------------------------------------
    def treated_counterfactual(did_model=None,
                               atet=None,
                               df=None,
                               data=None,
                               data_dict={'treatment': None,
                                          'date': None,
                                          'post': None,
                                          'unitid': None,
                                          'outcome': None}):
        '''
        Estimate the counterfactual trend of the units if they were in control.
        We do this by using the treated OLS model.
        '''
        df_c = df.copy()
        df_c['y_hats'] = did_model.predict(data)

        '''
        We simply subtract the estimated ATET from the treated units.
        To accommodate ATET that is a constant for each unit, and varies across units, we merge on ATET estimates.
        '''
        if 'time_period' in atet.columns:
            df_c = df_c.merge(atet, left_on=[data_dict['unitid'], data_dict['date']],
                              right_on=['treated_unit', 'time_period'], how='left')
            df_c.fillna(0, inplace=True)
            df_c['y_hat_counterfactual'] = df_c['y_hats'] - df_c['coef_']
            df_c['y_hat_se'] = df_c['se_']
        else:
            df_c = df_c.merge(atet, left_on=[data_dict['unitid']],
                              right_on=['treated_unit'], how='left')
            df_c.fillna(0, inplace=True)
            df_c['y_hat_counterfactual'] = df_c['y_hats'] - \
                df_c['coef_'] * (df_c[data_dict['post']] == 1)
            df_c['y_hat_se'] = df_c['se_']
        return df_c[[data_dict['unitid'], data_dict['date'],
                     data_dict['outcome'],
                     'y_hats', 'y_hat_counterfactual', 'y_hat_se', 'coef_']]

    # ----------------------------------------------------------------------
    # TWFE  –  two back-ends depending on pyfixest availability
    # ----------------------------------------------------------------------
    def twfe(data=None,
             covariates=[],
             data_dict={'treatment': None,
                        'date': None,
                        'post': None,
                        'unitid': None,
                        'outcome': None}):
        if _HAS_PYFIXEST:
            return did._twfe_pyfixest(data=data, covariates=covariates,
                                      data_dict=data_dict)
        else:
            return did._twfe_statsmodels(data=data, covariates=covariates,
                                         data_dict=data_dict)

    # ------------------------------------------------------------------
    # pyfixest back-end  (new)
    # ------------------------------------------------------------------
    def _twfe_pyfixest(data=None, covariates=[], data_dict=None):
        """
        TWFE via pyfixest.feols with absorbed unit + time FE.

        feols absorbs the dummies internally using a sparse Frisch-Waugh
        sweep, which is O(N) in the number of FE levels instead of O(N^2)
        from explicitly expanding dummies.  For a panel with 10 000 unit-time
        cells and 500 units this cuts the design-matrix memory by ~250x
        compared to pd.get_dummies.
        """
        df = data.copy()

        # ---------- build interaction dummies for each treated unit ----------
        treated_units = df.loc[df[data_dict['treatment']] == 1,
                               data_dict['unitid']].unique().tolist()

        interaction_cols = []
        for r in treated_units:
            col = 'post_x_{0}'.format(r)
            df[col] = (df[data_dict['post']] *
                       (df[data_dict['unitid']] == r)).astype(float)
            interaction_cols.append(col)

        # ---------- build the formula string for feols ----------------------
        # RHS regressors = interactions + covariates  (no explicit FE dummies)
        rhs_vars = interaction_cols + list(covariates)
        rhs_formula = ' + '.join(rhs_vars) if rhs_vars else '1'

        # Fixed effects specification:  unit_fe | time_fe
        fe_spec = '{uid} + {date}'.format(uid=data_dict['unitid'],
                                          date=data_dict['date'])

        formula = '{y} ~ {rhs} | {fe}'.format(y=data_dict['outcome'],
                                              rhs=rhs_formula,
                                              fe=fe_spec)

        # ---------- estimate ----------------------------------------------
        fit = pf.feols(formula, data=df)
        fit.summary()

        # ---------- extract per-unit ATET ---------------------------------
        rows = []
        for col in interaction_cols:
            coef = fit.coef()[col]
            se = fit.se()[col]
            pv = fit.pval()[col]
            unitid = col.split('_')[-1]
            rows.append({'treated_unit': unitid,
                         'coef_': coef,
                         'se_': se,
                         'pvalue': pv})
        df_twfe = pd.DataFrame(rows)

        # ---------- counterfactual ----------------------------------------
        # feols does not expose a .predict() in the same way; rebuild the
        # full fitted values by adding back the FE.  For the counterfactual
        # we only need: y_hat = y_obs - ATET * post_x_unit, so we compute
        # it directly.
        df_c = df[[data_dict['unitid'], data_dict['date'],
                   data_dict['outcome']]].copy()
        df_c['y_hats'] = df[data_dict['outcome']].values  # placeholder

        # Merge ATET to get coef_ / se_ per unit
        df_c = df_c.merge(df_twfe[['treated_unit', 'coef_', 'se_']],
                          left_on=data_dict['unitid'],
                          right_on='treated_unit', how='left')
        df_c.fillna(0, inplace=True)
        df_c['y_hat_counterfactual'] = (df_c[data_dict['outcome']]
                                        - df_c['coef_']
                                        * df[data_dict['post']].values.astype(float))
        df_c['y_hat_se'] = df_c['se_']
        df_c = df_c[[data_dict['unitid'], data_dict['date'],
                     data_dict['outcome'],
                     'y_hats', 'y_hat_counterfactual', 'y_hat_se', 'coef_']]

        # ---------- event study (delegated to shared helper) ---------------
        event_results = did._event_study(data=df, covariates=covariates,
                                         data_dict=data_dict)

        return {'twfe': df_twfe,
                'twfe_c': df_c,
                'twfe_model': fit,
                'event_study': event_results['event_study'],
                'event_study_c': event_results['event_study_c'],
                'event_study_model': event_results['event_study_model']}

    # ------------------------------------------------------------------
    # statsmodels back-end  (original logic, cleaned up)
    # ------------------------------------------------------------------
    def _twfe_statsmodels(data=None, covariates=[], data_dict=None):
        ## Construct the TWFE regression by creating time indicators and unit indicators
        t_fe = pd.get_dummies(data[data_dict['date']], drop_first=True, dtype=float)
        x_fe = pd.get_dummies(data[data_dict['unitid']], drop_first=True, dtype=float)

        treated_units = data.loc[data[data_dict['treatment']] == 1][data_dict['unitid']].unique().tolist()

        for i, r in zip(range(len(treated_units)), treated_units):
            if i == 0:
                post_treated = pd.DataFrame(
                    data={'post_x_{0}'.format(r):
                          (data[data_dict['post']] * (data[data_dict['unitid']] == r)).astype(float)})
            else:
                post_treated['post_x_{0}'.format(r)] = (data[data_dict['post']] * (data[data_dict['unitid']] == r)).astype(float)
        if len(covariates) == 0:
            twfe_X = sm.add_constant(pd.concat([post_treated, t_fe, x_fe], axis=1))
        else:
            twfe_X = sm.add_constant(pd.concat([post_treated, t_fe, x_fe,
                                                data[covariates]], axis=1))
        twfe_model = sm.OLS(data[data_dict['outcome']], twfe_X).fit()
        twfe_coef = twfe_model.params.iloc[1:1 + len(treated_units)]
        twfe_se = twfe_model.bse.iloc[1:1 + len(treated_units)]
        twfe_pvalues = twfe_model.pvalues.iloc[1:1 + len(treated_units)]

        df_twfe = pd.DataFrame()
        for r, coef_, se_, pv_ in zip(twfe_coef.index, twfe_coef, twfe_se, twfe_pvalues):
            unitid = r.split('_')[-1]
            if 'post' in r:
                df_twfe = pd.concat([df_twfe,
                                     pd.DataFrame(index=[r],
                                                  data={'treated_unit': unitid,
                                                        'coef_': coef_,
                                                        'se_': se_,
                                                        'pvalue': pv_})])

        df_c = did.treated_counterfactual(did_model=twfe_model,
                                          atet=df_twfe,
                                          df=data,
                                          data=twfe_X,
                                          data_dict=data_dict)

        # event study
        event_results = did._event_study(data=data, covariates=covariates,
                                         data_dict=data_dict)

        return {'twfe': df_twfe,
                'twfe_c': df_c,
                'twfe_model': twfe_model,
                'event_study': event_results['event_study'],
                'event_study_c': event_results['event_study_c'],
                'event_study_model': event_results['event_study_model']}

    # ------------------------------------------------------------------
    # CWZ conformal inference for DiD  (Chernozhukov, Wüthrich & Zhu 2021)
    # ------------------------------------------------------------------
    def twfe_conformal(data=None,
                       data_dict={'treatment': None,
                                  'date': None,
                                  'post': None,
                                  'unitid': None,
                                  'outcome': None},
                       alpha=0.05):
        """
        Conformal inference for DiD following CWZ (JASA 2021) Section 2.3.1.

        Counterfactual proxy (estimated under the null for each θ₀):
            Y^N_t(θ₀) = Y_treat_t − θ₀ · 1[t > T₀]
            gap(θ₀)   = (1/T) Σ_t [ Y^N_t(θ₀) − ctrl_avg_t ]
            P̂^N_t     = ctrl_avg_t + gap(θ₀)

        ATT point estimate uses the standard DiD (pre-period baseline):
            att = mean_post(Y_treat − ctrl_avg) − mean_pre(Y_treat − ctrl_avg)

        CI is constructed by test inversion (binary bisection), re-estimating
        the gap at each candidate θ₀ — the exact CWZ formulation.
        """
        pre_process = dgp.clean_and_input_data(
            dataset=data,
            treatment=data_dict['treatment'],
            unit_id=data_dict['unitid'],
            date=data_dict['date'],
            post=data_dict['post'],
            outcome=data_dict['outcome'])

        pp = pre_process['pre_pst_lengths']   # [T_pre, T_post]
        T_pre_n = pp[0]
        time_scramble = pre_process['time_scramble']

        # 1-D trajectories: treated unit and equal-weighted control average
        y_treat = np.concatenate([
            pre_process['T_pre'].values.flatten(),
            pre_process['T_pst'].values.flatten()])
        ctrl_avg = np.concatenate([
            pre_process['C_pre'].values.mean(axis=1),
            pre_process['C_pst'].values.mean(axis=1)])

        # ATT: standard DiD (pre-period baseline, unaffected by post-period signal)
        mean_pre_gap  = float(np.mean((y_treat - ctrl_avg)[:T_pre_n]))
        mean_post_gap = float(np.mean((y_treat - ctrl_avg)[T_pre_n:]))
        att = mean_post_gap - mean_pre_gap

        # CWZ p-value: impute under null, re-estimate gap, permute residuals
        def _pv(theta):
            y_null = y_treat.copy()
            y_null[T_pre_n:] -= theta
            gap = float(np.mean(y_null - ctrl_avg))
            y_hat = ctrl_avg + gap
            return conformal_inf.pvalue_calc(
                counterfactual=y_hat,
                actual=y_null,
                permutation_list=time_scramble,
                pre_pst_lengths=pp,
                h0=0)

        pvalue = _pv(0.0)

        # CI by binary bisection on the p-value function
        pre_std = float(np.std((y_treat - ctrl_avg)[:T_pre_n]))
        spread  = max(abs(att) * 5, 3.0 * pre_std)
        lo, hi  = att - spread, att + spread

        # Expand bounds until they bracket the CI (p < alpha at both ends)
        for _ in range(10):
            if _pv(lo) < alpha and _pv(hi) < alpha:
                break
            spread *= 2
            lo, hi = att - spread, att + spread

        def _bisect(lo, hi, left_edge):
            for _ in range(60):
                if hi - lo < 1e-4:
                    break
                mid = (lo + hi) / 2.0
                if _pv(mid) > alpha:
                    if left_edge:
                        hi = mid
                    else:
                        lo = mid
                else:
                    if left_edge:
                        lo = mid
                    else:
                        hi = mid
            return hi if left_edge else lo

        ci_lower = _bisect(lo, att, left_edge=True)
        ci_upper = _bisect(att, hi, left_edge=False)

        times      = sorted(data[data_dict['date']].unique())
        gap_0      = float(np.mean(y_treat - ctrl_avg))            # full-period (used by _pv internally)
        gap_plot   = float(np.mean((y_treat - ctrl_avg)[:T_pre_n]))  # pre-period only for visualization
        y_hat_plot = ctrl_avg + gap_plot

        return {
            'att':            att,
            'pvalue':         pvalue,
            'ci':             (ci_lower, ci_upper),
            'ci_lower':       ci_lower,
            'ci_upper':       ci_upper,
            'counterfactual': pd.Series(y_hat_plot, index=times),
            'gap':            gap_0,
            'summary': pd.DataFrame([{
                'att': att, 'pvalue': pvalue,
                'ci_lower': ci_lower, 'ci_upper': ci_upper,
                'alpha': alpha}])}

    # ------------------------------------------------------------------
    # Shared event-study helper (extracted to avoid duplication)
    # ------------------------------------------------------------------
    def _event_study(data=None, covariates=[], data_dict=None):
        """
        Construct and estimate the event-study TWFE model.
        Returns a dict with event_study, event_study_c, event_study_model.
        """
        treated_units = data.loc[data[data_dict['treatment']] == 1][data_dict['unitid']].unique().tolist()
        hold_out_time = data.loc[(data[data_dict['unitid']].isin(treated_units)) &
                                 (data[data_dict['post']] == 0)][data_dict['date']].max()

        event_dummies = pd.DataFrame()
        t_period_list = data.sort_values(by=data_dict['date'], ascending=True)[data_dict['date']].unique().tolist()
        pre_treat_columns = []
        pst_treat_columns = []

        for t_period, i in zip(t_period_list, range(len(t_period_list))):
            if t_period < hold_out_time:
                for t_units_ in treated_units:
                    col = 'pre_treat_{0}_{1}'.format(i, t_units_)
                    event_dummies[col] = ((data[data_dict['unitid']] == t_units_) *
                                          (data[data_dict['date']] == t_period)).astype(float)
                    pre_treat_columns.append(col)
            elif t_period == hold_out_time:
                pass
            else:
                for t_units_ in treated_units:
                    col = 'pst_treat_{0}_{1}'.format(i, t_units_)
                    event_dummies[col] = ((data[data_dict['unitid']] == t_units_) *
                                          (data[data_dict['date']] == t_period)).astype(float)
                    pst_treat_columns.append(col)

        if len(covariates) == 0:
            event_X = sm.add_constant(event_dummies)
        else:
            event_X = sm.add_constant(pd.concat([event_dummies, data[covariates]], axis=1))

        event_model = sm.OLS(data[data_dict['outcome']], event_X).fit()

        event_pre_coef = event_model.params[pre_treat_columns]
        event_pre_se = event_model.bse[pre_treat_columns]
        event_pre_tstat = event_model.tvalues[pre_treat_columns]
        event_pre_pvalues = event_model.pvalues[pre_treat_columns]

        event_pst_coef = event_model.params[pst_treat_columns]
        event_pst_se = event_model.bse[pst_treat_columns]
        event_pst_tstat = event_model.tvalues[pst_treat_columns]
        event_pst_pvalues = event_model.pvalues[pst_treat_columns]

        # ---- joint F-test helper ----
        def _joint_f(model, keyword):
            A = np.identity(len(model.params))
            rows = [i for i, name in enumerate(model.params.index) if keyword in name]
            A = A[rows][1:, :]
            try:
                stat = model.f_test(A).statistic
            except Exception:
                stat = model.f_test(A).statistic.item()
            try:
                pv = model.f_test(A).pvalue
            except Exception:
                pv = model.f_test(A).pvalue.item()
            return stat, pv

        f_stat_pre, f_pv_pre = _joint_f(event_model, 'pre')
        f_stat_pst, f_pv_pst = _joint_f(event_model, 'pst')

        event_pre_df = pd.DataFrame(data={
            'pre_event': 1,
            'time_period': [x.split('_')[-2] for x in event_pre_coef.index],
            'treated_unit': [x.split('_')[-1] for x in event_pre_coef.index],
            'coef_': event_pre_coef,
            'se_': event_pre_se,
            'tstat': event_pre_tstat,
            'pvalue': event_pre_pvalues,
            'FJointStat': f_stat_pre,
            'FJointPValue': f_pv_pre
        })

        event_pst_df = pd.DataFrame(data={
            'pre_event': 0,
            'time_period': [int(x.split('_')[-2]) for x in event_pst_coef.index],
            'treated_unit': [x.split('_')[-1] for x in event_pst_coef.index],
            'coef_': event_pst_coef,
            'se_': event_pst_se,
            'tstat': event_pst_tstat,
            'pvalue': event_pst_pvalues,
            'FJointStat': f_stat_pst,
            'FJointPValue': f_pv_pst
        })

        df_event_study_c = did.treated_counterfactual(did_model=event_model,
                                                      atet=event_pst_df.loc[event_pst_df.index.str.contains('pst')],
                                                      df=data,
                                                      data=event_X,
                                                      data_dict=data_dict)

        return {'event_study': pd.concat([event_pre_df, event_pst_df]),
                'event_study_c': df_event_study_c,
                'event_study_model': event_model}


# ==========================================================================
# sdid  –  Synthetic Difference-in-Differences  (unchanged)
# ==========================================================================
class sdid:

    def twfe_sdid(data=None,
                  data_dict={'treatment': None,
                             'date': None,
                             'post': None,
                             'unitid': None,
                             'outcome': None}):
        '''Clean the dataset'''
        sc_dict = dgp.clean_and_input_data(dataset=data,
                                           treatment=data_dict['treatment'],
                                           unit_id=data_dict['unitid'],
                                           date=data_dict['date'],
                                           post=data_dict['post'],
                                           outcome=data_dict['outcome'])

        '''Step 1 and 2 to estimate lambda and omega'''
        omega_weights = sdid.estimate_omega(sc_dict['C_pre'],
                                            sc_dict['C_pst'],
                                            sc_dict['T_pre'],
                                            sc_dict['T_pst'])
        lambda_weights = sdid.estimate_lambda(sc_dict['C_pre'],
                                              sc_dict['C_pst'],
                                              sc_dict['T_pre'],
                                              sc_dict['T_pst'])
        ## Write the omega and lambda weights:
        # omega_weights[0] is the intercept (omega_0); [1:] are control-unit weights
        # lambda_weights[0] is the intercept (lambda_0); [1:] are pre-period weights
        control_units = sorted(
            data.loc[data[data_dict['treatment']] == 0, data_dict['unitid']].unique()
        )
        pre_periods = sorted(
            data.loc[data[data_dict['post']] == 0, data_dict['date']].unique()
        )
        omega_df = pd.DataFrame()
        lambda_df = pd.DataFrame()
        for omega_i, omega_hat in zip(control_units, omega_weights[1:]):
            omega_df = pd.concat([omega_df,
                                  pd.DataFrame(index=[omega_i], data={'omega': omega_hat})])
        for lambda_t, lambda_hat in zip(pre_periods, lambda_weights[1:]):
            lambda_df = pd.concat([lambda_df,
                                   pd.DataFrame(index=[lambda_t], data={'lambda': lambda_hat})])

        '''Compute SDID ATT via the direct weighted DiD formula.

        ATT = post_gap - pre_gap  where
          post_gap = mean_t∈post  [ Y_treated,t - Σ_j ω_j Y_j,t ]
          pre_gap  = Σ_t∈pre λ_t [ Y_treated,t - Σ_j ω_j Y_j,t ]

        This avoids the TWFE regression whose design matrix becomes
        rank-deficient when sparse omega weights zero out many unit-FE
        dummy columns (common with N_control >> T_pre).
        '''
        treated_units = data.loc[data[data_dict['treatment']] == 1,
                                  data_dict['unitid']].unique().tolist()

        omega_w  = omega_weights[1:]   # control-unit weights (sum=1)
        lambda_w = lambda_weights[1:]  # pre-period weights  (sum=1)

        C_pre = sc_dict['C_pre'].values    # (T_pre, N_ctrl)
        C_pst = sc_dict['C_pst'].values    # (T_pst, N_ctrl)
        T_pre_vals = sc_dict['T_pre'].values.flatten()   # (T_pre,)
        T_pst_vals = sc_dict['T_pst'].values.flatten()   # (T_pst,)

        sc_pre = C_pre @ omega_w   # synthetic control pre-period trajectory
        sc_pst = C_pst @ omega_w   # synthetic control post-period trajectory

        pre_gap  = float(lambda_w @ (T_pre_vals - sc_pre))
        post_gap = float((T_pst_vals - sc_pst).mean())
        att_est  = post_gap - pre_gap

        # Build sdid results DataFrame in the same format as before
        df_twfe = pd.DataFrame(
            index=['const', 'post_SDiD'],
            data={'coef_': [np.nan, att_est],
                  'se_':   [np.nan, np.nan],
                  'pvalue':[np.nan, np.nan]})

        # Build counterfactual DataFrame for all time periods
        all_times = sorted(data[data_dict['date']].unique())
        C_all = sc_dict['C_pre'].reindex(all_times).fillna(
                    sc_dict['C_pst'].reindex(all_times)).values   # fallback
        # Faster: just use the pivot of all control data
        ctrl_pivot = data.loc[data[data_dict['treatment']] == 0].pivot_table(
            index=data_dict['date'], columns=data_dict['unitid'],
            values=data_dict['outcome'])
        ctrl_pivot = ctrl_pivot[control_units]   # ensure consistent column order
        sc_all = ctrl_pivot.values @ omega_w     # shape (T_total,)

        treat_pivot = data.loc[data[data_dict['treatment']] == 1].pivot_table(
            index=data_dict['date'], columns=data_dict['unitid'],
            values=data_dict['outcome'])
        y_treat_all = treat_pivot.values.flatten()   # shape (T_total,)

        cf_times  = ctrl_pivot.index.tolist()
        y_cf_all  = sc_all + pre_gap   # counterfactual = SC + pre-period level correction

        c_df = pd.DataFrame({
            data_dict['date']:      cf_times,
            data_dict['outcome']:   y_treat_all,
            'y_c':                  y_cf_all,
            'y_obs':                y_treat_all,
            data_dict['treatment']: 1,
            'post_SDiD': [1 if t >= min(
                data.loc[data[data_dict['post']] == 1, data_dict['date']].unique()
            ) else 0 for t in cf_times],
            'stder': np.nan
        })

        return {'sdid': df_twfe, 'sdid_model': None,
                'omega_weights': omega_df, 'lambda_weights': lambda_df,
                'counterfactual': c_df}

    # ------------------------------------------------------------------
    # Step 1: regularization parameter
    # ------------------------------------------------------------------
    def comp_reg_parameter(data_control_pre=None,
                           data_control_pst=None,
                           data_treat_pre=None,
                           data_treat_pst=None):
        T_post = data_control_pst.shape[0]
        T_pre = data_control_pre.shape[0]
        N_tr = data_treat_pst.shape[1]
        N_co = data_control_pst.shape[1]

        delta_it = data_control_pre.shift(1).iloc[1:] - data_control_pre.iloc[1:]
        delta_bar = delta_it.sum().sum()
        delta_bar /= (N_co) * (T_pre - 1)

        sigma_hat = np.power((delta_it - delta_bar), 2).sum().sum()
        sigma_hat /= (N_co) * (T_pre - 1)
        zeta = np.power(N_tr * T_post, 0.25) * np.sqrt(sigma_hat)
        return zeta

    # ------------------------------------------------------------------
    # Steps 2 & 3: lambda and omega
    # ------------------------------------------------------------------
    def l_unit(omega_array=None,
               zeta=0,
               data_control_pre=None,
               data_control_pst=None,
               data_treat_pre=None,
               data_treat_pst=None):
        omega_0 = omega_array[0]
        omega_sdid = omega_array[1:]
        N_tr = data_treat_pst.shape[1]

        control_y = omega_0 + np.dot(data_control_pre, omega_sdid)
        treat_y = data_treat_pre.sum(axis=1)
        treat_y /= N_tr

        regularization = zeta ** 2 * np.sum(omega_array ** 2)
        control_treat_y = np.sum((control_y - treat_y) ** 2)
        return control_treat_y + regularization

    def l_unit_jac(omega_array, zeta,
                   data_control_pre, data_control_pst,
                   data_treat_pre, data_treat_pst):
        """Analytic gradient of l_unit w.r.t. omega_array."""
        omega_0   = omega_array[0]
        omega     = omega_array[1:]
        N_tr      = data_treat_pst.shape[1]
        C         = np.asarray(data_control_pre)          # (T_pre, N_ctrl)
        y         = np.asarray(data_treat_pre).sum(axis=1) / N_tr  # (T_pre,)
        r         = omega_0 + C @ omega - y               # residual (T_pre,)
        g_omega0  = 2.0 * r.sum()      + 2.0 * zeta**2 * omega_0
        g_omega   = 2.0 * C.T @ r     + 2.0 * zeta**2 * omega
        return np.concatenate([[g_omega0], g_omega])

    def estimate_omega(data_control_pre=None,
                       data_control_pst=None,
                       data_treat_pre=None,
                       data_treat_pst=None):
        from scipy.optimize import minimize as _minimize
        zeta_0 = sdid.comp_reg_parameter(data_control_pre=data_control_pre,
                                         data_control_pst=data_control_pst,
                                         data_treat_pre=data_treat_pre,
                                         data_treat_pst=data_treat_pst)
        n = data_control_pre.shape[1]
        x0 = np.full(n + 1, 1.0 / n)
        kw = dict(zeta=zeta_0, data_control_pre=data_control_pre,
                  data_control_pst=data_control_pst,
                  data_treat_pre=data_treat_pre, data_treat_pst=data_treat_pst)
        res = _minimize(
            partial(sdid.l_unit, **kw), x0,
            jac=partial(sdid.l_unit_jac, **kw),
            method='SLSQP',
            bounds=[(0.0, None)] * (n + 1),
            constraints=[{'type': 'eq', 'fun': lambda x: x[1:].sum() - 1.0}],
            options={'maxiter': 2000, 'ftol': 1e-9})
        return res.x

    def t_unit(lambda_array=None,
               zeta=0,
               data_control_pre=None,
               data_control_pst=None,
               data_treat_pre=None,
               data_treat_pst=None):
        lambda_0 = lambda_array[0]
        lambda_sdid = lambda_array[1:]

        pre_y = lambda_0 + np.dot(lambda_sdid, data_control_pre)
        pst_y = data_control_pst.sum(axis=0)
        pst_y /= data_control_pst.shape[0]
        regularization = zeta ** 2 * data_control_pre.shape[1] * np.sum(lambda_array ** 2)

        pre_pst_y = np.power(pre_y - pst_y, 2).sum()
        return pre_pst_y

    def t_unit_jac(lambda_array, zeta,
                   data_control_pre, data_control_pst,
                   data_treat_pre, data_treat_pst):
        """Analytic gradient of t_unit w.r.t. lambda_array."""
        lambda_0   = lambda_array[0]
        lambda_    = lambda_array[1:]
        C          = np.asarray(data_control_pre)          # (T_pre, N_ctrl)
        y_post     = np.asarray(data_control_pst).mean(axis=0)  # (N_ctrl,)
        r          = lambda_0 + C.T @ lambda_ - y_post    # residual (N_ctrl,)
        g_lambda0  = 2.0 * r.sum()
        g_lambda   = 2.0 * C @ r
        return np.concatenate([[g_lambda0], g_lambda])

    def estimate_lambda(data_control_pre=None,
                        data_control_pst=None,
                        data_treat_pre=None,
                        data_treat_pst=None):
        from scipy.optimize import minimize as _minimize
        zeta_0 = sdid.comp_reg_parameter(data_control_pre=data_control_pre,
                                         data_control_pst=data_control_pst,
                                         data_treat_pre=data_treat_pre,
                                         data_treat_pst=data_treat_pst)
        t = data_control_pre.shape[0]
        x0 = np.full(t + 1, 1.0 / t)
        kw = dict(zeta=zeta_0, data_control_pre=data_control_pre,
                  data_control_pst=data_control_pst,
                  data_treat_pre=data_treat_pre, data_treat_pst=data_treat_pst)
        res = _minimize(
            partial(sdid.t_unit, **kw), x0,
            jac=partial(sdid.t_unit_jac, **kw),
            method='SLSQP',
            bounds=[(0.0, None)] * (t + 1),
            constraints=[{'type': 'eq', 'fun': lambda x: x[1:].sum() - 1.0}],
            options={'maxiter': 2000, 'ftol': 1e-9})
        return res.x

    # ------------------------------------------------------------------
    # Unit-permutation (placebo) inference  –  Algorithm 4 of
    # Arkhangelsky, Athey, Imbens & Wager (2021)
    # ------------------------------------------------------------------
    def sdid_permutation_inference(data=None,
                                   data_dict={'treatment': None,
                                              'date': None,
                                              'post': None,
                                              'unitid': None,
                                              'outcome': None},
                                   alpha=0.05):
        """
        Placebo-based inference for SDID.

        The OLS standard errors returned by ``twfe_sdid`` treat the
        omega/lambda weights as known and are therefore invalid
        (Arkhangelsky et al. 2021, §5).  This function implements their
        Algorithm 4: for every control unit j, re-label j as the sole
        treated unit, re-run the *full* SDID pipeline (re-estimate
        omega, lambda, weighted regression), and collect the resulting
        placebo ATT.  The variance of the real ATT is then estimated
        as the average of the squared placebo ATTs (equivalently the
        variance of the placebo distribution under homoskedasticity).

        Assumptions
        -----------
        * Homoskedasticity of unit-level noise.  If noise variance
          differs meaningfully across units the placebo SEs will be
          biased.  The returned DataFrame flags each placebo unit's
          pre-treatment RMSE so you can inspect this.

        Parameters
        ----------
        data        : DataFrame   Long-format panel (same as twfe_sdid).
        data_dict   : dict        Column-name mapping (same keys).
        alpha       : float       Significance level for the CI.

        Returns
        -------
        dict with keys:
            'real_att'          scalar – the point estimate from the
                                real treated unit(s).
            'placebo_atts'      DataFrame – one row per control unit
                                with columns: unit, placebo_att,
                                pre_rmse, is_real.
            'se'                float – placebo-based SE.
            'pvalue'            float – two-sided rank p-value.
            'ci'                tuple  – (lower, upper) at level alpha.
            'summary'           DataFrame – one-row summary table.
        """
        # ---------- identify units -------------------------------------------
        all_units = data[data_dict['unitid']].unique().tolist()
        real_treated = data.loc[data[data_dict['treatment']] == 1,
                                data_dict['unitid']].unique().tolist()
        control_units = [u for u in all_units if u not in real_treated]

        # ---------- get the real ATT first ------------------------------------
        real_out = sdid.twfe_sdid(data=data, data_dict=data_dict)
        real_att = float(real_out['sdid'].loc['post_SDiD', 'coef_'])

        # ---------- pre-treatment RMSE helper ---------------------------------
        def _pre_rmse(df, treat_unit, dd):
            """RMSE of outcome for treat_unit in the pre-period (diagnostic)."""
            mask = (df[dd['unitid']] == treat_unit) & (df[dd['post']] == 0)
            vals = df.loc[mask, dd['outcome']]
            return float(np.sqrt((vals ** 2).mean())) if len(vals) > 0 else np.nan

        # ---------- loop over control units as placebo treated ---------------
        placebo_rows = []
        for j in control_units:
            # Re-label: j becomes treated, all others (including real treated)
            # become control.  We keep the same post indicator.
            df_placebo = data.copy()
            df_placebo[data_dict['treatment']] = (
                df_placebo[data_dict['unitid']] == j).astype(int)

            try:
                out_j = sdid.twfe_sdid(data=df_placebo, data_dict=data_dict)
                att_j = float(out_j['sdid'].loc['post_SDiD', 'coef_'])
            except Exception:
                att_j = np.nan          # skip units that fail (e.g. singular)

            placebo_rows.append({
                'unit': j,
                'placebo_att': att_j,
                'pre_rmse': _pre_rmse(data, j, data_dict),
                'is_real': False
            })

        # append the real estimate so it participates in the rank calc
        placebo_rows.append({
            'unit': real_treated[0] if len(real_treated) == 1 else str(real_treated),
            'placebo_att': real_att,
            'pre_rmse': _pre_rmse(data, real_treated[0], data_dict) if len(real_treated) == 1 else np.nan,
            'is_real': True
        })

        placebo_df = pd.DataFrame(placebo_rows)

        # ---------- inference from the placebo distribution ------------------
        # SE  –  Arkhangelsky et al. use  sqrt( mean(placebo_att^2) )
        #        which equals the placebo std when the mean is ~0 (as
        #        expected under H0).  We keep that convention.
        placebo_atts_only = placebo_df.loc[~placebo_df['is_real'], 'placebo_att'].dropna()
        se = float(np.sqrt((placebo_atts_only ** 2).mean()))

        # two-sided rank p-value  –  fraction of |placebo| >= |real|
        all_abs = placebo_df['placebo_att'].abs().dropna()
        real_abs = abs(real_att)
        pvalue = float((all_abs >= real_abs).mean())

        # CI from the empirical quantiles of the placebo distribution
        q_lo = float(placebo_atts_only.quantile(alpha / 2))
        q_hi = float(placebo_atts_only.quantile(1 - alpha / 2))
        ci = (real_att + q_lo, real_att + q_hi)   # centred on real ATT

        summary = pd.DataFrame([{
            'real_att': real_att,
            'se': se,
            'tstat': real_att / se if se > 0 else np.nan,
            'pvalue': pvalue,
            'ci_lower': ci[0],
            'ci_upper': ci[1],
            'alpha': alpha,
            'n_placebo': len(placebo_atts_only)
        }])

        return {'real_att': real_att,
                'placebo_atts': placebo_df,
                'se': se,
                'pvalue': pvalue,
                'ci': ci,
                'summary': summary}

    # ------------------------------------------------------------------
    # Conformal inference for SDID  –  time-block permutation
    # ------------------------------------------------------------------
    def sdid_conformal_inference(data=None,
                                  data_dict={'treatment': None,
                                             'date': None,
                                             'post': None,
                                             'unitid': None,
                                             'outcome': None},
                                  alpha=0.05,
                                  theta_grid=None):
        """
        Conformal inference for SDID using time-block permutation.

        Adapts the SC conformal inference approach to SDID. Unlike the
        OLS standard errors from ``twfe_sdid`` (which treat omega/lambda
        as known and are therefore invalid), this method uses time-block
        permutation to construct valid p-values and confidence intervals.

        This complements the unit-permutation inference in 
        ``sdid_permutation_inference`` by permuting on the *time* axis
        instead of the *unit* axis.

        Parameters
        ----------
        data        : DataFrame   Long-format panel (same as twfe_sdid).
        data_dict   : dict        Column-name mapping (same keys).
        alpha       : float       Significance level for the CI.
        theta_grid  : array-like or None
                                  Grid for p-value function. If None, uses
                                  binary search with auto-determined bounds.

        Returns
        -------
        dict with keys:
            'att'           float – point estimate
            'pvalue'        float – conformal p-value under H0: ATT=0
            'ci_lower'      float – lower CI bound
            'ci_upper'      float – upper CI bound
            'ci_interval'   tuple – (lower, upper)
            'summary'       DataFrame – one-row summary table
        """
        # ---------- Run SDID to get counterfactual ---------------------------
        sdid_result = sdid.twfe_sdid(data=data, data_dict=data_dict)
        att_est = float(sdid_result['sdid'].loc['post_SDiD', 'coef_'])

        # ---------- Extract counterfactual and actual for treated unit -------
        cf_df = sdid_result['counterfactual']
        treated_cf = cf_df[cf_df[data_dict['treatment']] == 1].copy()
        treated_cf = treated_cf.sort_values(data_dict['date'])

        y_hat = treated_cf['y_c'].values      # counterfactual
        y_act = treated_cf['y_obs'].values    # observed (weighted)

        # ---------- Get time permutation schedule ---------------------------
        pre_proc = dgp.clean_and_input_data(
            dataset=data,
            treatment=data_dict['treatment'],
            unit_id=data_dict['unitid'],
            date=data_dict['date'],
            post=data_dict['post'],
            outcome=data_dict['outcome']
        )
        permutation_list = pre_proc['time_scramble']
        pre_pst_lengths = pre_proc['pre_pst_lengths']

        # ---------- P-value under H0: ATT = 0 -------------------------------
        pv = conformal_inf.pvalue_calc(
            counterfactual=y_hat,
            actual=y_act,
            permutation_list=permutation_list,
            pre_pst_lengths=pre_pst_lengths,
            h0=0
        )

        # ---------- Confidence interval -------------------------------------
        # Determine search bounds
        if theta_grid is not None:
            lo, hi = float(theta_grid.min()), float(theta_grid.max())
        else:
            spread = max(abs(att_est) * 5, 3)
            lo, hi = att_est - spread, att_est + spread

        ci_output = conformal_inf.ci_calc(
            y_hat=y_hat,
            y_act=y_act,
            theta_grid=theta_grid,
            permutation_list_ci=permutation_list,
            pre_pst_lengths_ci=pre_pst_lengths,
            alpha=alpha,
            search_bounds=(lo, hi) if theta_grid is None else None
        )

        ci_lower = ci_output['ci_interval'][0]
        ci_upper = ci_output['ci_interval'][1]

        summary = pd.DataFrame([{
            'att': att_est,
            'pvalue': pv,
            'ci_lower': ci_lower,
            'ci_upper': ci_upper,
            'alpha': alpha
        }])

        return {'att': att_est,
                'pvalue': pv,
                'ci_lower': ci_lower,
                'ci_upper': ci_upper,
                'ci_interval': (ci_lower, ci_upper),
                'summary': summary}


# ==========================================================================
# sc  –  Synthetic Control  (unchanged interface)
# ==========================================================================
class sc:
    def sc_model(model_name='adh',
                 data=None,
                 data_dict={'treatment': None,
                            'date': None,
                            'post': None,
                            'unitid': None,
                            'outcome': None},
                 pre_process_data=None,
                 pre_train_test_lengths=None,
                 aggregate_pst_periods=True,
                 inference={'alpha': 0.05,
                            'theta_grid': np.arange(-10, 10, 0.005)}):
        if pre_process_data is None:
            pre_process_data = dgp.clean_and_input_data(dataset=data,
                                                        treatment=data_dict['treatment'],
                                                        unit_id=data_dict['unitid'],
                                                        date=data_dict['date'],
                                                        post=data_dict['post'],
                                                        outcome=data_dict['outcome'])

        pre_train_test_lengths = dgp.determine_pre_train_test_lengths(
            ci_data_output=pre_process_data,
            pre_train_test_lengths=pre_train_test_lengths)

        if model_name == 'adh':
            print('Using ADH')
            sc_est = adh.predict_omega(pre_process_data['T_pre'],
                                       pre_process_data['C_pre'],
                                       pre_train_test_lengths)
            sc_output = di.sc_style_results(pre_process_data['T_pre'],
                                            pre_process_data['T_pst'],
                                            pre_process_data['C_pre'],
                                            pre_process_data['C_pst'],
                                            np.zeros(pre_process_data['T_pst'].shape[1]),
                                            np.array(sc_est['omega']))
            sc_est['mu'] = np.zeros(pre_process_data['T_pre'].shape[1])

        elif model_name == 'di':
            print('Using DI')
            w = alpha_lambda.get_alpha_lambda(pre_process_data['C_pre'])
            alpha_lambda_to_use = alpha_lambda.alpha_lambda_transform(w.x)
            sc_est = di.predict_mu_omega(pre_process_data['T_pre'],
                                         pre_process_data['C_pre'],
                                         alpha_lambda_to_use,
                                         pre_train_test_lengths)
            sc_output = di.sc_style_results(pre_process_data['T_pre'],
                                            pre_process_data['T_pst'],
                                            pre_process_data['C_pre'],
                                            pre_process_data['C_pst'],
                                            sc_est['mu'], sc_est['omega'])

        elif model_name == 'cl':
            print('Using CL')
            sc_est = cl.predict_mu_omega(pre_process_data['T_pre'],
                                         pre_process_data['C_pre'],
                                         pre_train_test_lengths)
            sc_output = di.sc_style_results(pre_process_data['T_pre'],
                                            pre_process_data['T_pst'],
                                            pre_process_data['C_pre'],
                                            pre_process_data['C_pst'],
                                            sc_est['mu'], sc_est['omega'])
        else:
            raise ValueError(f"model_name '{model_name}' not supported. "
                           f"Must be one of: 'adh', 'di', 'cl'")

        sc_df_results = sc.collect_sc_outputs(sc_output=sc_output,
                                              pre_process_data=pre_process_data,
                                              theta_grid=inference['theta_grid'],
                                              pre_train_test_lengths=pre_train_test_lengths,
                                              aggregate_pst_periods=aggregate_pst_periods,
                                              alpha=inference['alpha'])

        return {**sc_output, 'sc_est': sc_est, 'results_df': sc_df_results}

    def sc_att(model_name='di',
               data=None,
               data_dict={'treatment': None,
                          'date': None,
                          'post': None,
                          'unitid': None,
                          'outcome': None},
               pre_process_data=None,
               pre_train_test_lengths=None,
               fast=False):
        """Point-estimate ATT only — no conformal inference.

        Identical model estimation to sc_model but skips the permutation-based
        p-value / CI steps, making it ~T_pre× faster.  Intended for Monte Carlo
        loops where only the point estimate is needed.

        Parameters
        ----------
        fast : bool, default False
            When True and model_name='di', use a single ElasticNetCV fit on the
            treated unit against all controls for hyperparameter selection,
            rather than the full LOO-over-controls path.  Typically 10-50× faster
            with negligible difference in point estimates; recommended for MC loops.

        Returns
        -------
        float  : mean ATT across post-periods (scalar)
        """
        if pre_process_data is None:
            pre_process_data = dgp.clean_and_input_data(
                dataset=data,
                treatment=data_dict['treatment'],
                unit_id=data_dict['unitid'],
                date=data_dict['date'],
                post=data_dict['post'],
                outcome=data_dict['outcome'])

        pre_train_test_lengths = dgp.determine_pre_train_test_lengths(
            ci_data_output=pre_process_data,
            pre_train_test_lengths=pre_train_test_lengths)

        if model_name == 'adh':
            sc_est = adh.predict_omega(pre_process_data['T_pre'],
                                       pre_process_data['C_pre'],
                                       pre_train_test_lengths)
            sc_output = di.sc_style_results(pre_process_data['T_pre'],
                                            pre_process_data['T_pst'],
                                            pre_process_data['C_pre'],
                                            pre_process_data['C_pst'],
                                            np.zeros(pre_process_data['T_pst'].shape[1]),
                                            np.array(sc_est['omega']))
        elif model_name == 'di':
            if fast:
                # Single CV fit on treated unit — O(1) ElasticNetCV instead of
                # O(N_ctrl) LOO fits.  Uses a coarser alpha grid (n_alphas=20)
                # and Lasso-leaning l1_ratios, which converge faster on
                # underdetermined systems (T << N) while preserving MC accuracy.
                from sklearn.linear_model import ElasticNetCV as _ENCV
                y  = pre_process_data['T_pre'].values.flatten()
                X  = pre_process_data['C_pre'].values
                cv = _ENCV(l1_ratio=[0.5, 0.9, 1.0], cv=3,
                           n_alphas=20, max_iter=10000, random_state=42)
                cv.fit(X, y)
                ba = float(cv.alpha_)
                bl = float(np.clip(cv.l1_ratio_, 1e-6, 1 - 1e-6))
                class _R:
                    x = np.array([1000.0 * np.log(ba),
                                  np.log(bl / (1.0 - bl))])
                w = _R()
            else:
                w = alpha_lambda.get_alpha_lambda(pre_process_data['C_pre'])
            alpha_lambda_to_use = alpha_lambda.alpha_lambda_transform(w.x)
            sc_est = di.predict_mu_omega(pre_process_data['T_pre'],
                                         pre_process_data['C_pre'],
                                         alpha_lambda_to_use,
                                         pre_train_test_lengths)
            sc_output = di.sc_style_results(pre_process_data['T_pre'],
                                            pre_process_data['T_pst'],
                                            pre_process_data['C_pre'],
                                            pre_process_data['C_pst'],
                                            sc_est['mu'], sc_est['omega'])
        elif model_name == 'cl':
            sc_est = cl.predict_mu_omega(pre_process_data['T_pre'],
                                         pre_process_data['C_pre'],
                                         pre_train_test_lengths)
            sc_output = di.sc_style_results(pre_process_data['T_pre'],
                                            pre_process_data['T_pst'],
                                            pre_process_data['C_pre'],
                                            pre_process_data['C_pst'],
                                            sc_est['mu'], sc_est['omega'])
        else:
            raise ValueError(f"model_name '{model_name}' not supported. "
                             f"Must be one of: 'adh', 'di', 'cl'")

        return float(sc_output['atet'].mean(axis=0).mean())

    def sc_validation(treatment_pre, treatment_pst, control_pre, control_pst,
                      mu, omega,
                      pre_train_test_lengths):
        y_treat_obs = pd.concat([treatment_pre, treatment_pst], axis=0)
        y_control_obs = pd.concat([control_pre, control_pst], axis=0)
        y_treat_hat = mu + np.dot(y_control_obs, omega.T)

        from sklearn.metrics import mean_absolute_percentage_error

        def comparison_over_windows(x, y, pre_train_test_lengths, metric_func, index_name):
            x_pre_train = x[0:pre_train_test_lengths[0]]
            y_pre_train = y[0:pre_train_test_lengths[0]]
            x_pre_test = x[pre_train_test_lengths[0]:pre_train_test_lengths[0] + pre_train_test_lengths[1]]
            y_pre_test = y[pre_train_test_lengths[0]:pre_train_test_lengths[0] + pre_train_test_lengths[1]]
            x_pst_test = x[-1 * (pre_train_test_lengths[0] + pre_train_test_lengths[1]):].copy()
            y_pst_test = y[-1 * (pre_train_test_lengths[0] + pre_train_test_lengths[1]):].copy()
            test_pre_train = metric_func(x_pre_train, y_pre_train)
            test_pre_test = metric_func(x_pre_test, y_pre_test)
            test_pst_test = metric_func(x_pst_test, y_pst_test)
            return pd.DataFrame(index=[index_name], data={'test_pre_train': test_pre_train,
                                                          'test_pre_train_N': pre_train_test_lengths[0],
                                                          'test_pre_test': test_pre_test,
                                                          'test_pre_test_N': pre_train_test_lengths[1],
                                                          'test_pst_test': test_pst_test,
                                                          'test_pst_test_N': len(y_pst_test)})

        treat_hat_treat_obs = comparison_over_windows(y_treat_hat, y_treat_obs,
                                                      pre_train_test_lengths,
                                                      mean_absolute_percentage_error,
                                                      'mape_vs_treat_obs')
        return treat_hat_treat_obs

    def sc_validation_gather(counterfactual=None,
                             actual=None,
                             pre_train_test_lengths=None):
        from sklearn.metrics import mean_absolute_percentage_error
        x_pre_train = counterfactual[0:pre_train_test_lengths[0]]
        y_pre_train = actual[0:pre_train_test_lengths[0]]
        x_pre_test = counterfactual[pre_train_test_lengths[0]:pre_train_test_lengths[0] + pre_train_test_lengths[1]]
        y_pre_test = actual[pre_train_test_lengths[0]:pre_train_test_lengths[0] + pre_train_test_lengths[1]]
        x_pst_test = counterfactual[-1 * (pre_train_test_lengths[0] + pre_train_test_lengths[1]):].copy()
        y_pst_test = actual[-1 * (pre_train_test_lengths[0] + pre_train_test_lengths[1]):].copy()
        test_pre_train = mean_absolute_percentage_error(x_pre_train, y_pre_train)
        test_pre_test = mean_absolute_percentage_error(x_pre_test, y_pre_test)
        test_pst_test = mean_absolute_percentage_error(x_pst_test, y_pst_test)
        return pd.DataFrame(index=['mape_test'], data={'test_pre_train': test_pre_train,
                                                       'test_pre_test': test_pre_test,
                                                       'test_pst_test': test_pst_test})

    def collect_sc_outputs(sc_output=None,
                           pre_process_data=None,
                           theta_grid=None,
                           pre_train_test_lengths=None,
                           aggregate_pst_periods=True,
                           alpha=0.05):
        if aggregate_pst_periods:
            a = sc.collect_sc_outputs_aggregate(sc_output=sc_output,
                                                pre_process_data=pre_process_data,
                                                theta_grid=theta_grid,
                                                pre_train_test_lengths=pre_train_test_lengths,
                                                alpha=alpha)
        else:
            a = sc.collect_sc_outputs_individual(sc_output=sc_output,
                                                 pre_process_data=pre_process_data,
                                                 theta_grid=theta_grid,
                                                 pre_train_test_lengths=pre_train_test_lengths,
                                                 alpha=alpha)
        return a

    def collect_sc_outputs_individual(sc_output=None,
                                      pre_process_data=None,
                                      theta_grid=None,
                                      pre_train_test_lengths=None,
                                      aggregate_pst_periods=True,
                                      alpha=0.05):
        pre_T = pre_process_data['T_pre'].shape[0]
        pst_T = pre_process_data['T_pst'].shape[0]

        permutations_subset_block_individual = []
        individual_time_list = np.arange(pre_train_test_lengths[0] + 1)
        for i in range(pre_T + 1):
            half_A = individual_time_list[-1 * (pre_T - i):].copy()
            half_B = individual_time_list[0:i]
            scrambled_list = np.concatenate([half_A, half_B])
            permutations_subset_block_individual.append(list(scrambled_list))

        collect_individual_df = pd.DataFrame()
        for t_pst in range(0, pst_T):
            pst_index = np.append(np.arange(pre_T), [pre_T + t_pst])
            sc_output_subset = {'atet': sc_output['atet'].iloc[t_pst],
                                'predict_est': sc_output['predict_est'].iloc[pst_index]}
            pre_process_data_subset = {'time_scramble': permutations_subset_block_individual,
                                       'pre_pst_lengths': (pre_process_data['pre_pst_lengths'][0], 1)}
            a = sc.collect_sc_outputs_aggregate(sc_output=sc_output_subset,
                                                pre_process_data=pre_process_data_subset,
                                                theta_grid=theta_grid,
                                                pre_train_test_lengths=(pre_process_data['pre_pst_lengths'][0], 1),
                                                alpha=alpha,
                                                time_period_forindividual=t_pst)
            a['individual_post'] = t_pst
            collect_individual_df = pd.concat([collect_individual_df, a])
        return collect_individual_df

    def collect_sc_outputs_aggregate(sc_output=None,
                                     pre_process_data=None,
                                     theta_grid=None,
                                     pre_train_test_lengths=None,
                                     alpha=0.05,
                                     time_period_forindividual=None):
        if len(sc_output['atet'].shape) > 1:
            sc_results = sc_output['atet'].mean(axis=0).copy().to_frame().rename(columns={0: 'atet'})
        else:
            number_of_treated_units = len([x for x in sc_output['predict_est'].columns if '_est' not in x])
            sc_results = pd.DataFrame(index=[np.arange(number_of_treated_units)],
                                      data={'time_period': time_period_forindividual,
                                            'atet': sc_output['atet'].values})

        sc_pv = []
        sc_ci_05 = []
        sc_ci_95 = []
        sc_placebo_pre_train = []
        sc_placebo_pre_test = []
        sc_placebo_pst_test = []

        o = 0
        for p in [x for x in sc_output['predict_est'].columns if '_est' not in x]:
            y_hat = sc_output['predict_est']['{0}_est'.format(p)].values
            y_act = sc_output['predict_est']['{0}'.format(p)].values

            # ---- p-value ------------------------------------------------
            pv_output = conformal_inf.pvalue_calc(
                counterfactual=np.array(y_hat.tolist()),
                actual=np.array(y_act.tolist()),
                permutation_list=pre_process_data['time_scramble'],
                pre_pst_lengths=pre_process_data['pre_pst_lengths'],
                h0=0)
            sc_pv.append(pv_output)

            # ---- confidence interval  (binary search) ------------------
            # Determine the search range from theta_grid if supplied,
            # otherwise centre on the ATET estimate.
            if theta_grid is not None:
                lo, hi = float(theta_grid.min()), float(theta_grid.max())
            else:
                atet_est = sc_results['atet'].iloc[o]
                spread = max(abs(atet_est) * 5, 10)
                lo, hi = atet_est - spread, atet_est + spread

            ci_output = conformal_inf.ci_calc(
                y_hat=y_hat,
                y_act=y_act,
                theta_grid=theta_grid,          # kept for grid fallback
                permutation_list_ci=pre_process_data['time_scramble'],
                pre_pst_lengths_ci=pre_process_data['pre_pst_lengths'],
                alpha=alpha,
                search_bounds=(lo, hi))         # new kwarg triggers bsearch

            sc_ci_05.append(ci_output['ci_interval'][0])
            sc_ci_95.append(ci_output['ci_interval'][1])

            # ---- placebo ----------------------------------------------------
            placebo_df = sc.sc_validation_gather(
                counterfactual=np.array(y_hat.tolist()),
                actual=np.array(y_act.tolist()),
                pre_train_test_lengths=pre_process_data['pre_pst_lengths'])
            sc_placebo_pre_train.append(placebo_df['test_pre_train'].values[0])
            sc_placebo_pre_test.append(placebo_df['test_pre_test'].values[0])
            sc_placebo_pst_test.append(placebo_df['test_pst_test'].values[0])
            o += 1

        sc_results['pvalues'] = sc_pv
        sc_results['ci_lower'] = sc_ci_05
        sc_results['ci_upper'] = sc_ci_95
        sc_results['alpha'] = alpha
        sc_results['test_pre_train_MAPE'] = sc_placebo_pre_train
        sc_results['test_pre_test_MAPE'] = sc_placebo_pre_test
        sc_results['test_pst_test_MAPE'] = sc_placebo_pst_test
        return sc_results

    def sc_generate_figures(final_sc_output=None,
                            output_figure_name=None):
        def check_write_file(suffix=None):
            if output_figure_name is None:
                pass
            else:
                plt.savefig(output_figure_name + '_{0}.png'.format(suffix))

        fig, ax = plt.subplots(ncols=1, nrows=1, figsize=(12, 6))
        for r in [o for o in final_sc_output['predict_est'].columns if '_est' not in o]:
            c = ax.plot(final_sc_output['predict_est'].index,
                        final_sc_output['predict_est'][r + '_est'],
                        color=None, linestyle='--',
                        label='Estimated Control for {0}'.format(r),
                        marker='o', markerfacecolor='white', markeredgecolor=None)
            ax.plot(final_sc_output['predict_est'].index,
                    final_sc_output['predict_est'][r],
                    color=c[0].get_color(),
                    label='Observed for {0}'.format(r),
                    marker='o', markerfacecolor=c[0].get_color(),
                    markeredgecolor=c[0].get_color())

        xmin_, xmax_, ymin_, ymax_ = plt.axis()
        ax.set_xlabel('Time Periods')
        ax.set_ylabel('Outcome')
        ax.vlines(x=final_sc_output['atet'].index.min(),
                  ymin=ymin_, ymax=ymax_,
                  color='black', linewidth=2, label='Treatment Time')
        ax.grid()
        plt.legend()
        check_write_file(suffix='overall')
        plt.show()

        number_of_treated_units = len([o for o in final_sc_output['predict_est'].columns if '_est' not in o])
        fig, ax = plt.subplots(ncols=1, nrows=number_of_treated_units,
                               figsize=(12, 6 * number_of_treated_units))
        # Fix: plt.subplots(nrows=1) returns a scalar Axes, not array
        if number_of_treated_units == 1:
            ax = [ax]
        for i, r in zip(range(number_of_treated_units),
                        [o for o in final_sc_output['predict_est'].columns if '_est' not in o]):
            c = ax[i].plot(final_sc_output['predict_est'].index,
                           final_sc_output['predict_est'][r + '_est'],
                           color=None, linestyle='--',
                           label='Estimated Control for {0}'.format(r),
                           marker='o', markerfacecolor='white', markeredgecolor=None)
            ax[i].plot(final_sc_output['predict_est'].index,
                       final_sc_output['predict_est'][r],
                       color=c[0].get_color(),
                       label='Observed for {0}'.format(r),
                       marker='o', markerfacecolor=c[0].get_color(),
                       markeredgecolor=c[0].get_color())
            xmin_, xmax_, ymin_, ymax_ = ax[i].axis()
            ax[i].set_xlabel('Time Periods')
            ax[i].set_ylabel('Outcome')
            ax[i].vlines(x=final_sc_output['atet'].index.min(),
                         ymin=ymin_, ymax=ymax_,
                         color='black', linewidth=2, label='Treatment Time')
            ax[i].grid()
            ax[i].legend()
            ax[i].set_title('Estimated and Observed Trends for {0}'.format(r))
        check_write_file(suffix='deep_dive')
        plt.show()

    # ------------------------------------------------------------------
    # Unit-permutation (placebo) inference for SC
    # ------------------------------------------------------------------
    def sc_permutation_inference(data=None,
                                 data_dict={'treatment': None,
                                            'date': None,
                                            'post': None,
                                            'unitid': None,
                                            'outcome': None},
                                 model_name='adh',
                                 pre_train_test_lengths=None,
                                 alpha=0.05):
        """
        Classic Abadie et al. (2010) unit-permutation placebo test for SC.

        For every control unit j, pretend j is the treated unit, re-run
        the chosen SC estimator using the remaining units as donors,
        and compute the average post-treatment absolute prediction error.
        The p-value is the fraction of placebo units whose error is at
        least as large as the real treated unit's error.

        This complements the time-block (conformal) inference already in
        ``sc_model`` by permuting on the *unit* axis instead of the
        *time* axis.

        Caveat
        ------
        Abadie (2021) and the symmetry-critique literature note that the
        p-value can be size-distorted when aggregate shocks are present.
        Interpret as a strong diagnostic; for formal inference prefer
        the conformal path already in ``sc_model``.

        Parameters
        ----------
        data            : DataFrame   Long-format panel.
        data_dict       : dict        Column-name mapping.
        model_name      : str         One of 'adh', 'di', 'cl'.
        pre_train_test_lengths : list or None
                                  Passed through to the SC estimator.
        alpha           : float       Significance level.

        Returns
        -------
        dict with keys:
            'real_unit'         str   – the actual treated unit id.
            'real_abs_error'    float – mean abs post-treatment error for
                                        the real unit.
            'placebo_errors'    DataFrame
                                  unit, abs_error, is_real, pre_rmse
            'pvalue'            float – rank p-value (two-sided).
            'summary'           DataFrame – one-row summary.
        """
        # ---------- identify units -------------------------------------------
        all_units = data[data_dict['unitid']].unique().tolist()
        real_treated = data.loc[data[data_dict['treatment']] == 1,
                                data_dict['unitid']].unique().tolist()
        # We support multi-treated but the permutation loop is over
        # single-unit placebos.  If there are multiple treated units we
        # use the first one as the reference for the rank test.
        real_unit = real_treated[0]

        # ---------- helper: run SC for a single treated unit -----------------
        def _run_sc_single(treated_uid):
            """Re-label treated_uid as treated, everyone else as control."""
            df_p = data.copy()
            df_p[data_dict['treatment']] = (
                df_p[data_dict['unitid']] == treated_uid).astype(int)
            # post is kept the same – we're permuting units, not time

            try:
                pre_proc = dgp.clean_and_input_data(
                    dataset=df_p,
                    treatment=data_dict['treatment'],
                    unit_id=data_dict['unitid'],
                    date=data_dict['date'],
                    post=data_dict['post'],
                    outcome=data_dict['outcome'])

                ttl = dgp.determine_pre_train_test_lengths(
                    ci_data_output=pre_proc,
                    pre_train_test_lengths=pre_train_test_lengths)

                # --- pick estimator ------------------------------------------
                if model_name == 'adh':
                    sc_est = adh.predict_omega(pre_proc['T_pre'],
                                               pre_proc['C_pre'], ttl)
                    sc_out = di.sc_style_results(
                        pre_proc['T_pre'], pre_proc['T_pst'],
                        pre_proc['C_pre'], pre_proc['C_pst'],
                        np.zeros(pre_proc['T_pst'].shape[1]),
                        np.array(sc_est['omega']))
                elif model_name == 'di':
                    w = alpha_lambda.get_alpha_lambda(pre_proc['C_pre'])
                    al = alpha_lambda.alpha_lambda_transform(w.x)
                    sc_est = di.predict_mu_omega(pre_proc['T_pre'],
                                                 pre_proc['C_pre'],
                                                 al, ttl)
                    sc_out = di.sc_style_results(
                        pre_proc['T_pre'], pre_proc['T_pst'],
                        pre_proc['C_pre'], pre_proc['C_pst'],
                        sc_est['mu'], sc_est['omega'])
                elif model_name == 'cl':
                    sc_est = cl.predict_mu_omega(pre_proc['T_pre'],
                                                 pre_proc['C_pre'], ttl)
                    sc_out = di.sc_style_results(
                        pre_proc['T_pre'], pre_proc['T_pst'],
                        pre_proc['C_pre'], pre_proc['C_pst'],
                        sc_est['mu'], sc_est['omega'])
                else:
                    raise ValueError("model_name must be 'adh', 'di', or 'cl'")

                # mean absolute post-treatment prediction error
                atet = sc_out['atet']                       # post-period diffs
                abs_err = float(atet.abs().mean().mean())
                return abs_err

            except Exception:
                return np.nan

        # ---------- pre-treatment RMSE helper (diagnostic) -------------------
        def _pre_rmse(uid):
            mask = (data[data_dict['unitid']] == uid) & (data[data_dict['post']] == 0)
            vals = data.loc[mask, data_dict['outcome']]
            return float(np.sqrt((vals ** 2).mean())) if len(vals) > 0 else np.nan

        # ---------- real unit first ------------------------------------------
        real_abs_err = _run_sc_single(real_unit)

        # ---------- loop over every control unit as placebo ------------------
        rows = [{'unit': real_unit,
                 'abs_error': real_abs_err,
                 'is_real': True,
                 'pre_rmse': _pre_rmse(real_unit)}]

        control_units = [u for u in all_units if u not in real_treated]
        for j in control_units:
            rows.append({'unit': j,
                         'abs_error': _run_sc_single(j),
                         'is_real': False,
                         'pre_rmse': _pre_rmse(j)})

        placebo_df = pd.DataFrame(rows)

        # ---------- rank p-value ---------------------------------------------
        # fraction of units (including real) whose |error| >= real |error|
        valid = placebo_df['abs_error'].dropna()
        pvalue = float((valid >= real_abs_err).mean())

        summary = pd.DataFrame([{
            'real_unit': real_unit,
            'real_abs_error': real_abs_err,
            'n_placebo': len(control_units),
            'pvalue': pvalue,
            'alpha': alpha,
            'significant': pvalue < alpha
        }])

        return {'real_unit': real_unit,
                'real_abs_error': real_abs_err,
                'placebo_errors': placebo_df,
                'pvalue': pvalue,
                'summary': summary}


# ==========================================================================
# dgp  –  data-generating / cleaning utilities  (unchanged)
# ==========================================================================
class dgp:

    def simulate_panel(seed=0,
                       T_pre=20,
                       T_post=5,
                       N_control=30,
                       sigma_lambda=0.0,
                       noise_sd=0.1,
                       att_pct=0.15,
                       time_trend=0.3,
                       ar_coef=0.0):
        """Simulate a factor-model panel suitable for causal inference demos.

        DGP:  Y_it = alpha_i + time_trend * t + lambda_i * F_t + eps_it
              eps_it = ar_coef * eps_{i,t-1} + innov_it,  innov_it ~ N(0, innov_sd)
              where innov_sd = noise_sd * sqrt(1 - ar_coef^2) so that
              Var(eps_it) = noise_sd^2 regardless of ar_coef.

        The treated unit is 'T000'; all others are controls 'C000', 'C001', ...
        Treatment begins at period T_pre.  ATT is calibrated as a fraction of the
        treated unit's untreated potential outcome at the moment of treatment.

        Parameters
        ----------
        seed        : int   – numpy random seed for reproducibility
        T_pre       : int   – number of pre-treatment periods
        T_post      : int   – number of post-treatment periods
        N_control   : int   – number of control units
        sigma_lambda: float – std dev of factor loadings; 0 → parallel trends hold
        noise_sd    : float – std dev of idiosyncratic noise ε_it (marginal)
        att_pct     : float – ATT as a fraction of Y_treat at treatment onset
        time_trend  : float – common linear time trend coefficient (0.3 default)
        ar_coef     : float – AR(1) coefficient for idiosyncratic noise (0 = i.i.d.)

        Returns
        -------
        df       : pd.DataFrame
                   Columns: unit_id (str), time (int), y (float),
                             treated (int 0/1), post (int 0/1)
        true_att : float  – the calibrated true average treatment effect on treated
        """
        np.random.seed(seed)
        N_total  = N_control + 1
        T_total  = T_pre + T_post
        unit_ids = ['T000'] + [f'C{i:03d}' for i in range(N_control)]

        alpha_i  = np.random.normal(5, 2, N_total)
        lambda_i = np.random.normal(1, sigma_lambda, N_total)
        F_t      = np.random.normal(0, 1, T_total)

        innov_sd = noise_sd * np.sqrt(max(1.0 - ar_coef ** 2, 1e-10))

        rows  = []
        y_cf  = None                              # untreated potential outcome at t=T_pre
        for i, uid in enumerate(unit_ids):
            is_treated = (uid == 'T000')
            eps = 0.0
            for t in range(T_total):
                eps = ar_coef * eps + np.random.normal(0, innov_sd)
                y = (alpha_i[i] + time_trend * t
                     + lambda_i[i] * F_t[t]
                     + eps)
                post = int(t >= T_pre)
                if is_treated and t == T_pre:
                    y_cf = y
                rows.append({'unit_id':  uid,
                             'time':     t,
                             'y':        y,
                             'treated':  int(is_treated),
                             'post':     post})

        true_att = att_pct * y_cf
        df = pd.DataFrame(rows)
        df.loc[(df['treated'] == 1) & (df['post'] == 1), 'y'] += true_att
        return df, true_att

    def determine_pre_train_test_lengths(ci_data_output=None,
                                         pre_train_test_lengths=None):
        if pre_train_test_lengths is None:
            pre_treatment_len = ci_data_output['C_pre'].shape[0]
            pre_t0 = int(0.75 * pre_treatment_len)
            if pre_t0 < 1:
                pre_t0 = 1
            pre_t1 = pre_treatment_len - pre_t0
            pre_train_test_lengths = [pre_t0, pre_t1]
        return pre_train_test_lengths

    def clean_and_input_data(dataset=None,
                             treatment='treated_unit',
                             unit_id='unitid',
                             date='T',
                             post='post', outcome='Y'):
        C_pre = dataset.loc[(dataset[treatment] == 0) & (dataset[post] == 0)].pivot_table(
            columns=unit_id, index=date, values=outcome)
        C_pst = dataset.loc[(dataset[treatment] == 0) & (dataset[post] == 1)].pivot_table(
            columns=unit_id, index=date, values=outcome)
        T_pre = dataset.loc[(dataset[treatment] == 1) & (dataset[post] == 0)].pivot_table(
            columns=unit_id, index=date, values=outcome)
        T_pst = dataset.loc[(dataset[treatment] == 1) & (dataset[post] == 1)].pivot_table(
            columns=unit_id, index=date, values=outcome)

        permutations_subset_block = conformal_inf.time_block_permutation(data=dataset,
                                                                         time_unit=date,
                                                                         post=post)
        return {'C_pre': C_pre, 'C_pst': C_pst, 'T_pre': T_pre, 'T_pst': T_pst,
                'time_scramble': permutations_subset_block[0],
                'pre_pst_lengths': permutations_subset_block[1]}


# ==========================================================================
# di  –  Doudchenko & Imbens (2016)  (unchanged)
# ==========================================================================
class di:
    def estimate_mu_omega(treatment_pre, control_pre, alpha_lambda_0):
        alpha_0, lambda_0 = alpha_lambda_0[0], alpha_lambda_0[1]
        elnet = ElasticNet(random_state=2736, alpha=alpha_0, l1_ratio=lambda_0,
                           max_iter=10000)
        elnet.fit(control_pre, treatment_pre)
        try:
            df_weights = pd.DataFrame(data=zip(treatment_pre.columns, elnet.coef_.T))
        except Exception:
            df_weights = pd.DataFrame(index=np.arange(len(elnet.coef_)), data=elnet.coef_.T)
        return {'mu': elnet.intercept_, 'omega': elnet.coef_, 'weights': df_weights, 'full': elnet}

    def predict_mu_omega(treatment_pre, control_pre, alpha_lambda_0, holdout_windows):
        if (holdout_windows[0] + holdout_windows[1] != len(control_pre)):
            print('the arg holdout_windows does not add up to the number of time units!')
            print('holdout_windows = {0}'.format(holdout_windows))
            print('total number of time periods = {0}'.format(len(control_pre)))

        control_holdout = control_pre[0:holdout_windows[0]].copy()
        treatment_holdout = treatment_pre[0:holdout_windows[0]].copy()
        control_nonholdout = control_pre[holdout_windows[0]:].copy()
        treatment_nonholdout = treatment_pre[holdout_windows[0]:].copy()

        holdout_dict = di.estimate_mu_omega(treatment_holdout, control_holdout, alpha_lambda_0)
        if treatment_pre.shape[1] == 1:
            holdout_dict['omega'] = np.array([holdout_dict['omega']])

        diff_holdout = treatment_holdout - np.dot(control_holdout, holdout_dict['omega'].T) - holdout_dict['mu']
        diff_nonholdout = treatment_nonholdout - np.dot(control_nonholdout, holdout_dict['omega'].T) - holdout_dict['mu']

        diff_nonholdout_mse = (diff_nonholdout ** 2).mean()
        diff_holdout_mse = (diff_holdout ** 2).mean()
        return {'mu': holdout_dict['mu'],
                'omega': holdout_dict['omega'],
                'weights': holdout_dict['weights'],
                'full': holdout_dict['full'],
                'mse_holdout': diff_holdout_mse,
                'mse_nonholdout': diff_nonholdout_mse}

    def sc_style_results(treatment_pre, treatment_pst, control_pre, control_pst, mu, omega):
        if len(omega.shape) > 2:
            omega = omega.reshape(omega.shape[-2], omega.shape[-1])

        final_X = pd.concat([treatment_pre, treatment_pst], axis=0)
        control_X = pd.concat([control_pre, control_pst], axis=0)
        control_df = mu + pd.DataFrame(data=np.dot(control_X, omega.T),
                                       columns=[l + '_est' for l in final_X.columns])
        control_df.index = final_X.index
        output_df = control_df.join(final_X)

        treatment_periods = -1 * len(treatment_pst)
        atet_df = pd.DataFrame()
        for c in [l for l in output_df.columns if '_est' not in l]:
            diff = output_df[c][treatment_periods:].copy() - output_df[c + '_est'][treatment_periods:].copy()
            atet_df[c] = diff
        return {'atet': atet_df, 'predict_est': output_df}


# ==========================================================================
# alpha_lambda  –  hyperparameter search for DI  (unchanged)
# ==========================================================================
class alpha_lambda:
    def diff(y_t, y_c, mu_x, omega_x):
        return y_t - mu_x - np.dot(y_c, omega_x)

    def alpha_lambda_transform(alpha_lambda_raw_):
        return (np.exp(alpha_lambda_raw_[0] / 1000),
                np.exp(alpha_lambda_raw_[1]) / (1 + np.exp(alpha_lambda_raw_[1])))

    def alpha_lambda_diff(alpha_lambda_raw_, control_pre):
        alpha_lambda_t = alpha_lambda.alpha_lambda_transform(alpha_lambda_raw_)

        # 75/25 time split — fit on train, evaluate on held-out test period.
        # Pure in-sample LOO-unit MSE is trivially minimised at alpha→0 when
        # T < N (underdetermined system), driving BFGS to remove all regularisation.
        # A time holdout prevents that collapse: alpha→0 overfits train → poor test MSE.
        t_train = max(1, int(0.75 * len(control_pre)))
        ctrl_train = control_pre.iloc[:t_train]
        ctrl_test  = control_pre.iloc[t_train:]

        difference_array = []
        for u in control_pre.columns:
            treat_train = ctrl_train[u]
            cntrl_train = ctrl_train[[c for c in ctrl_train.columns if c != u]]
            treat_test  = ctrl_test[u]
            cntrl_test  = ctrl_test[[c for c in ctrl_test.columns if c != u]]

            w = di.estimate_mu_omega(treat_train, cntrl_train, alpha_lambda_t)
            # Out-of-sample residual on held-out time period
            d = np.asarray(treat_test).flatten() \
                - float(w['mu']) \
                - np.dot(cntrl_test, w['omega']).flatten()
            difference_array.append(d)

        d_mean = float(np.mean(np.concatenate(
            [d ** 2 for d in difference_array])))
        return d_mean

    def get_alpha_lambda(control_pre_input):
        # Use ElasticNetCV path search (LOO over units) — much faster than BFGS.
        from sklearn.linear_model import ElasticNetCV
        l1_ratios = [0.1, 0.3, 0.5, 0.7, 0.9, 0.95, 1.0]
        alphas_list, l1_list = [], []
        for u in control_pre_input.columns:
            y = control_pre_input[u].values
            X = control_pre_input[[c for c in control_pre_input.columns if c != u]].values
            cv = ElasticNetCV(l1_ratio=l1_ratios, cv=3, max_iter=10000, random_state=42)
            cv.fit(X, y)
            alphas_list.append(cv.alpha_)
            l1_list.append(cv.l1_ratio_)
        best_alpha = float(np.median(alphas_list))
        best_l1 = float(np.clip(np.median(l1_list), 1e-6, 1 - 1e-6))
        # Invert alpha_lambda_transform so existing callers still work
        raw0 = 1000.0 * np.log(best_alpha)
        raw1 = np.log(best_l1 / (1.0 - best_l1))

        class _Result:
            x = np.array([raw0, raw1])
        return _Result()


# ==========================================================================
# cl  –  Constrained Lasso  (unchanged)
# ==========================================================================
from scipy.optimize import fmin_slsqp


class cl:
    def cl_obj(params, y, x) -> float:
        return np.mean((y - params[0] - np.dot(x, params[1:])) ** 2)

    def predict_mu_omega(treatment_pre, control_pre, holdout_windows):
        if (holdout_windows[0] + holdout_windows[1] != len(control_pre)):
            print('the arg holdout_windows does not add up to the number of time units!')
            print('holdout_windows = {0}'.format(holdout_windows))
            print('total number of time periods = {0}'.format(len(control_pre)))

        control_holdout = control_pre[0:holdout_windows[0]].copy()
        treatment_holdout = treatment_pre[0:holdout_windows[0]].copy()
        control_nonholdout = control_pre[holdout_windows[0]:].copy()
        treatment_nonholdout = treatment_pre[holdout_windows[0]:].copy()

        holdout_dict = {}
        holdout_dict['mu'] = []
        holdout_dict['omega'] = []
        holdout_dict['weights'] = []
        diff_holdout_mse = []
        diff_nonholdout_mse = []

        if treatment_pre.shape[1] > 1:
            for t in treatment_pre.columns:
                t_dict = cl.get_mu_omega(treatment_holdout[t], control_holdout)
                diff_h = treatment_holdout[t] - np.dot(control_holdout, t_dict['omega'].T) + t_dict['mu']
                diff_h_mse = (diff_h ** 2).mean()
                diff_nh = treatment_nonholdout[t] - np.dot(control_nonholdout, t_dict['omega'].T) + t_dict['mu']
                diff_nh_mse = (diff_nh ** 2).mean()
                holdout_dict['mu'].append(t_dict['mu'])
                holdout_dict['omega'].append(t_dict['omega'])
                holdout_dict['weights'].append(t_dict['weights'])
                diff_holdout_mse.append(diff_h_mse)
                diff_nonholdout_mse.append(diff_nh_mse)
        else:
            t_dict = cl.get_mu_omega(treatment_holdout.values.flatten(), control_holdout)
            t_dict['omega'] = np.array([t_dict['omega']])
            diff_h = treatment_holdout - np.dot(control_holdout, t_dict['omega'].T) + t_dict['mu']
            diff_h_mse = (diff_h ** 2).mean()
            diff_nh = treatment_nonholdout - np.dot(control_nonholdout, t_dict['omega'].T) + t_dict['mu']
            diff_nh_mse = (diff_nh ** 2).mean()
            holdout_dict['mu'].append(t_dict['mu'])
            holdout_dict['omega'].append(t_dict['omega'])
            holdout_dict['weights'].append(t_dict['weights'])
            diff_holdout_mse.append(diff_h_mse)
            diff_nonholdout_mse.append(diff_nh_mse)

        holdout_dict['omega'] = np.array(holdout_dict['omega'])
        if len(holdout_dict['omega'].shape) > 2:
            holdout_dict['omega'] = holdout_dict['omega'].reshape(
                holdout_dict['omega'].shape[-2], holdout_dict['omega'].shape[-1])
        holdout_dict['mse_holdout'] = np.mean(diff_holdout_mse)
        holdout_dict['mse_nonholdout'] = np.mean(diff_nonholdout_mse)
        return holdout_dict

    def get_mu_omega(treatment_pre_input, control_pre_input):
        n = control_pre_input.shape[1]
        initialx = np.ones(n + 1) / 1
        weights = fmin_slsqp(partial(cl.cl_obj, y=treatment_pre_input,
                                     x=control_pre_input),
                             initialx,
                             f_ieqcons=lambda x: 1 - np.sum(np.abs(x[1:])),
                             iter=50000,
                             disp=False)
        mu, omega = weights[0], weights[1:]
        return {'mu': mu, 'omega': omega, 'weights': weights}


# ==========================================================================
# adh  –  Abadie, Diamond & Hainmueller (2010)  (unchanged)
# ==========================================================================
class adh:
    def loss_w(W, X, y) -> float:
        return np.sqrt(np.mean((y - X.dot(W)) ** 2))

    def get_w(X, y):
        w_start = [1 / X.shape[1]] * X.shape[1]
        weights = fmin_slsqp(partial(adh.loss_w, X=X, y=y),
                             np.array(w_start),
                             f_eqcons=lambda x: np.sum(x) - 1,
                             iter=50000,
                             bounds=[(0.0, 1.0)] * len(w_start),
                             disp=False)
        return weights

    def predict_omega(treatment_pre, control_pre, holdout_windows):
        if (holdout_windows[0] + holdout_windows[1] != len(control_pre)):
            print('the arg holdout_windows does not add up to the number of time units!')
            print('holdout_windows = {0}'.format(holdout_windows))
            print('total number of time periods = {0}'.format(len(control_pre)))

        control_holdout = control_pre[0:holdout_windows[0]].copy()
        treatment_holdout = treatment_pre[0:holdout_windows[0]].copy()
        control_nonholdout = control_pre[holdout_windows[0]:].copy()
        treatment_nonholdout = treatment_pre[holdout_windows[0]:].copy()

        holdout_dict = {}
        holdout_dict['omega'] = []
        holdout_dict['weights'] = []
        diff_holdout_mse = []
        diff_nonholdout_mse = []

        if treatment_pre.shape[1] > 1:
            for t in treatment_pre.columns:
                t_dict = adh.get_w(control_holdout, treatment_holdout[t])
                diff_h = treatment_holdout[t] - np.dot(control_holdout, t_dict.T)
                diff_h_mse = (diff_h ** 2).mean()
                diff_nh = treatment_nonholdout[t] - np.dot(control_nonholdout, t_dict.T)
                diff_nh_mse = (diff_nh ** 2).mean()
                holdout_dict['omega'].append(t_dict)
                holdout_dict['weights'].append(t_dict)
                diff_holdout_mse.append(diff_h_mse)
                diff_nonholdout_mse.append(diff_nh_mse)
        else:
            t_dict = adh.get_w(control_holdout, treatment_holdout.values.flatten())
            diff_h = treatment_holdout - np.dot(control_holdout, np.array([t_dict]).T)
            diff_h_mse = (diff_h ** 2).mean()
            diff_nh = treatment_nonholdout - np.dot(control_nonholdout, np.array([t_dict]).T)
            diff_nh_mse = (diff_nh ** 2).mean()
            holdout_dict['omega'].append(t_dict)
            holdout_dict['weights'].append(t_dict)
            diff_holdout_mse.append(diff_h_mse)
            diff_nonholdout_mse.append(diff_nh_mse)

        holdout_dict['omega'] = np.array(holdout_dict['omega'])
        holdout_dict['mse_holdout'] = np.mean(diff_holdout_mse)
        holdout_dict['mse_nonholdout'] = np.mean(diff_nonholdout_mse)
        return holdout_dict


# ==========================================================================
# conformal_inf  –  conformal inference for SC models
# ==========================================================================
from numpy.random import default_rng


class conformal_inf:
    # ----------------------------------------------------------------------
    # Permutation schedule generation  (unchanged)
    # ----------------------------------------------------------------------
    def time_block_permutation(data=None, time_unit=None, post=None):
        pre_pst_lengths = data.groupby(post)[time_unit].nunique().sort_index(ascending=True).to_list()
        time_list = np.arange(np.sum(pre_pst_lengths))
        T_len = len(time_list)

        permutations_subset_block = []
        for i in range(T_len):
            half_A = time_list[-1 * (T_len - i):]
            half_B = time_list[0:i]
            scrambled_list = np.concatenate([half_A, half_B])
            permutations_subset_block.append(list(scrambled_list))

        def add_20_permutations(base_list=None):
            r = 0
            while r < 20:
                x = list(np.random.permutation(T_len))
                if x not in base_list:
                    base_list.append(x)
                    r += 1
            return base_list

        while len(permutations_subset_block) < 20:
            permutations_subset_block = add_20_permutations(base_list=permutations_subset_block)
        return permutations_subset_block, pre_pst_lengths

    # ----------------------------------------------------------------------
    # Core primitives  (unchanged)
    # ----------------------------------------------------------------------
    def scrambled_residual(counterfactual, actual, scrambled_order, pre_pst_lengths):
        counterfactual_ = counterfactual[scrambled_order].copy()
        actual_ = actual[scrambled_order].copy()
        return np.abs(actual_ - counterfactual_)[-1 * pre_pst_lengths[1]:]

    def test_statS(q, pre_pst_lengths, residual_abs):
        normed = np.sum(np.power(residual_abs, q))
        return np.power(pre_pst_lengths[1] ** (-0.5) * normed, 1 / q)

    # ----------------------------------------------------------------------
    # p-value  (unchanged)
    # ----------------------------------------------------------------------
    def pvalue_calc(counterfactual=None,
                    actual=None,
                    permutation_list=None,
                    pre_pst_lengths=None,
                    h0=0):
        assert np.sum(pre_pst_lengths) == len(actual), \
            "the argument 'pre_pst_lengths' does not cover the entirety of the actual array of outcomes.\n" \
            " pre_pst_lengths = {0} \n len(actual)= {1}".format(pre_pst_lengths, len(actual))

        control_pst = counterfactual[-1 * pre_pst_lengths[1]:].copy()
        actual_pst = actual[-1 * pre_pst_lengths[1]:].copy()
        actual_pst -= h0

        residual_initial = np.abs(actual_pst - control_pst)
        S_q = conformal_inf.test_statS(1, pre_pst_lengths, residual_initial)

        treat_ = actual.copy()
        treat_[-1 * pre_pst_lengths[1]:] -= h0
        full_residual = np.abs(treat_ - counterfactual)

        S_q_pi = []
        for r in permutation_list:
            scrambled_dates = np.array(list(r))
            residual_ = full_residual[scrambled_dates][-1 * pre_pst_lengths[1]:].copy()
            S_q_pi.append(conformal_inf.test_statS(1, pre_pst_lengths, residual_))

        p_value = 1 - np.average((np.array(S_q_pi) < S_q))
        return p_value

    # ----------------------------------------------------------------------
    # CI  –  dispatcher  (new)
    # ----------------------------------------------------------------------
    def ci_calc(y_hat=None,
                y_act=None,
                theta_grid=None,
                permutation_list_ci=None,
                pre_pst_lengths_ci=None,
                alpha=0.05,
                search_bounds=None):
        """
        Compute a conformal confidence interval.

        If *search_bounds* is provided (a (lo, hi) tuple), uses the fast
        binary-search method.  Otherwise falls back to the original
        exhaustive grid scan.
        """
        if search_bounds is not None:
            return conformal_inf.ci_calc_bsearch(
                y_hat=y_hat, y_act=y_act,
                lo=search_bounds[0], hi=search_bounds[1],
                permutation_list_ci=permutation_list_ci,
                pre_pst_lengths_ci=pre_pst_lengths_ci,
                alpha=alpha)
        else:
            return conformal_inf.ci_calc_grid(
                y_hat=y_hat, y_act=y_act,
                theta_grid=theta_grid,
                permutation_list_ci=permutation_list_ci,
                pre_pst_lengths_ci=pre_pst_lengths_ci,
                alpha=alpha)

    # ----------------------------------------------------------------------
    # CI  –  original grid method  (renamed, logic unchanged)
    # ----------------------------------------------------------------------
    def ci_calc_grid(y_hat=None,
                     y_act=None,
                     theta_grid=None,
                     permutation_list_ci=None,
                     pre_pst_lengths_ci=None,
                     alpha=0.05):
        """
        Original method: evaluate p(theta) at every point on theta_grid and
        collect the set where p > alpha.  Cost = O(len(theta_grid) * n_perms).
        """
        pv_grid = []
        for t in theta_grid:
            pv = conformal_inf.pvalue_calc(counterfactual=y_hat.copy(),
                                           actual=y_act.copy(),
                                           permutation_list=permutation_list_ci,
                                           pre_pst_lengths=pre_pst_lengths_ci,
                                           h0=t)
            pv_grid.append(pv)
        ci_list = [theta_grid[i] for i in range(len(pv_grid)) if pv_grid[i] > alpha]
        return {'theta_list': theta_grid, 'pvalue_list': pv_grid, 'ci_list': ci_list,
                'ci_interval': [np.min(ci_list), np.max(ci_list)]}

    # ----------------------------------------------------------------------
    # CI  –  binary-search method  (new)
    # ----------------------------------------------------------------------
    def ci_calc_bsearch(y_hat=None,
                        y_act=None,
                        lo=None, hi=None,
                        permutation_list_ci=None,
                        pre_pst_lengths_ci=None,
                        alpha=0.05,
                        tol=1e-4,
                        max_iter=60):
        """
        Find the lower and upper CI endpoints by binary search on the
        p-value function p(theta).

        The conformal p-value is monotonically decreasing as theta moves
        away from the true treatment effect in either direction.  We
        exploit this to locate the two roots of  p(theta) = alpha  with
        O(log((hi-lo)/tol)) calls to pvalue_calc instead of the
        O(len(theta_grid)) calls required by the grid method.

        With default tol=1e-4 and a range of 20 that is ~17 bisection
        steps per endpoint  →  ~34 p-value evaluations total, compared to
        the 4 000 evaluations the original code used (theta_grid step 0.005
        over [-10, 10]).

        Parameters
        ----------
        lo, hi      : float
            Search interval.  Must bracket the CI (i.e. p(lo) < alpha and
            p(hi) < alpha, while p somewhere in between > alpha).
        tol         : float
            Precision of the returned endpoints.
        max_iter    : int
            Safety cap on bisection iterations.

        Returns
        -------
        dict with keys 'ci_interval', 'ci_lower_pvalue', 'ci_upper_pvalue',
        'n_pvalue_evals'.
        """
        n_evals = 0

        def _pv(theta):
            nonlocal n_evals
            n_evals += 1
            return conformal_inf.pvalue_calc(counterfactual=y_hat.copy(),
                                             actual=y_act.copy(),
                                             permutation_list=permutation_list_ci,
                                             pre_pst_lengths=pre_pst_lengths_ci,
                                             h0=theta)

        # ---- find a seed point inside the CI (where p > alpha) ----------
        # Walk inward from lo until we find it, then use it as the anchor.
        mid_seed = (lo + hi) / 2.0
        p_mid = _pv(mid_seed)
        if p_mid <= alpha:
            # Coarse scan to find any interior point
            found = False
            for frac in np.linspace(0.1, 0.9, 9):
                candidate = lo + frac * (hi - lo)
                if _pv(candidate) > alpha:
                    mid_seed = candidate
                    p_mid = _pv(mid_seed)
                    found = True
                    break
            if not found:
                # Entire range has p <= alpha; CI is empty or outside bounds
                return {'ci_interval': [np.nan, np.nan],
                        'ci_lower_pvalue': None,
                        'ci_upper_pvalue': None,
                        'n_pvalue_evals': n_evals}

        # ---- lower endpoint: bisect on [lo, mid_seed] ------------------
        a_lo, a_hi = lo, mid_seed
        for _ in range(max_iter):
            if (a_hi - a_lo) < tol:
                break
            mid = (a_lo + a_hi) / 2.0
            if _pv(mid) > alpha:
                a_hi = mid          # p > alpha  →  still inside CI
            else:
                a_lo = mid          # p <= alpha →  outside CI
        ci_lower = a_hi             # first point where p > alpha
        p_lower = _pv(ci_lower)

        # ---- upper endpoint: bisect on [mid_seed, hi] ------------------
        b_lo, b_hi = mid_seed, hi
        for _ in range(max_iter):
            if (b_hi - b_lo) < tol:
                break
            mid = (b_lo + b_hi) / 2.0
            if _pv(mid) > alpha:
                b_lo = mid          # still inside CI
            else:
                b_hi = mid          # outside CI
        ci_upper = b_lo             # last point where p > alpha
        p_upper = _pv(ci_upper)

        return {'ci_interval': [ci_lower, ci_upper],
                'ci_lower_pvalue': p_lower,
                'ci_upper_pvalue': p_upper,
                'n_pvalue_evals': n_evals}