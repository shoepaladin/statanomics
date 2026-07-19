# Proposed fixes: `stnomics.py` (statanomics) and `panelib.py` (panellibs)

Companion to `ECONOMETRIC_AUDIT.md`. For each finding this gives a concrete code fix
(before → after, with line anchors against the audited revision) and the reason it is correct.
Fixes are grouped by file and ordered to match the audit's master ranking. Line numbers refer to
the state of the files at the audit commit; apply by matching the `before` snippet.

Conventions: `e = ê(X)` propensity score, `m0/m1` outcome regressions, `p̂ = E[D]`, `n` = effective
sample size after trimming. "Influence-function SE" = `np.std(ψ)/√n` with ψ the (mean-zero)
first-order influence function of the estimator.

---

## Part I — `stnomics.py`

### S-1 — Make the cross-fitting helpers order-agnostic (A → resolved)

**Why.** `predict_cv` / `predict_treatment_indicator` / `predict_counterfactual_outcomes` /
`predict_continuous` (48–117) `.extend()` fold-by-fold and return `np.array(...)`, which callers
subtract from columns in *row* order. Correct only if the split column is block-contiguous. Fixing
the helpers to write predictions into their **original positional slots** removes the entire
silent-misalignment class (incl. `feature_balance`'s random-split DML mode).

`predict_cv` (48–56):

```python
# before
def predict_cv(dataset, split_name, n_data_splits, feature, x, model):
    y_hat = []
    for r in np.arange(n_data_splits):
        train = (dataset[split_name] != r)
        test = (dataset[split_name]==r)
        lg = model.fit(dataset[feature][train==True],dataset[x][train==True])
        prediction = lg.predict(dataset[feature][test==True])
        y_hat.extend(prediction)
    return np.array(y_hat)

# after
def predict_cv(dataset, split_name, n_data_splits, feature, x, model):
    X = dataset[feature].to_numpy(); y = dataset[x].to_numpy()
    splits = dataset[split_name].to_numpy()
    y_hat = np.full(len(dataset), np.nan)
    for r in np.arange(n_data_splits):
        train = splits != r; test = splits == r
        y_hat[test] = model.fit(X[train], y[train]).predict(X[test])
    return y_hat
```

Apply the identical pattern to `predict_treatment_indicator` (use `predict_proba(...)[:,1]`),
`predict_continuous`, and `predict_counterfactual_outcomes`:

```python
# predict_counterfactual_outcomes (89–107) — after
def predict_counterfactual_outcomes(dataset, split_name, n_data_splits, feature, treatment, outcome, model):
    X = dataset[feature].to_numpy(); Y = dataset[outcome].to_numpy(); D = dataset[treatment].to_numpy()
    splits = dataset[split_name].to_numpy()
    yhat_treat   = np.full(len(dataset), np.nan)
    yhat_control = np.full(len(dataset), np.nan)
    for r in np.arange(n_data_splits):
        train = splits != r; test = splits == r
        yhat_treat[test]   = model.fit(X[train & (D==1)], Y[train & (D==1)]).predict(X[test])
        yhat_control[test] = model.fit(X[train & (D==0)], Y[train & (D==0)]).predict(X[test])
    return yhat_treat, yhat_control
```

`.to_numpy()` guarantees positional alignment regardless of the dataframe index. All existing
callers already treat the return as a positional array, so no call-site change is needed.

### S-4 — `bootstrap.go_reps` ATT keys (A, trivial)

376–377:

```python
# before
'ATT mean':ate_te[0], 'ATT std':ate_te[1],
'ATT SE mean':ate_se[0], 'ATT SE std':ate_se[1]
# after
'ATT mean':att_te[0], 'ATT std':att_te[1],
'ATT SE mean':att_se[0], 'ATT SE std':att_se[1]
```

### S-2/S-3/S-10(IPW) — `ate.ipw_main` Hájek estimators + IF standard errors (A/B)

Replace the body from 850 onward:

```python
# after (replaces lines 850–859)
keep = (that >= aux_dictionary['lower']) & (that <= aux_dictionary['upper'])
D = data_est[treatment_name].to_numpy()[keep]
Y = data_est[outcome_name].to_numpy()[keep]
e = that[keep]
n = len(Y)

# --- Hájek (self-normalized) ATE ---
w1, w0 = D/e, (1-D)/(1-e)
mu1 = np.sum(w1*Y)/np.sum(w1)
mu0 = np.sum(w0*Y)/np.sum(w0)
ate = mu1 - mu0
psi_ate = w1*(Y-mu1)/np.mean(w1) - w0*(Y-mu0)/np.mean(w0)   # linearized IF of the ratio estimator
se_ate = np.std(psi_ate)/np.sqrt(n)

# --- Hájek ATT ---  weights: treated=D, control=(1-D)·e/(1-e)
wt, wc = D, (1-D)*e/(1-e)
m1 = np.sum(wt*Y)/np.sum(wt)
m0 = np.sum(wc*Y)/np.sum(wc)
att = m1 - m0
psi_att = wt*(Y-m1)/np.mean(wt) - wc*(Y-m0)/np.mean(wc)
se_att = np.std(psi_att)/np.sqrt(n)

return {'ATE TE':ate, 'ATE SE':se_ate, 'ATT TE':att, 'ATT SE':se_att, 'PScore':e}
```

**Why.** The old ATT `mean((Y·D/e − Y(1−D)/(1−e))·e)` equals `P(D=1)·ATT`; the correct ATT is the
Hájek difference of the treated mean and the `e/(1−e)`-reweighted control mean. The old "SE"
(`np.std` of the HT summand) is neither divided by √n nor the influence-function variance; the
replacement is the standard linearized-ratio IF scaled by 1/√n. This IF conditions on `e` (ignores
first-stage estimation) — conservative for the ATE in the Hirano–Imbens–Ridder framework; documented
as a known limitation. Combined with S-4 this makes `ate.ipw`'s bootstrapped ATT correct as well.

### S-5 / S-12 — `ate.dml.dml_irm` CCDDHNR ATT score and trimmed-n variance (A/B)

Replace the ATE-SE block (1004–1016) and the ATT block (1018–1034):

```python
# after — ATE variance (replaces 1004–1016; deletes the dead j0/var_hat lines 1004–1007)
n_kept = int(keep_these.sum())
score = treatment_estimate - np.mean(treatment_estimate)     # already trimmed at 1001
SE = np.std(score) / np.sqrt(n_kept)                          # divide by KEPT n, not full n

# after — ATT estimator + score (replaces 1018–1034)
prob_unconditional = data_est[treatment_name].mean()
Dk  = data_est[treatment_name].to_numpy()[keep_these]
ycr = (data_est[outcome_name].to_numpy() - yhat_control)[keep_these]
ek  = that[keep_these]
att_term = (Dk*ycr)/prob_unconditional - (ek*(1-Dk)*ycr)/((1-ek)*prob_unconditional)
theta_att = np.mean(att_term)
score_att = att_term - theta_att*Dk/prob_unconditional        # CCDDHNR ATT score centering
SE_ATT = np.std(score_att)/np.sqrt(n_kept)
return {'ATE TE':np.mean(treatment_estimate), 'ATE SE':SE,
        'ATT TE':theta_att, 'ATT SE':SE_ATT, 'PScore':that,
        "treatment_estimate":treatment_estimate}
```

**Why.** The old ATT control-term weight was 1 (the `that` cancelled), which is not Neyman-orthogonal
— the Chernozhukov et al. (2018) interactive-model ATT score weights the control residual by
`ê/(1−ê)`, restoring double robustness. Centering the score by `θ·D/p̂` is required for the correct
asymptotic variance. Dividing the ATE SE by the kept count fixes the √(n/n_kept) understatement
under trimming (flag the estimand as the trimmed-population ATE/ATT).

### S-6/S-7 — `secondstage` prediction SE and intercept leak (A/B)

Both `second_stage` (OLS branch prediction 190–191, SE 242–245) and `het_ols` (SE 1193–1195). Fix
the OLS-branch CATE and the shared SE block:

```python
# CATE prediction (replaces 190–191): drop the intercept column; dot the derivative design
X_test = pd.concat([test_data[['ones']], test_data[het_feature]], axis=1)   # was ['cons','ones']+het
treatment_estimate = np.asarray(X_test, float) @ np.asarray(finalmodel_fit.params[1:], float)

# SE (replaces 242–245): correct quadratic form sqrt(diag(Xd V Xd'))
var_cov = np.asarray(finalmodel_fit.cov_params())[1:, 1:]
Xd = np.asarray(X_test, float)                       # columns align with params[1:] = [t, t_x...]
output_se = np.sqrt(np.einsum('ij,jk,ik->i', Xd, var_cov, Xd))
```

**Why.** The CATE is the derivative of the fitted surface w.r.t. treatment: `t_coef + Σ t_x·h`. The
old code dotted `[cons, ones, het]` against `[const, t, t_x]`, adding the fitted intercept to every
CATE. The old SE, `√|x′Vβ̂|`, contracts the covariance with the coefficient vector; the correct
prediction SE is `√(x′Vx)`. `het_ols` gets the same `einsum` treatment on its own `var_cov`/design.
(`second_stage_no_interactions`' OLS branch already aligns `add_constant(test[het])` with the fitted
`[const, het]`, so only its SE needs the `einsum` fix, not the prediction.)

Deeper (CVLasso branches 211–216, 295–299): selection, refit, inference and prediction all happen on
the *same* test half — invalid post-selection inference. Correct pattern: select on one half, then
fit **and predict out-of-fold** on the other (two-way, as the DR-learner already scaffolds). Marked
Medium effort; the SE/intercept fixes above are the immediate correctness wins.

### S-6 (crash) — undefined `lasso_alpha` (C)

Add a class-level default next to `lasso_max_iter` (173) and reference it:

```python
class secondstage:
    lasso_max_iter = 1000
    lasso_alpha = 0.1            # add
    ...
    lasso_fit = Lasso(alpha=secondstage.lasso_alpha, max_iter=secondstage.lasso_max_iter).fit(...)  # 219, 305
```

### S-8/S-9 — HTE-DML module made runnable (C, trivial)

`np.int` → `int` at 1121, 1380, 1476. Unpack the 4-tuple at 1409 and 1504:

```python
# 1409 / 1504 — before
treatment_estimate,se_estimate, coef_pd = secondstage.second_stage(approach, test_data, train_data, covar_list, het_feature )
# after
treatment_estimate, se_estimate, coef_pd, _ = secondstage.second_stage(approach, test_data, train_data, covar_list, het_feature)
```

### S-13 — `dml_plm` robust SE, `yhat` guard, estimand label (B/C)

`yhat` logic (938–942):

```python
# after
if aux_dictionary.get('yhat') is not None:
    outcome_hat = np.asarray(aux_dictionary['yhat'])
else:
    outcome_hat = []
    for r in np.arange(n_data_splits):
        train = (data_est[split_name] != r); test = (data_est[split_name]==r)
        outcome_hat.extend(ymodel.fit(data_est[feature_name][train], data_est[outcome_name][train])
                                 .predict(data_est[feature_name][test]))
    outcome_hat = np.array(outcome_hat)   # (positional if S-1 helpers are used; here splits are block-contiguous)
```

Second-stage variance (961):

```python
finalmodel_fit = sm.OLS(list(outcome_residual[keep_these]), X).fit(cov_type='HC1')
```

**Why.** `(aux_dictionary['yhat'] != None).any()` throws when `yhat` is `None`; `.get(...) is not None`
is safe. The Robinson residual-on-residual OLS needs the heteroskedasticity-robust (HC1) sandwich,
which coincides with the CCDDHNR DML variance up to the finite-fold correction. Point estimate is
already the correct partialling-out estimator. Also relabel: the PLM identifies a variance-weighted
projection, not the ATT under heterogeneity — either drop the `ATT` keys or document (line 963).

### S-10 — Oster δ diagnostics (A)

Two parts. **(a) R² measure — definitely wrong, cheap to fix** (456, 462, 495, and `predQC.r2`
164–165):

```python
from sklearn import metrics
# out-of-sample R² for the cross-fit branches (delta_ols 456/462, te_r2_output already uses r2_score)
R_dot   += metrics.r2_score(test[y], model_predict_test) / random_partitions
R_tilde += metrics.r2_score(test[y], model_predict_test) / random_partitions
# predQC.r2
def r2(truth, estimate): return metrics.r2_score(truth, estimate)
```

`np.sqrt(np.corrcoef(a,b)[0,1])` is `√r`, not R², and returns NaN when `r<0`. For `delta_ml`,
also make `R_dot` cross-fitted to match `R_tilde`'s scale (do not mix in-sample `.rsquared` at 495
with the cross-fitted `te_r2_output` at 501).

**(b) δ formula — structurally wrong, needs a verified port.** The `A`-term (472–473) uses
`(β̇−β̃)²` where Oster (2019, JBES) Prop. 3 / `psacalc` has the first power in that product, and the
`num += 2*A` step (525) double-counts the cubic term. Rather than hand-patch, port the closed form
from `psacalc.ado` (the δ that sets `β=β*`), and pin it with a regression test against Oster's
published example. Target expressions (to be verified in scrutiny):

```
num = (β̃−β*)(R̃−Ṙ)σ_y·τ_x + (β̃−β*)·σ_x·τ_x·(β̇−β̃)²
    + 2(β̃−β*)²·τ_x·(β̇−β̃)·σ_x + (β̃−β*)³·(τ_x·σ_x − τ_x²)
den = (Rmax−R̃)σ_y·(β̇−β̃)·σ_x + (β̃−β*)(Rmax−R̃)σ_y·(σ_x − τ_x)
    + (β̃−β*)²·τ_x·(β̇−β̃)·σ_x + (β̃−β*)³·(τ_x·σ_x − τ_x²)
δ = num / den
```

Effort Medium; **flagged for scrutiny** — the algebra must be checked term-by-term against a
reference implementation before merge, not trusted from this document.

### S-11 — `propbinning` SE and binning (B)

`propbinning_main` return (829): the analytic SE is the dispersion of τ̂(Xᵢ), not an SE — keep the
bootstrap wrapper as the headline SE and stop advertising the dispersion as a standard error:

```python
n = len(treatment_estimate)
return {'ATE TE':np.average(treatment_estimate), 'ATE SE':np.std(treatment_estimate)/np.sqrt(n),
        'ATT TE':np.average(treatment_estimate[data_est[treatment_name]==1]),
        'ATT SE':np.std(treatment_estimate[data_est[treatment_name]==1])/np.sqrt((data_est[treatment_name]==1).sum()),
        'PScore':that}
```

and make the binning robust to tied propensity scores (799): `pd.qcut(..., duplicates='drop')`.
Fix `propbinning` (783) to report `main_result` point estimates rather than the bootstrap mean, and
`go_reps` per S-4.

### S-14 — `hte.other.DR` trimming + random split (D)

Apply the (currently commented-out) overlap trim and randomize the cross-fitting split (1113–1122):

```python
keep = data_for_2nd_stage['t_hat'].between(aux_dictionary['lower'], aux_dictionary['upper'])
keep &= data_for_2nd_stage['y'].replace([np.inf,-np.inf], np.nan).notna()
data_for_2nd_stage = data_for_2nd_stage.loc[keep].copy()
rng = np.random.default_rng(aux_dictionary.get('seed', 0))
data_for_2nd_stage['half'] = rng.integers(0, 2, len(data_for_2nd_stage))   # was positional; was np.int
```

### S-15 — remaining stnomics items

| Lines | Fix |
|---|---|
| 139–153 `predQC.battery` | qualify calls (`predQC.outcome`, `predQC.treatment`) and rename the array arg `treatment`→`treatment_status` so it stops shadowing the method |
| 764, 918, 455, 461, 494 | `params[1]` → `.iloc[1]` (or `params[treatment_name]`) for pandas ≥ 3 |
| 346–362 `bootstrap.reps` | add optional `cluster` column; resample unique cluster ids with replacement instead of rows; replace positional `args[1..8]` with `functools.partial`/`**kwargs` |
| 695 `feature_balance` | `result_df = pd.concat([result_df, row])` (`.append` removed in pandas ≥ 2.0) |
| 60–74 `block_splits` | build folds by position over a (optionally shuffled) index and skip if `split_name` already present, so user-supplied splits are respected rather than overwritten; operate without mutating unexpectedly. See snippet below |
| 1329–1341 `HR` | build the instrument as `(D − ê)·xₖ` (as `SGCT` already does) rather than `D·xₖ − Ê[D·xₖ\|X]` |
| 162, 1097/1396/1491, 21 | guard `mape` against zero denominators; `== None` → `is None`; drop the module-level `chained_assignment=None` |

```python
# block_splits (60–74) — after
def block_splits(data_est=None, split_name='splits', n_data_splits=4, shuffle=False, seed=None):
    if split_name in data_est.columns:
        return                                   # respect a user-supplied split column
    n = len(data_est)
    order = np.random.default_rng(seed).permutation(n) if shuffle else np.arange(n)
    fold = np.empty(n, dtype=int)
    edges = np.linspace(0, n, n_data_splits + 1).astype(int)
    for p in range(n_data_splits):
        fold[order[edges[p]:edges[p+1]]] = p
    data_est[split_name] = fold
```

---

## Part II — `panelib.py`

### P-1 — SDID weight programs match Arkhangelsky et al. (2021) (A)

Free and un-penalize the intercepts; scale the ω penalty by `T_pre` and penalize slopes only; shrink
the λ ridge to a uniqueness-only term.

```python
# l_unit (599–615) — after
def l_unit(omega_array, zeta, data_control_pre, data_control_pst, data_treat_pre, data_treat_pst):
    omega_0, omega_sdid = omega_array[0], omega_array[1:]
    N_tr  = data_treat_pst.shape[1]
    T_pre = data_control_pre.shape[0]
    control_y = omega_0 + np.dot(data_control_pre, omega_sdid)
    treat_y = data_treat_pre.sum(axis=1) / N_tr
    regularization = zeta**2 * T_pre * np.sum(omega_sdid**2)     # slopes only, ×T_pre
    return np.sum((control_y - treat_y)**2) + regularization

# estimate_omega bounds (636) — after: free intercept
bounds=[(-np.inf, np.inf)] + [(0.0, np.inf)]*(len(initial_omega)-1)

# t_unit (656) — after: tiny ridge, slopes only (paper's uniqueness term ~1e-6·σ)
regularization = 1e-6 * np.sum(lambda_array[1:]**2)

# estimate_lambda bounds (682) — after: free intercept
bounds=[(-np.inf, np.inf)] + [(0.0, np.inf)]*(len(initial_lambda)-1)
```

**Why.** The paper solves ω over a free intercept and a simplex slope, penalized by `ζ²·T_pre·‖ω‖²`
(intercept excluded); λ carries only a vanishing ridge for uniqueness. The old bounds pinned ω₀,λ₀≥0
(so the intercept could not absorb a level gap, forcing the unit weights to distort), the ω penalty
dropped `T_pre` and hit the intercept, and the λ ridge (`ζ²·N_co`) was large enough to shrink λ
toward uniform. The audit's corrected program (this one) recovered the true ATT to +0.3% with 14×
lower variance in a clean common-trends DGP. The ATT formula (429–455) is already correct.

### P-2 / P-9(names) — event-study & TWFE metadata registry instead of name-parsing (A/C)

Stop encoding `(period, unit)` in column-name strings and parsing them with `split('_')` (fragile for
`_`-containing ids, and — the load-bearing bug — the post dummies are keyed by the **positional
index** `i`, so the counterfactual merge at 104–105 compares a positional index against a calendar
date and silently `fillna(0)`s). Carry an explicit registry.

```python
# twfe: per-unit dummies (135–141) — build a name→unit map
post_treated = pd.DataFrame(index=data.index)
unit_of = {}
for r in treated_units:
    col = 'post_x_{0}'.format(r)
    post_treated[col] = (data[data_dict['post']]*(data[data_dict['unitid']]==r)).astype(float)
    unit_of[col] = r
...
# building df_twfe (157–165): use the registry, not r.split('_')[-1]
for col, coef_, se_, pv_ in zip(twfe_coef.index, twfe_coef, twfe_se, twfe_pvalues):
    df_twfe = pd.concat([df_twfe, pd.DataFrame(index=[col],
        data={'treated_unit': unit_of[col], 'coef_':coef_, 'se_':se_, 'pvalue':pv_})])

# event study (186–206): key on the actual period value; keep a registry
event_meta = {}   # colname -> (kind, period, unit)
for t_period in t_period_list:
    if t_period == hold_out_time:
        continue
    kind = 'pre_treat' if t_period < hold_out_time else 'pst_treat'
    for t_units_ in treated_units:
        col = '{0}::{1}::{2}'.format(kind, t_period, t_units_)     # '::' never in ids/dates
        event_dummies[col] = ((data[data_dict['unitid']]==t_units_) &
                              (data[data_dict['date']]==t_period)).astype(float)
        event_meta[col] = (kind, t_period, t_units_)
        (pre_treat_columns if kind=='pre_treat' else pst_treat_columns).append(col)

# event_pst_df (270–280): pull period/unit from the registry, correct dtype
event_pst_df = pd.DataFrame(data={
    'pre_event':0,
    'time_period':[event_meta[c][1] for c in event_pst_coef.index],   # == calendar date, native dtype
    'treated_unit':[event_meta[c][2] for c in event_pst_coef.index],
    'coef_':event_pst_coef, 'se_':event_pst_se, 'tstat':event_pst_tstat,
    'pvalue':event_pst_pvalues, 'FJointStat':FJointStat, 'FJointPValue':FJointPValue})
```

Now `treated_counterfactual`'s merge (104–105) matches `time_period` (a real date) against the date
column and actually subtracts the ATT. Also fixes the pre/post `time_period` dtype mismatch (str vs
int) and the `_`-in-unit-id garbling. The F-test row selectors (230, 258) should test
`event_meta[name][0]` membership rather than substring `'pre'`/`'pst'` in the name.

### P-3 — staggered-adoption guardrail now; cohort estimator later (A)

Immediate (in `did.twfe`, after `treated_units` is known ~133):

```python
onset = (data.loc[data[data_dict['post']]==1]
             .groupby(data_dict['unitid'])[data_dict['date']].min())
if onset.reindex(treated_units).nunique() > 1:
    raise NotImplementedError(
        "Staggered adoption detected: treatment onset varies across treated units. "
        "A single post-dummy-per-unit TWFE is biased under heterogeneous timing "
        "(Goodman-Bacon 2021; Sun-Abraham 2021). Use a cohort/event-time estimator.")
```

Full fix (High effort): a Callaway–Sant'Anna (2021) group-time estimator — for each cohort g and
period t, ATT(g,t) = [ΔȲ treated-g] − [ΔȲ controls] using never-treated (or not-yet-treated)
controls, reference period `e = −1`, aggregated to event-time/overall with the CS influence-function
SEs; or Sun–Abraham interaction-weighted cohort×event-time dummies. Coding sketch is provided in the
PR body; delivered as a separate `did.cs_att` method so the fast 2×2 path is untouched.

### P-4 — cluster-robust / randomization inference for DiD (B)

The per-unit `post_x_unit` coefficients are each identified within a single cluster, so a CRVE is
degenerate for them — clustering must be applied to a **pooled** treated×post specification, and
per-unit inference must come from randomization/conformal methods.

```python
# pooled ATT with unit-clustered SEs (add to did.twfe output)
data = data.copy()
data['_treatpost'] = (data[data_dict['post']].values *
                      data[data_dict['unitid']].isin(treated_units).values).astype(float)
Xp = sm.add_constant(pd.concat([data[['_treatpost']], t_fe, x_fe], axis=1))
pooled = sm.OLS(data[data_dict['outcome']], Xp).fit(
    cov_type='cluster', cov_kwds={'groups': data[data_dict['unitid']]})
pooled_att = {'coef_': pooled.params['_treatpost'],
              'se_':   pooled.bse['_treatpost'],       # G-1 dof applied by statsmodels
              'pvalue':pooled.pvalues['_treatpost'], 'n_clusters': data[data_dict['unitid']].nunique()}
```

With few treated clusters, replace the CRVE with a wild-cluster bootstrap (Cameron–Gelbach–Miller
2008) or Conley–Taber (2011) inference — sketch in the PR body. Drop or explicitly flag the per-unit
`se_`/`pvalue` columns produced by `treated_counterfactual`, which are not valid t-inference.

### P-5 — delete the iid conformal padding (B, trivial)

`time_block_permutation` (1515–1522): remove the whole `while` loop; return the T cyclic shifts only.

```python
# after (replaces 1506–1523)
permutations_subset_block = []
for i in range(T_len):
    permutations_subset_block.append(
        list(np.concatenate([time_list[-1*(T_len-i):], time_list[0:i]])))
return permutations_subset_block, pre_pst_lengths
```

**Why.** The T moving-block (cyclic) shifts are the CWZ (2021) permutation set that is valid under
weak serial dependence; padding to 200 with unrestricted iid permutations makes ~87% of the null
distribution valid only under exchangeability (size 19% at ρ=0.9). With |Π|=T the test is merely
conservative (min p = 1/T), never invalid — the stated rationale for padding is backwards. If finer
p-value resolution is required, report the achievable grid (multiples of 1/T) rather than fabricating
permutations. (Also: seed any remaining randomness — line 1520 used the unseeded global RNG.)

### P-6 — per-period conformal permutations over the right index set (B)

`collect_sc_outputs_individual` (896–902): build the cyclic shifts over `pre_T+1` positions with the
non-negative-slice loop from `time_block_permutation`.

```python
# after
permutations_subset_block_individual = []
L = pre_T + 1
tl = np.arange(L)
for i in range(L):
    permutations_subset_block_individual.append(
        list(np.concatenate([tl[-1*(L-i):], tl[0:i]])))
```

**Why.** The old code built permutations over the *train* length (`pre_train_test_lengths[0]+1`)
while the residual vectors have `pre_T+1` entries, looped `range(pre_T+1)` so `i=pre_T` hit the
`[-0:]` whole-list slice, and never permuted the post index. The replacement yields `L` genuine
cyclic shifts of the correct length with the post period (last index) rotated through.

### P-7 — CWZ inference must re-estimate the counterfactual under H₀ (B/D)

The validity of CWZ (2021) rests on residuals being (approximately) exchangeable under H₀, which
requires re-fitting the counterfactual with post outcomes shifted by θ₀ before permuting. `pvalue_calc`
(1543–1573) instead fits once and shifts only `actual`, so pre-residuals are in-sample and
post-residuals out-of-sample even under H₀. Fix by threading a refit callback:

```python
def pvalue_calc(fit_counterfactual, y_obs, permutation_list, pre_pst_lengths, h0=0):
    y_star = y_obs.copy()
    y_star[-pre_pst_lengths[1]:] -= h0            # impose H0: subtract θ0 from post
    cf = fit_counterfactual(y_star)               # RE-ESTIMATE on the shifted series
    resid = np.abs(y_star - cf)
    S_q = conformal_inf.test_statS(1, pre_pst_lengths, resid[-pre_pst_lengths[1]:])
    S_pi = [conformal_inf.test_statS(1, pre_pst_lengths,
              resid[np.asarray(r)][-pre_pst_lengths[1]:]) for r in permutation_list]
    return 1 - np.mean(np.array(S_pi) < S_q)
```

`fit_counterfactual` is supplied by each caller (SC/DiD/SDID) and refits its model on `y_star`.
Effort Medium–High; correctness-critical for the library's headline inference claim.

### P-8 — DI hyperparameter tuning must be out-of-sample (A/D)

`alpha_lambda_diff` (1276–1297) scores in-sample MSE (monotone in the penalty → optimum α→0). Score
on a held-out time window:

```python
def alpha_lambda_diff(alpha_lambda_raw_, control_pre, holdout_frac=0.25):
    a_l = alpha_lambda.alpha_lambda_transform(alpha_lambda_raw_)
    T = len(control_pre); h0 = max(1, int(round(T*(1-holdout_frac))))
    errs = []
    for u in control_pre.columns:
        yt = control_pre[u]
        yc = control_pre[[l for l in control_pre.columns if l != u]]
        w = di.estimate_mu_omega(yt.iloc[:h0], yc.iloc[:h0], a_l)
        pred = w['mu'] + np.dot(yc.iloc[h0:], np.asarray(w['omega']).ravel())
        errs.append(np.mean((yt.iloc[h0:].to_numpy() - pred)**2))
    return float(np.mean(errs))
```

### P-9 / P-15 — dead API and invalidated MC evidence (C/A)

- `did.twfe_conformal` and `sc.sc_att` referenced by the tests/MC scripts do not exist. Either add
  thin wrappers with those names or update the callers to `did.conformal_inference` / the real SC
  ATT accessor. Add to `class did`: `twfe_conformal = conformal_inference` (alias) — or a purpose-built
  wrapper — and fix `run_200_mc.py:71`'s `sc.sc_att` call.
- The MC scripts feed `1000*log(α)` into `alpha_lambda_transform` (exp → α¹⁰⁰⁰ ≈ 0). Replace
  `1000*np.log(alpha_cv)` with `np.log(alpha_cv)` (run_2x2_mc.py:72, monte_carlo_100.py:103) so the
  DI model is the tuned elastic net rather than unregularized OLS; regenerate the affected tables.
- Remove the bare `except` that hides all-NaN SC columns (run_200_mc.py) so failures surface.

### P-10 — remaining panelib items

| Lines | Fix |
|---|---|
| 520 SDID placebo p-value | `(1 + np.sum(np.abs(placebo_atts) >= np.abs(real_att))) / (1 + len(placebo_atts))` (exact, never 0); with N_tr>1 assign N_tr pseudo-treated units per draw |
| 873, 879 | pass `alpha=alpha` through `collect_sc_outputs` instead of the hardcoded `0.05` |
| 965–968 | fix the empty grid: `step=max(abs(center)/100,1e-6); np.arange(center-500*step, center+500*step, 5*step)` |
| 1196 | `pd.DataFrame(zip(control_pre.columns, elnet.coef_))` (control-unit labels, not treated) |
| 549–551 | when N_tr>1, aggregate observed/counterfactual to the per-date treated mean before conformal, so lengths match `pre_pst_lengths` |
| 432, 1089 | `pivot_table(...)` silently averages duplicate `(unit,date)` cells — validate uniqueness and raise, or make `aggfunc` explicit |
| 628/674/1389/1421 | capture `fmin_slsqp(..., full_output=True)` and warn on non-zero `imode` |
| 45 | wrap `from IPython.display import display` in `try/except ImportError` for non-notebook use |

---

## Suggested PR commit sequence

1. **Trivial correctness** (one commit): S-4, S-8/9, S-13(guard), P-5, P-10(alpha, p-value, grid,
   di-labels), MC `log(α)` fix. No behavioral risk; unblocks the crashing modules.
2. **Cross-fitting & IPW/DML point estimates**: S-1, S-2/3, S-5/S-12, S-6/S-7. Add regression tests
   comparing to hand-computed AIPW/Hájek on a fixed seed.
3. **SDID & event-study**: P-1, P-2, P-3(guardrail), P-6, P-8. Add the audit's `repro4b`/`repro1`/
   `repro6` as tests.
4. **Inference upgrades**: P-4 (pooled cluster + wild bootstrap), P-7 (re-estimation under H₀),
   S-10 (Oster port + published-example test), S-15 (bootstrap clustering).
5. **New estimator**: P-3 full Callaway–Sant'Anna / Sun–Abraham `did.cs_att`.
