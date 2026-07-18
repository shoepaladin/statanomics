# Econometric audit: `stnomics.py` (statanomics) and `panelib.py` (panellibs)

Independent line-by-line audit of the two estimation libraries in this repo, conducted from the
standpoint of an economist/statistician checking each estimator against its source literature
(Chernozhukov et al. 2018 [CCDDHNR]; Hirano–Imbens–Ridder 2003; Kennedy 2020/2023; Oster 2019;
Arkhangelsky et al. 2021; Callaway–Sant'Anna 2021; Sun–Abraham 2021; Goodman-Bacon 2021;
Bertrand–Duflo–Mullainathan 2004; Chernozhukov–Wüthrich–Zhu 2021 [CWZ]; Angrist 1998).

Every finding cites exact lines. Findings marked **SIM** were verified by running simulated-data
reproductions (scripts in `audit/replication/`; `v*.py` target `stnomics.py`, `repro*.py` target
`panelib.py`; all run in under ~60s each on numpy 2.4.6 / pandas 3.0.3 / statsmodels 0.14.6 /
sklearn 1.9.0). Findings marked **THEORY** were argued analytically where simulation was not worth
the compute (or requires econml, which is not installable here).

**Severity classes**

- **A** — silently wrong point estimates (worst: no error, wrong number)
- **B** — wrong inference (SEs, p-values, CIs, test size)
- **C** — crash / unrunnable code path
- **D** — conceptual limitation, robustness, estimand mislabeling

**Effort**: Trivial (≤2 lines), Low (≤10 lines), Medium (ineed a rewrite of a function's logic),
High (new estimator required).

---

## Master ranking

Ranked primarily by how wrong the output is (A > B > C > D — a silent bias is worse than a crash,
because a crash at least announces itself), secondarily by how cheap the fix is (cheap fixes to bad
bugs are the highest-leverage items). Top-15 across both packages:

| Rank | File:lines | Defect (short) | Class | Effort | Verified |
|---|---|---|---|---|---|
| 1 | `panelib.py:613,626-638,656,672-684` | SDID weight programs: intercept constrained ≥0 **and** penalized; ω penalty missing T_pre; λ ridge ×N_co·ζ² instead of ~0 → ATT −22% biased in a clean DGP | A | Low | SIM |
| 2 | `stnomics.py:48-56` (+78-117) via `:626,655,674` | Cross-fitted predictions returned in fold order, consumed in row order → residualization on garbage whenever splits aren't block-contiguous (corr with truth ≈ 0) | A | Low | SIM |
| 3 | `stnomics.py:855-857` | IPW "ATT" = P(D=1)·ATT (unnormalized, missing ÷p̂) | A | Low | SIM |
| 4 | `stnomics.py:376-377` | `bootstrap.go_reps` copy-paste: ATT keys filled with ATE values → `ate.ipw`, `ate.propbinning` report ATE as ATT | A | Trivial | SIM |
| 5 | `panelib.py:103-109,186-206,244,272` | Event-study counterfactual merges coefficient's **positional period index** against **calendar date**, `fillna(0)` → treatment adjustment silently dropped unless dates are 0,1,2,… | A | Low | SIM |
| 6 | `stnomics.py:1021-1025` | DML-IRM ATT score wrong (control weight 1, not ê/(1−ê); `that` cancels algebraically) → orthogonality/double-robustness destroyed; bias 20× the correct score's under nuisance misspecification | A | Low | SIM |
| 7 | `panelib.py:133-141,178-206,227-252` | Staggered adoption accepted without warning: late-cohort ATT −54% biased; "pre-trend" F-test rejects (p≈0) under exact parallel trends | A | High | SIM |
| 8 | `stnomics.py:456,462,472-480,495,521-528` | Oster δ: "R²" is √(corrcoef) not R²; δ formula mangled vs Oster (2019)/psacalc (A-term power, doubled cubic); delta_ml mixes in-sample and cross-fitted R² | A | Med | SIM+THEORY |
| 9 | `stnomics.py:190,215,233` | `second_stage` CATE prediction adds the fitted **intercept** to every CATE (positional design mismatch) | A | Low | SIM |
| 10 | `stnomics.py:857` | IPW "SE" = `np.std` of the HT summand (no ÷√n, wrong summand) — 85× the Monte-Carlo sd | B | Low | SIM |
| 11 | `panelib.py:146,148,213-222` | iid OLS SEs on unit×post treatment with serially correlated errors: 35% rejection at nominal 5% (BDM 2004) | B | Low–Med | SIM |
| 12 | `panelib.py:1515-1522` | CWZ conformal permutation set padded with ~87% **iid** permutations → 19% size at 5% under ρ=0.9; padding rationale backwards (small Π is only conservative) | B | Trivial | SIM |
| 13 | `stnomics.py:242-245,327-330,1193-1195` | Prediction SE = √\|x′Vβ̂\| — covariance contracted against the coefficient vector instead of √(x′Vx) — dimensionally wrong, ratios to correct SE 0.53–1.19 (noise) | B | Low | SIM |
| 14 | `stnomics.py:829` | propbinning "SE" = cross-sectional sd of τ̂(Xᵢ), not an SE; does not shrink with n | B | Trivial | SIM |
| 15 | `panelib.py:896-902,912-915` | Per-period conformal: permutations built over the wrong index set with a `[-0:]` slice bug — wrong lengths, duplicated indices, post index never permuted | B | Med | SIM |

Crashes (Class C) rank below these because they are self-announcing, but note that **the entire
`hte.het_dml_approaches` module (HR, SGCT) has never been runnable in any environment** (finding
S-8), and the panel test suite targets functions that do not exist (finding P-9).

---

## Part I — `stnomics.py` (statanomics)

### S-1. Cross-fitting order/misalignment landmine — A, Low effort, SIM

`predict_cv` (48–56), `predict_treatment_indicator` (78–86), `predict_counterfactual_outcomes`
(89–107), `predict_continuous` (109–117) all `.extend()` predictions fold-by-fold (r = 0…K−1) and
return `np.array(...)`, which every caller subtracts from dataframe columns **in row order**.
Alignment therefore requires the split column to be block-contiguous in row position. `block_splits`
(60–74) guarantees that, so `dml_plm`/`dml_irm`/`ipw_main` are internally safe — but
`diagnostics.balance.feature_balance` assigns `df['splits'] = np.random.choice(5, len(df))` at line
626 and then calls `predict_cv` at 655 and 674: the returned x̂ is in fold-grouped order while
`df[x]` is in original order, so the DML-mode balance regressions at 656/675 run on scrambled
residuals.

```python
X = rng.normal(size=(2000,3)); target = X @ np.array([1.,-2.,.5])   # deterministic
d = pd.DataFrame(X, columns=['a','b','c']); d['tgt'] = target
d['splits_random'] = rng.integers(0, 4, 2000)                        # as in feature_balance:626
xhat = st.predict_cv(d, 'splits_random', 4, ['a','b','c'], 'tgt', LinearRegression())
# corr(xhat, target) = 0.005, RMSE = 3.25
st.block_splits(d, 'splits_block', 4)
xhat2 = st.predict_cv(d, 'splits_block', 4, ['a','b','c'], 'tgt', LinearRegression())
# corr = 1.000, RMSE = 3e-15
```

**Fix** (~3 lines per helper): preallocate `y_hat = np.empty(len(dataset))` and assign
`y_hat[test.to_numpy()] = prediction` inside the loop. Makes every helper order-agnostic and
retires this entire bug class (highest-leverage single change in the package). Script: `v6_manual.py`.

### S-2/S-3/S-4. `ate.ipw_main` (840–859) — ATT, SE, and go_reps — A/B, Low, SIM

- **ATT (855–857), Class A**: `np.average((ipw_a − ipw_b)·ê)`. Algebra: `ipw_a·ê = Y·D`,
  `ipw_b·ê = Y(1−D)·ê/(1−ê)`, so the mean estimates E[DY] − E[ê·Y(0)] = **P(D=1)·ATT**. Verified:
  reports 0.555 against true ATT 1.112 with p = 0.5 — off by exactly p̂.
- **SE (857), Class B**: `np.std(ipw_a − ipw_b)` is the cross-sectional sd of the Horvitz–Thompson
  summand, not the SE of its mean: missing ÷√n and not the efficient-score variance (HIR 2003).
  Verified 85× the Monte-Carlo sd over 200 reps.
- **`bootstrap.go_reps` (376–377), Class A**: the ATT output keys are populated from `ate_te`/`ate_se`
  — copy-paste — so the bootstrap wrappers `ate.ipw` (877) and `ate.propbinning` (783) return ATE
  labeled as ATT. Verified with a stub estimator returning ATE=1, ATT=2 → `'ATT mean': 1.0`.
- The ATE itself (852–854) is unnormalized Horvitz–Thompson rather than Hájek — consistent
  (verified mean 1.002) but inefficient and not what the HIR citation in the docstring (833–838)
  implements — Class D.

**Fixes**: Hájek-normalize ATE (`ipw_a.sum()/(D/ê).sum() − ipw_b.sum()/((1−D)/(1−ê)).sum()`);
ATT = `(Y·D).mean()/D.mean() − (Y(1−D)ê/(1−ê)).mean()/((1−D)ê/(1−ê)).mean()`; SE from the influence
function `np.std(ψ)/√n`; in go_reps, `'ATT mean': att_te[0]` etc. — four one-liners. Script: `v1_ipw.py`.

### S-5. DML-IRM ATT score is not Neyman-orthogonal (1021–1025) — A, Low, SIM

Lines 1022–1023: `att_second_fraction = that·(D==0)·(Y−m̂₀)/(that·p̂)` — the `that` cancels exactly,
leaving ψ̂ = [1{D=1}(Y−m̂₀) − 1{D=0}(Y−m̂₀)]/p̂. The CCDDHNR (2018) IRM-ATT score is

ψ = [D(Y−m₀(X)) − ê(X)(1−D)(Y−m₀(X))/(1−ê(X))]/p̂ − θ·D/p̂.

The code's control-group weight is 1 instead of ê/(1−ê): both have the same expectation when m̂₀ is
correct, so consistency survives **only through the outcome model** — orthogonality and double
robustness are gone. Verified with misspecified linear m̂₀ + correct logistic ê on identical
cross-fitted nuisances: bias +0.078 (t≈15 across 200 reps) vs +0.004 for the correct score. The ATE
score (992–998) is the correct AIPW score (verified). **Fix** (2 lines):
`att_second_fraction = that*(D==0)*y_control_residual/((1-that)*prob_unconditional)` plus the SE
centering below. Script: `v2_dml_irm.py`.

### S-6/S-7. `secondstage` SEs and intercept leak (172–335; also 1193–1195) — A+B, Low, SIM

- **SE (242–245, 327–330, and `het_ols` 1193–1195), Class B**: `output_se = sqrt(|X₋₀ V β̂₋₀|)`
  contracts the covariance matrix with the **coefficient vector**. The prediction SE is
  `sqrt(diag(X V X′))`. Verified ratios to the correct SE of 0.53–1.19 — essentially random.
  Fix: `np.sqrt(np.einsum('ij,jk,ik->i', Xt, V, Xt))`.
- **CATE prediction (190, 215, 233), Class A**: `X_test = [cons, ones, het]` is dotted positionally
  against params `[const, t, t_x…]`, so the fitted intercept is added one-for-one to every CATE.
  Verified: mean(τ̂ − τ) = 0.699 with fitted const = 0.702. Small when the second-stage outcome is
  well-residualized (const≈0), unbounded otherwise. Fix: drop the `cons` column and dot `[1, x]`
  against `params[1:]`. Note `second_stage_no_interactions`' OLS branch (263–270) is correctly
  aligned; only its SEs are wrong.
- **CVLasso branches (211–216, 295–299), Class B/D, THEORY**: selection on the train half but OLS
  refit + inference + prediction all on the **same test half** — invalid post-selection inference
  and in-sample CATEs (violates the sample-splitting that makes Kennedy-style DR-learners work).
- **`approach='Lasso'` (219, 305), Class C**: `lasso_alpha` undefined → NameError. Line 282 refits
  lasso on the unmasked X after computing the NaN/inf mask → ValueError with invalid rows.

Scripts: `v3_misc.py`, `v4_more.py`.

### S-8/S-9. The HTE-DML module has never run (1121, 1380, 1409, 1476, 1504) — C, Trivial, SIM

`np.int` (1121, 1380, 1476) crashes on numpy ≥ 1.24. Patch that and `HR` (1409) and `SGCT` (1504)
then throw `ValueError: too many values to unpack (expected 3)` — `second_stage` returns a 4-tuple
(253). So **HR and SGCT crash in every environment, old numpy or new** — they cannot have produced
output since the 4th return value was added. Fixes: `int(...)`; add `, _` to the two unpack sites.

### S-10. Oster coefficient-stability diagnostics (439–529) — A, Medium, SIM+THEORY

Three independent defects, each fatal on its own:

1. **R-metric (456, 462, 495)**: `np.sqrt(np.corrcoef(pred, y)[0,1])` = r^(1/2), not R². Verified:
   true out-of-sample R² 0.787 → code reports 0.942; NaN whenever out-of-fold correlation < 0.
   Same defect in `predQC.r2` (164–165).
2. **δ formula (472–479 = 521–528), THEORY** (checked against Oster 2019 Prop. 3 / `psacalc.ado`):
   the code's `A` term uses (β̇−β̃)² where the paper has first power in the quadratic term, and adding
   `2A` to the numerator also doubles the cubic term. The correct expressions:
   - num = (β̃−β*)(R̃−Ṙ)σ_y τ_x + (β̃−β*)σ_x τ_x(β̇−β̃)² + 2(β̃−β*)²τ_x(β̇−β̃)σ_x + (β̃−β*)³(τ_xσ_x−τ_x²)
   - den = (Rmax−R̃)σ_y(β̇−β̃)σ_x + (β̃−β*)(Rmax−R̃)σ_y(σ_x−τ_x) + (β̃−β*)²τ_x(β̇−β̃)σ_x + (β̃−β*)³(τ_xσ_x−τ_x²)
3. **delta_ml (494–495 vs 500–501)**: Ṙ is in-sample `rsquared`, R̃ is cross-fitted — different
   scales; the sign of (R̃−Ṙ) can flip mechanically. β̃ is a DML ATE while β̇ is an OLS slope —
   defensible only in a homogeneous-linear world; document.

**Fix**: transcribe from psacalc and unit-test against a published psacalc example (~10 lines but
requires care).

### S-11. `propbinning` (768–829) — B, Trivial, SIM

Point estimates align correctly (the sort at 802 matches the extend order — verified). But 829
returns `np.std(treatment_estimate)` — the dispersion of τ̂(Xᵢ) across units — as "ATE SE": it
converges to sd(τ(X)), not 0 (verified: n 2000→8000, "SE" 0.61→0.49). The bootstrap wrapper is the
right idea but its ATT is destroyed by S-4 and it reports the bootstrap mean instead of the
full-sample point estimate. Also `pd.qcut` at 799 needs `duplicates='drop'`.

### S-12. `dml_irm` variance details (1004–1016, 1027–1034) — B, Low, SIM+THEORY

- 1004–1007 are dead code (`j0 = 1 − 2·mean(score) + mean(score²)` is not any CCDDHNR Jacobian;
  overwritten twice).
- After trimming (999–1001), `SE = sqrt(var_hat)/sqrt(len(data_est))` divides by **full** n while
  the score contains only kept rows: verified 1.34× SE understatement at 44% trimming. Divide by the
  kept count (and flag the estimand as trimmed-population).
- ATT SE (1028–1034) centers the score at its mean instead of subtracting θ·D/p̂ — omits the
  θ²Var(D)/p̂² and cross terms from the asymptotic variance. One line once S-5 is fixed.

### S-13. `dml_plm` (923–963) — B mild + C, Trivial

Point estimate is the correct Robinson/DML2 partialling-out estimator (verified: mean 0.996, truth
1.0). SE is homoskedastic OLS — use `.fit(cov_type='HC1')` (the HC1 sandwich on the residual
regression equals the CCDDHNR variance up to folds). Line 963 labels ATT=ATE — under heterogeneity
the PLM identifies a variance-weighted projection, neither ATE nor ATT (document). The
`aux_dictionary['yhat']` logic (938–942) crashes: `yhat=None` → `(None != None).any()` AttributeError;
key-present-but-falsy leaves `outcome_hat` undefined → NameError (the `else` at 942 pairs with the
outer `if`).

### S-14. `hte.other.DR` (1059–1146) — D, Low, SIM

The pseudo-outcome (1080–1084) is the correct AIPW/DR-learner transform (verified corr(τ̂,τ)=0.976
in a well-overlapped DGP). Defects: trimming is commented out (1113–1117) so `aux['lower'/'upper']`
is silently ignored — with thin overlap 1/ê explodes and the downstream NaN/inf masks don't catch
large-but-finite values; the half split (1120–1122) is positional, not random, and train-half
pseudo-outcomes embed nuisances trained partly on test-half rows (violates Kennedy's independence
conditions); `output_se_hat = −1` sentinel (1099) is a footgun; `np.int` crash (S-9).

### S-15. Remaining stnomics findings (brief)

| Lines | Defect | Class |
|---|---|---|
| 139–153 | `predQC.battery`: bare `outcome(...)` → NameError; arg `treatment` shadows `predQC.treatment` → TypeError. Never worked | C |
| 346–362 | `bootstrap.reps`: iid row resampling — no cluster option; duplicated rows straddle CV folds within reps (overfit nuisances); hard-coded `args[1..8]` arity | B/D |
| 764, 918, 455, 461, 494 | `params[1]` int-key on labeled Series → KeyError on pandas ≥ 3 (position itself was correct: const first) | C |
| 758–764 | `ols_vanilla` reports ATT=ATE; OLS coefficient is the treatment-variance-weighted average (Angrist 1998), neither ATE nor ATT | D |
| 903–918 | `ipw_wls`: weights **correct** (ATE and ATT); WLS `bse` conditions on ê — ignores first-stage estimation (conservative for ATE per HIR; not guaranteed for ATT) | D |
| 60–74 | `block_splits`: positionally correct even with non-default index (verified) but mutates the caller's frame, silently overwrites user split columns in every estimator, and assumes pre-shuffled rows (sorted input → unbalanced folds) | D |
| 695 | `feature_balance`: `DataFrame.append` removed in pandas ≥ 2.0 → crash | C |
| 1329–1341 | `HR` residualizes D·xₖ on X instead of (D−ê)xₖ (SGCT at 1467 does it right); population-equivalent iff xₖ ⊆ X, strictly noisier, inconsistent otherwise | D |
| 162, 1097/1396/1491, 21 | `mape` div-by-zero; `== None` instead of `is None`; module-level suppression of chained-assignment warnings for the user's whole session | D |

**Sound as-written** (on period-correct numpy/pandas): `dml_plm` point estimate, `dml_irm` ATE point
estimate and untrimmed SE, `ipw_wls` point estimates, the DR pseudo-outcome.

---

## Part II — `panelib.py` (panellibs)

Scope: dummy-saturated TWFE DiD with per-unit ATTs + event study (`did`, 82–334); synthetic DiD per
Arkhangelsky et al. 2021 (`sdid`, 340–685); ADH / Doudchenko–Imbens / constrained-lasso synthetic
controls; CWZ-style conformal inference (1496–1658). No Callaway–Sant'Anna, no Sun–Abraham, no unit
bootstrap, no lagged DVs (no Nickell exposure).

### P-1. SDID weights deviate from Arkhangelsky et al. (2021) three ways (613, 626–638, 656, 672–684) — A, Low, SIM

The paper solves ω̂ = argmin over ω₀ ∈ ℝ (free intercept), ω ∈ simplex, of the pre-period fit plus
**ζ²·T_pre·‖ω‖²** (intercept unpenalized); λ̂ carries only a vanishing uniqueness ridge (~10⁻⁶σ̂).
The code:

- 636/682: `bounds=[(0.0, np.inf)]*len(initial_omega)` — element 0 is the intercept, so **ω₀ ≥ 0
  and λ₀ ≥ 0**. When the treated unit sits below the weighted controls, the intercept pins at 0 and
  the unit weights distort to chase the level gap instead of matching trends.
- 613: `zeta**2 * np.sum(omega_array**2)` — penalizes the intercept and omits the T_pre factor
  (under-regularizes by ×T_pre).
- 656: `zeta**2 * N_co * np.sum(lambda_array**2)` — a *large* ridge on λ (shrinks toward uniform);
  the paper wants ≈0.

Verified (treated = mean(controls) − 5, common trends, true ATT 2.0, 40 seeds):

```
library twfe_sdid : mean = 1.559   bias = −0.441 (−22%)   sd = 0.453
corrected weights : mean = 2.006   bias = +0.006           sd = 0.033
```

The ATT formula itself (429–455) is correct — the bias enters purely through the weights. **Fix**
(~10 lines): free the intercept bound, penalize `x[1:]` only, multiply the ω penalty by T_pre,
replace λ's ridge with ~1e-6·σ̂. Scripts: `repro4_sdid_weights.py`, `repro4b_sdid_fix.py`.

### P-2. Event-study counterfactual: positional index merged against calendar date (103–109, 186–206, 244, 272) — A, Low, SIM

Post-treatment dummies are named `pst_treat_{i}_{unit}` with `i` the **enumeration index** (186),
and `time_period = int(name.split('_')[-2])` (272) is therefore positional; `treated_counterfactual`
merges `left_on=[unitid, date]` vs `right_on=['treated_unit','time_period']` (104–105) and
`fillna(0)` (106). For any date variable that isn't exactly 0,1,2,… nothing matches and the
"counterfactual" equals the fitted values — silently. Verified with dates 2001–2012: event
coefficients estimated correctly (≈3.07/3.15/2.93) yet
`max |y_hats − y_hat_counterfactual| = 0.0` in the post period. Also `time_period` stays str for pre
rows (244) but int for post rows (272). **Fix**: name dummies with the actual date value or carry an
i→date map into the merge. Script: `repro1_api_and_merge.py`.

### P-3. Staggered adoption: silently accepted, badly wrong (133–141, 178–206, 227–252) — A, High, SIM

The header TODO (line 22) admits staggered support is missing, but a unit-varying `post` column runs
without warning:

- Each treated unit gets one constant `post_x_unit` dummy (135–141): with dynamic effects in an
  early cohort, the misfit dynamics load onto the common time FEs and contaminate late-cohort ATTs —
  the Goodman-Bacon/Sun–Abraham contamination mechanism operating through the time-FE estimates.
- `hold_out_time` is the max pre-period over **all** treated units (180–181), so early-cohort
  *treated* periods get coded as `pre_treat` dummies (187–195) — the "pre-trend" F-test tests
  treatment effects.

Verified (cohorts g=5 and g=15, 20 never-treated units, exact parallel trends, noise sd 0.02):
late-cohort mean coefficient 0.462 vs true 1.000 (−54%); pre-trend joint F p-value = 0.0.
**Fix**: hard-fail on unit-varying onset (cheap guardrail, do this immediately), then implement
cohort×event-time dummies with cohort-specific reference e=−1 (Sun–Abraham) or Callaway–Sant'Anna
group-time ATTs (never-treated coded explicitly, e.g. +inf). Script: `repro6_staggered.py`.

### P-4. No clustered inference anywhere in the DiD (146, 148, 213–222) — B, Low–Med, SIM

`sm.OLS(...).fit()` classical SEs feed `se_`, `y_hat_se`, event-study CIs, and both F-tests. This is
the textbook BDM (2004) failure. Verified (N=30 units, 5 treated, T=20, AR(1) ρ=0.8, zero effect,
150 reps): 35.3% rejection at nominal 5%; a pooled treated×post ATT with unit clustering: 8.0%.
Subtlety (verified): naively adding `cov_type='cluster'` to *this* per-unit-dummy specification
does not work — each `post_x_unit` coefficient is identified within a single cluster, so CRVE is
degenerate for it (any-unit rejection → 1.0). **Fix**: pooled ATT with cluster(unit) SEs and G−1
dof (wild cluster bootstrap if few treated clusters, Cameron–Gelbach–Miller 2008); per-unit ATT
inference must come from randomization/conformal methods (Conley–Taber 2011, Ferman–Pinto 2019) —
drop or flag the per-unit `se_`/`pvalue` columns. Scripts: `repro2_se_size.py`, `repro2b_se_size.py`.

### P-5. Conformal permutations padded with iid draws (1515–1522) — B, Trivial, SIM

`time_block_permutation` correctly builds the T cyclic shifts — the CWZ moving-block set whose
validity under weak dependence is the whole point — then pads with unrestricted random permutations
"so min p-value is small enough," up to 200. With T≈25, ~87% of the null distribution is iid
permutations, valid only under exchangeability; and the rationale is backwards — |Π|=T is merely
conservative (min p = 1/T), never invalid. Verified (H₀ true, AR(1) ρ=0.9 gap, 800 reps): size
18.9% padded vs 8.1% cyclic-only at nominal 5%. **Fix**: delete lines 1515–1522. Also 1520 uses
unseeded global `np.random` → non-reproducible p-values. Script: `repro3_conformal_padding.py`.

### P-6. Per-period conformal permutations malformed (896–902, 912–915) — B, Med, SIM

`collect_sc_outputs_individual` builds permutations over `np.arange(pre_train_test_lengths[0]+1)`
(train length, 897) while the residual arrays have `pre_T+1` elements (908), and the loop
`for i in range(pre_T+1)` (898) hits the `[-1*(pre_T-i):]` = `[-0:]` whole-list slice at i=pre_T.
Verified (pre_T=20, train=15): required length 21, observed lengths {16,…,20,32}; 20 of 21
permutations contain duplicates; indices 16–20 — including the post period under test — are never
permuted. Every `aggregate_pst_periods=False` p-value/CI is meaningless. **Fix**: build cyclic
shifts over `np.arange(pre_T+1)` with the same guard used in `time_block_permutation`. Script:
`repro5_misc.py`.

### P-7. CWZ without re-estimation under H₀ (543–573, 1543–1573; also 297–334, 531–565) — B/D, Med, THEORY

CWZ Algorithm 1: for each θ₀, subtract θ₀ from post-period treated outcomes, **re-fit the
counterfactual on all T periods**, then permute residuals — validity leans on residuals being
(approximately) exchangeable under H₀. `pvalue_calc` takes a counterfactual estimated once (SC on
the pre-train window; DiD/SDID with treatment dummies in) and merely shifts `actual` by θ₀:
pre-period residuals are in-sample (shrunken), post residuals out-of-sample, so the identity
ordering has systematically larger post residuals even under H₀ → over-rejection on top of P-5,
and asymmetric θ-inversion. **Fix**: move the θ₀ subtraction before estimation inside the CI loop
(that is the algorithm; costlier), or a Cattaneo–Feng–Titiunik-style prediction-interval correction.

### P-8. DI hyperparameter "CV" is in-sample (1276–1297) — A/D, Med, SIM

`alpha_lambda_diff` fits each leave-one-out placebo ElasticNet and scores MSE **on the fitting
observations**. In-sample MSE is monotone in the penalty → the "optimum" is always α→0; with
N_co > T_pre (the MC scripts use 100 vs 10) this selects an interpolating, unstable SC. Verified:
objective falls monotonically 0.664→0.145 as α: 7.39→5e-5. **Fix**: score the placebo on a held-out
time window (the split machinery already exists in `predict_mu_omega`). Script: `repro7_tuning_labels.py`.

### P-9. Crashes, dead API, and invalidated Monte-Carlo evidence — C/A, Low, SIM

- Non-string unit ids: merge of str `treated_unit` (from `r.split('_')[-1]`, 158) vs native-dtype
  `unitid` → `ValueError` (verified); ids containing `_` get truncated → silent zero-fill via the
  P-2 merge path.
- `did.twfe_conformal` (test_did_conformal.py:41 etc.) and `sc.sc_att` (run_200_mc.py:71) **do not
  exist** — the pytest suite errors on import, and run_200_mc's bare `except` makes its "SC (DI)"
  column silently all-NaN.
- run_2x2_mc.py:72 / monte_carlo_100.py:103 feed `1000*log(α)` into `alpha_lambda_transform`, i.e.
  exp(1000·ln α) = α¹⁰⁰⁰ ≈ 0 → the published MC "SC (DI)" results are unregularized OLS, not the
  tuned elastic net the scripts claim (verified).

### P-10. Remaining panelib findings (brief)

| Lines | Defect | Class |
|---|---|---|
| 507–521 | SDID placebo inference: p = mean(\|placebo\|≥\|real\|) can be exactly 0 (use (1+k)/(1+M), Firpo–Possebom 2018); placebo treats one unit even when N_tr>1; ±1.96·σ̂_placebo CI assumes normality from ≤20 draws. (Placebo dispersion per se is the right variance proxy here — these are the actual defects.) | B |
| 868–879 | `collect_sc_outputs` hardcodes `alpha=0.05`; user's `inference['alpha']` silently ignored (yet echoed in the output column, 995) | B |
| 965–968 | `theta_grid=None` fallback: `np.arange(a, a, step)` start==stop → empty grid → CI = [nan, nan] (verified) | B/C |
| 549–551 vs 1548 | `sdid.conformal_inference` with N_tr>1: N_tr·T observed rows vs lengths summing to T → AssertionError. Aggregate treated to per-date means first | C |
| 1196–1201 | DI weights table zips control-unit coefficients with **treated** column names, truncated to N_tr rows (verified) | D |
| 315–316 | `did.conformal_inference` silently uses only the first treated unit | D |
| 1214–1221, 1330–1334, 1438–1443 | Final SC/DI/CL weights fit on the 75% train window only, never refit on the full pre-period → avoidable variance | D |
| 1089–1100, 398–400, 628/674/1389/1421, 1501, 45 | `pivot_table` silently *averages* duplicate (unit,date) cells; dead `obs_weights` code; `fmin_slsqp` exit status never checked (non-convergent weights propagate); `pre_pst_lengths` from global date counts wrong for unbalanced panels; unconditional `from IPython.display import display` breaks non-notebook use | D |

**Sound as-written**: `comp_reg_parameter` ζ = (N_tr·T_post)^{1/4}·σ̂ (571–589, including the
`sqrt(sigma_hat)` conversion at 588); the SDID ATT/trajectory construction (429–455) incl. ω/column
alignment; `test_statS` with q=1 = CWZ's S(û); identity permutation included so p ≥ 1/|Π|; the θ₀
shift applied to post entries only, correct *given* the P-7 shortcut; `y_hats` index alignment at 97.

---

## Recommended fix order (marginal value per unit effort)

1. **Trivial, huge payoff**: delete `panelib.py:1515-1522` (P-5); fix `go_reps` keys (S-4); `HC1` in
   `dml_plm` (S-13); `duplicates='drop'` at `stnomics.py:799`; `np.int`→`int` and the two unpack
   fixes (S-8/9); `params[1]`→`.iloc[1]`/label.
2. **Low effort, kills Class-A bias**: order-agnostic `predict_*` helpers (S-1); Hájek IPW ATT + ÷√n
   SE (S-2/3); CCDDHNR ATT score in `dml_irm` (S-5); SDID intercept/penalty corrections (P-1);
   event-study merge on calendar dates (P-2); hard-fail on staggered `post` (P-3 guardrail);
   intercept-leak + √(x′Vx) in `secondstage` (S-6/7).
3. **Medium**: pooled clustered/wild-bootstrap DiD inference (P-4); per-period conformal index fix
   (P-6); out-of-sample DI tuning (P-8); Oster δ transcription from psacalc (S-10); trimmed-n SE and
   ATT centering in `dml_irm` (S-12).
4. **High / new estimators**: true staggered support (Sun–Abraham or Callaway–Sant'Anna) (P-3);
   CWZ re-estimation under H₀ (P-7); cluster-aware bootstrap in `stnomics.bootstrap` (S-15).
