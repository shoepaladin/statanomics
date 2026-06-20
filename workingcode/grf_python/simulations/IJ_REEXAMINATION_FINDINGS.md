# Re-examination of the IJ variance estimator

## TL;DR

The original infinitesimal-jackknife (IJ) variance was **not** algebraically
broken, and its 0% null false-positive rate (FPR) was **not** a fundamental
"you need B ≫ 9,000 trees" limitation. It had two fixable problems:

1. **Missing the finite-B Monte-Carlo bias correction.** The raw IJ sum of `n`
   squared covariances is positively biased by an O(n/B) Monte-Carlo term.
   Empirically the raw IJ SE was **2.5×–11× too large** → 0% FPR. Adding the
   Wager–Hastie–Efron (1984/2014) correction (`IJ-U`) removes that bias on
   average, but the corrected estimator is **high-variance at moderate B**
   (FPR 15–45% at B≤200), because it differences two large near-equal
   quantities.

2. **Wrong estimator structure for moderate B.** The estimator R `grf` and
   `econml.grf` actually use is the **bootstrap of little bags (BLB)** (Athey,
   Tibshirani & Wager 2019), not the raw per-tree IJ. BLB is what gives those
   packages calibrated CIs with only **100–400 trees**. We verified `econml`
   directly: FPR ≈ 2–8% at B=100–200 on the same null DGP.

**Fix shipped:** trees are now grown in groups (`subforest_size`, default 4)
that share a common half-sample, and `predict(return_std=True)` defaults to the
BLB between-group-minus-within-group variance. The delta method and the
bias-corrected IJ-U remain available via `variance={'delta','ij'}`.

## What was audited

`compute_ij_variance` (in `grf/numba_core.py`) was checked against the
Wager–Hastie–Efron / Athey–Wager formula:

* covariance centering — **correct** (`(1/B)Σ T_b·I_bj − T̄·p_j`, equal to
  `(1/B)Σ (I_bj−p_j)(T_b−T̄)`, not the earlier "deviation-sum-to-zero" form);
* scale `n/s` — **correct**;
* `subsample_flags` — **correctly built** from `tree.in_subsample_`
  (populated in `tree_numba.py`), with the right shape — they were genuinely
  passed in, not all-zeros.

So the 0% FPR was the **missing bias correction**, not a coding bug.

## Evidence 1 — raw IJ vs bias-corrected IJ-U (independent-subsample forest)

`simulations/ij_calibration.py`, 40 reps/cell, 5 null test points. Target
FPR 0.05, ratio = mean(SE)/MC-SE ≈ 1.0.

```
    n     B |  FPR_raw  rat_raw |  FPR_IJU  rat_IJU | FPR_delt rat_delt
  200    50 |   0.0000    4.703 |   0.3850    0.851 |   0.0150    1.139
  200   200 |   0.0000    2.562 |   0.3400    0.671 |   0.0400    1.096
  200  1000 |   0.0000    1.370 |   0.1500    0.763 |   0.0250    1.102
  400    50 |   0.0000    6.631 |   0.3350    1.094 |   0.0600    0.932
  400   200 |   0.0000    3.490 |   0.4100    0.614 |   0.0950    0.865
  400  1000 |   0.0000    1.669 |   0.3250    0.589 |   0.1050    0.851
  800    50 |   0.0000   10.871 |   0.3500    1.458 |   0.0450    0.995
  800   200 |   0.0000    6.431 |   0.4500    0.770 |   0.0500    0.990
```

Reading: raw IJ ratio grows with `n` (up to ~11×) → SE wildly too large →
0% FPR. IJ-U fixes the *mean* (ratio → ~1) but is erratic per-replication at
moderate B. Confirms diagnosis (1).

## Evidence 2 — econml ground truth (BLB)

`econml.grf.CausalForest`, same null DGP, 30 reps:

```
econml  n=400 B=200 | FPR=0.047  SE/MC_SE=1.029
econml  n=800 B=200 | FPR=0.067  SE/MC_SE=1.244
econml  n=400 B=100 | FPR=0.080  SE/MC_SE=1.062
econml  n=200 B=200 | FPR=0.020  SE/MC_SE=1.398
```

A correct GRF variance estimator is well-calibrated at B = 100–200. Settles
the "B ≫ 9,000" question: **false**.

## Evidence 3 — our BLB implementation

`simulations/blb_calibration.py`, grouped (little-bags) forest,
`subforest_size=4`, 40 reps/cell. FPR (5 null test pts):

```
    n     B |  FPR_blb  rat_blb | FPR_delt rat_delt |  FPR_IJU  rat_IJU
  200    50 |   0.0450    1.202 |   0.0300    1.171 |   0.2150    2.207
  200   200 |   0.0550    1.156 |   0.0150    1.171 |   0.0900    1.531
  200  1000 |   0.0350    1.192 |   0.0250    1.165 |   0.0300    1.150
  400    50 |   0.0650    1.137 |   0.0650    0.928 |   0.3200    2.722
  400   200 |   0.0950    1.016 |   0.0800    0.884 |   0.1350    1.576
  400  1000 |   0.1200    0.906 |   0.0800    0.869 |   0.0800    0.974
  800    50 |   0.0250    1.205 |   0.0550    0.957 |   0.3800    3.454
  800   200 |   0.0450    1.099 |   0.0900    0.938 |   0.3050    1.907
  800  1000 |   0.0850    0.972 |   0.0800    0.951 |   0.1150    1.158
 1600    50 |   0.0200    1.375 |   0.0700    0.996 |   0.4250    5.261
 1600   200 |   0.0400    1.327 |   0.0350    1.023 |   0.3050    2.956
 1600  1000 |   0.0600    1.132 |   0.0500    1.018 |   0.1100    1.614
```

**At the practical operating point B≈200**, BLB FPR = {5.5, 9.5, 4.5, 4.0}%
for n = {200, 400, 800, 1600} — calibrated except n=400, and at n=800 it beats
the delta method (4.5% vs 9.0%). IJ-U is still unusable below B=1000.

## The residual n=400 anti-conservatism is a forest-level (nuisance) issue

At n=400 **all three** variance methods are anti-conservative (8–12%), and the
SE/MC ratio is ≈1.0 — i.e. the mean SE is right but the studentized statistic
has heavy tails. This is not a property of the variance formula; it is the
forest point estimator at small n. See `simulations/n400_diag.py`: bypassing
the cross-fitted nuisance orthogonalization (using the true null residuals)
changes the n=400 FPR, isolating nuisance-estimation error — which none of the
forest-only variance estimators account for — as the driver:

```
n=400 B=200 bypass_nuisance=False | BLB FPR=0.095  SE/MC=1.016  MC_SE=0.161
n=400 B=200 bypass_nuisance=True  | BLB FPR=0.075  SE/MC=1.008  MC_SE=0.153
```

Removing the cross-fitted nuisance step (using the true null residuals) drops
the FPR 9.5% → 7.5% and the Monte-Carlo SE 0.161 → 0.153: nuisance-estimation
error is a real contributor that the forest-only variance does not model. It
does not fully close the gap, so there is also a residual point-estimator tail
effect at small n; both are orthogonal to the choice of variance estimator.

There is also a mild "FPR rises with B" effect (e.g. n=800: 2.5→4.5→8.5%): as
B→∞ the Monte-Carlo noise vanishes and the forest-only variance converges to a
limit that slightly undercounts nuisance uncertainty at finite n. The same
effect was documented for the delta method in `b_effect_results.txt`.

## Addendum — auditing the BLB scaling against R `grf` (the target package)

Follow-up concern: "increasing B increases FPR" looks like a denominator bug
(within-bag noise divided by B instead of by trees-per-bag k). We audited
`compute_blb_variance` line-by-line against R `grf` master.

**The variance function is a faithful port of grf.** grf's
`RegressionPredictionStrategy::compute_variance` computes:

```cpp
var_between = rho_grouped_squared / num_good_groups;
var_total   = rho_squared / (num_good_groups * ci_group_size);
group_noise = (var_total - var_between) / (ci_group_size - 1);   // ÷ (k-1), NOT B
var_debiased = bayes_debiaser.debias(var_between, group_noise, num_good_groups);
```

and `ObjectiveBayesDebiaser::debias`:

```cpp
initial_estimate = var_between - group_noise;
initial_se   = max(var_between, group_noise) * sqrt(2.0 / num_good_groups);
ratio        = initial_estimate / initial_se;
numerator    = exp(-ratio^2/2) / sqrt(2pi);
denominator  = 0.5 * erfc(-ratio / sqrt2);
return initial_estimate + initial_se * numerator / denominator;
```

Our `compute_blb_variance` matches both, term for term (the within correction
divides by `L-1 = ci_group_size-1`; the cushion divides by the number of
groups, not B). grf's two-level CI sampling is also replicated: per group draw
one half-sample (fraction 0.5), then each of the `ci_group_size` trees
subsamples `sample_fraction*2` of that half (= `sample_fraction*n`; with the
grf default `sample.fraction=0.5` every tree uses the whole half).

**So there is no denominator bug.** A deterministic synthetic test (true
between-bag var = 0.25, large within-bag noise) confirms the *core* estimate is
scale-invariant once the cushion is negligible:

```
G=400 (cushion ~0):   L=2 -> 0.292,  L=4 -> 0.250,  L=50 -> 0.249   (true 0.25)
G=20  (cushion large): L=2 -> 0.785,  L=4 -> 0.444,  L=50 -> 0.236
```

The B-dependence is entirely grf's **ObjectiveBayesDebiaser cushion**, which is
`max(var_between, group_noise) * sqrt(2/num_good_groups)`: large when there are
few groups (small B), vanishing as groups accumulate. So "FPR rises with B" is
grf's *designed* behavior — small forests are deliberately conservative; large
forests relax toward the uncushioned estimate. R `grf` exhibits the same.
(Captured by `tests/test_blb_variance.py::test_core_formula_is_scale_invariant_in_group_size`.)

**Defaults aligned to grf:** `subforest_size` 4 → **2** (`ci.group.size`),
`subsample_ratio` 0.45 → **0.5** (`sample.fraction`).  This is not cosmetic —
it materially improves calibration (`simulations/grf_defaults_bsweep.py`,
30 reps, null DGP):

```
                        FPR     SE/MC    groups
n=400  B=50    0.027   1.356     25       (was 6.5% at subforest_size=4)
n=400  B=200   0.053   1.140    100       (was 9.5%)
n=400  B=1000  0.067   0.969    500       (was 12.0%)
n=800  B=50    0.000   1.562     25
n=800  B=200   0.047   1.304    100       (was 4.5%)
n=800  B=1000  0.060   1.115    500       (was 8.5%)
```

The B-effect (FPR drifting up with B) persists but is now mild, and the
large-B values sit at 6–7% rather than 8–12%.  The drift is the grf
ObjectiveBayes cushion relaxing as groups accumulate (`sqrt(2/num_groups)`);
small forests are deliberately conservative (n=800/B=50 → 0% at just 25
groups).

## Addendum 2 — matching grf's nuisance step (OOB regression forest)

We then replaced the k-fold `RandomForestRegressor` cross-fit with grf's actual
orthogonalization: out-of-bag predictions from a regression forest with
`max(50, num.trees/4)` trees (R grf `causal_forest`:
`Y.hat <- predict(regression_forest(X, Y))$predictions`).

Direct comparison under grf defaults (subforest_size=2, subsample_ratio=0.5,
30 reps, null DGP; FPR with SE/MC in parentheses):

```
   cell        OOB nuisance     k-fold nuisance
 n=400 B=50    3.3% (1.28)      2.7% (1.36)
 n=400 B=200   7.3% (1.09)      5.3% (1.14)
 n=400 B=1000 10.0% (0.96)      6.7% (0.97)
 n=800 B=50    1.3% (1.58)      0.0% (1.56)
 n=800 B=200   2.0% (1.30)      4.7% (1.30)
 n=800 B=1000  7.3% (1.05)      6.0% (1.12)
```

Conclusion: switching to grf's exact nuisance step is **a wash within
Monte-Carlo noise** (30 reps x 5 correlated points => FPR SE ~1.8%; the
n=800/B=1000 cell alone moved 6.0%->7.3% between two nominally-similar runs).
It does NOT resolve the mid-n large-B drift, so that drift is **not** a
nuisance-method artifact.  (We keep the OOB nuisance because it is the faithful
grf replication, not because it changes calibration.)

Having now ruled out, in turn:
  * the variance formula (faithful, scale-invariant grf port);
  * the BLB cushion / B-scaling (grf-intended ObjectiveBayes behavior);
  * the nuisance method (k-fold vs OOB: a wash);
  * nuisance *presence* (bypassing it helps slightly, yet *tighter* OOB
    nuisance hurts slightly — contradictory, so nuisance is not the driver),

the most plausible remaining cause is the finite-n GRF studentized statistic
having heavier-than-normal tails at mid n (an inherent property R grf shares),
compounded by Monte-Carlo noise at 30 reps.

## Addendum 3 — high-rep confirmation (250 reps, clustered 95% CI)

`simulations/highrep_confirm.py`, current grf-faithful config; FPR with a
clustered 95% CI (per-rep rejection rate, CI across reps).  Calibrated iff the
CI brackets 0.05:

```
 n=400 B=200   FPR=0.054  CI[0.040, 0.069]  SE/MC=1.16   OK   (calibrated)
 n=400 B=1000  FPR=0.088  CI[0.069, 0.107]  SE/MC=1.02   HIGH (real)
 n=800 B=200   FPR=0.034  CI[0.024, 0.045]  SE/MC=1.33   LOW  (conservative)
 n=800 B=1000  FPR=0.050  CI[0.037, 0.063]  SE/MC=1.11   OK   (calibrated)
```

So the answer to "is the drift just simulation noise?" is **mostly yes**: three
of four cells are calibrated or conservative once powered properly (the 30-rep
n=800/B=1000 = 7.3% was noise; at 250 reps it is exactly 0.050).  The **one
genuine exception is small n with many trees** — n=400/B=1000 sits at 8.8%
(CI excludes 0.05).

Crucially its `SE/MC = 1.02`: the *mean* SE is correct, so this is **not** a
variance-magnitude error.  It is heavier-than-normal tails in the studentized
statistic `tau_hat / SE` at small n — the normal-approximation CI
(`+/- 1.96*SE`) is slightly too thin in the tails.  This is an inherent
finite-sample property of GRF that R `grf` shares (grf also forms normal CIs);
it shows up only once the ObjectiveBayes cushion has decayed (large B) and the
sample is small.  At the practical operating point (B <= 200) the cushion keeps
coverage calibrated-to-conservative.

## Addendum 4 — the `mtry` bug (resolves BOTH the non-null gap and n=400/B=1000)

A later code audit (non-null DGP, which the null-only work above could not see)
found the actual culprit behind the residual issues: the `mtry` default was
`ceil(p/3)`, which the code claimed "matches R grf" but does not.  R `grf`'s
default is `min(p, ceil(sqrt(p) + 20))` (= all features for p<=~22), verified
against grf 2.6.1.  `ceil(p/3)` (e.g. 2 of 5 features) starves splits of the
relevant covariate.

On identical data (n=800, tau(x)=x0, B=2000), benchmarked against R grf:

```
                  slope   coverage   mean|err|
 ours, mtry=2     0.61     0.57       0.285     (heavily attenuated)
 R grf            0.76     0.77       0.189
 ours, mtry fix   0.84     0.85       0.154     (matches/beats grf)
```

Crucially the fix ALSO removes the n=400/B=1000 null anti-conservatism that
Addendum 3 attributed to "inherent finite-n heavy tails" — that diagnosis was
wrong; it was under-splitting.  Null recheck after the fix (150 reps):

```
 n=400 B=200   5.6%  CI[3.6, 7.6]   OK
 n=400 B=1000  6.0%  CI[4.1, 7.9]   OK    (was 8.8% HIGH before the fix)
 n=800 B=200   3.5%  CI[2.0, 4.9]   conservative
 n=800 B=1000  5.5%  CI[3.7, 7.2]   OK
```

So with the correct grf `mtry`, every cell is calibrated or conservative, and
non-null accuracy matches R grf.  Lesson: null-only validation cannot detect
estimation bias (attenuation toward 0 is harmless under tau=0); non-null
coverage / slope-vs-truth must be part of the test suite
(`tests/test_phase1_correctness.py::TestNonNullCalibration`).

## Recommendation

* **Default to BLB** (`variance='blb'`) with grf defaults (`subforest_size=2`,
  `subsample_ratio=0.5`), grf's OOB regression-forest nuisance, and the grf
  `mtry = min(p, ceil(sqrt(p)+20))` default — a faithful R `grf` replication.
* With the `mtry` fix (Addendum 4), null FPR is calibrated/conservative across
  the n x B grid (no remaining anti-conservative cell), and non-null accuracy
  and coverage match/beat R `grf` on identical data.
* Keep `delta` and `ij` available for comparison.
* Always validate with a **non-null** DGP (slope-vs-truth + coverage), not just
  null FPR: null-only tests cannot see estimation bias.
