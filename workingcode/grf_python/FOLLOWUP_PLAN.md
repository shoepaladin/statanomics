# GRF Python — follow-up plan (deferred from PR #4)

PR #4 is scoped to inference (the variance estimator) + the `mtry` correctness
fix. These three items were intentionally deferred. Tracked in issue #5.

> **Status (issue #5 PR):** all three items implemented.
> 1. ✅ Timing test replaced with a deterministic correctness check; the
>    wall-clock guard is now opt-in behind `@pytest.mark.performance`
>    (excluded from the default run).
> 2. ✅ `n_folds` removed (clean break) — passing it now raises `TypeError`.
> 3. ✅ Invalid estimation leaves dropped (not zeroed) from the point average
>    and BLB; honest min-leaf size enforced on the estimation sample during the
>    split search. A/B: invalid (tree, point) leaf rate 0.04% → 0.0000% on the
>    n=800 harness; non-null slope ≈0.85 / coverage ≈0.85 (was 0.84 / 0.85).
>    Residual attenuation is finite-sample forest bias the normal CI does not
>    model — see the statistician critique in the PR description.

Suggested order: **#1 and #2 first (quick wins), then #3.**

---

## 1. Flaky timing test

**Problem.** `tests/test_phase2_performance.py::TestBatchPrediction::test_forest_predict_faster_than_point_loop`
asserts a wall-clock ratio (`t_batch <= t_loop * 2.0`). On tiny forests both
times are ~milliseconds, so OS scheduling / CPU contention (e.g. a sim running
alongside) flips it. Passes in isolation, fails under load — non-deterministic.

**Solutions (preferred first).**
- Replace the wall-clock assertion with a *correctness* assertion (batch vs
  point-loop predictions are identical); drop the speed claim.
- If a perf guard is wanted: larger forest, best-of-k timing, generous margin
  (~5x), behind a `performance` mark excluded from the default run.
- Or just mark it `@pytest.mark.performance` and exclude it from CI.

## 2. Dead `n_folds` parameter

**Problem.** Nuisance estimation now uses grf-style OOB regression-forest
predictions, so `n_folds` does nothing, yet it remains in `__init__`,
`get_params`, and is accepted at fit. A user setting `n_folds=10` expects k-fold
cross-fitting and gets silence.

**Solutions (preferred first).**
- Remove `n_folds` entirely (clean break; update `get_params`/tests).
- Back-compat: keep it but emit `DeprecationWarning` when set to a non-default
  and document it as ignored.
- Re-activate: `nuisance='oob'` (default) vs `nuisance='kfold'`, with `n_folds`
  applying only to k-fold. More surface area; only if k-fold is actually wanted.

## 3. Split / estimation bias (point-estimate quality)

**Problem.** Even after the `mtry` fix (slope 0.61 -> 0.84 vs R grf's 0.76 on
identical data), CATEs are still shrunk toward 0 (slope < 1) and held-out
coverage is ~85%, not 95% — finite-sample forest bias the normal CI doesn't
account for. Two concrete contributors:
- **Empty/tiny estimation leaves** — `batch_predict_from_leaves` returns 0 when
  leaf size < 2 or `W'W ~= 0` (~0.09% of (tree, point) at n=800). These silently
  inject tau=0, biasing toward the null and distorting BLB groups. R grf drops
  such leaves.
- **Honest-split min-size enforced on the split sample only** — the estimation
  child can fall below `min_leaf` (down to 0), producing those tiny leaves.

**Solutions.**
- Drop invalid (size<2 / `W'W~=0`) leaves from both the point average and the
  BLB group/within computations (mirror grf's "good group" logic) instead of
  treating them as tau=0.
- Enforce a minimum *estimation*-leaf size during splitting (reject candidate
  splits whose estimation child < `min_leaf`) — grf's `min.node.size` semantics
  on the honest sample.
- Quantify residual attenuation vs R grf with finer leaves / split tuning.
- Optional, orthogonal: t-style CI quantile for tighter small-n interval
  coverage — an inference knob, a deliberate deviation from grf, not a bias fix.

**Validation.** All bias work must use the **non-null** harness
(`tests/test_phase1_correctness.py::TestNonNullCalibration`): slope-vs-truth and
coverage. Null-only sims cannot see estimation bias — that is the core lesson
from PR #4.
