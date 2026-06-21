"""
pytest test suite for did.twfe_conformal()
CWZ conformal inference for DiD (Chernozhukov, Wüthrich & Zhu, JASA 2021).

Run with:
  C:\\Users\\hsuju\\anaconda3\\python.exe -m pytest test_did_conformal.py -v
"""
import sys
import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, r'C:\Users\hsuju\OneDrive\Documents\GitHub\statanomics\workingcode\panelmodels')
from panelib import did, dgp

DATA_DICT = {
    'treatment': 'treated',
    'date':      'time',
    'post':      'post',
    'unitid':    'unit_id',
    'outcome':   'y',
}
SIM_KWARGS = dict(seed=42, T_pre=20, T_post=5, N_control=30, noise_sd=0.1, att_pct=0.15)


@pytest.fixture(scope='module')
def pt_holds():
    df, true_att = dgp.simulate_panel(**SIM_KWARGS, sigma_lambda=0.0)
    return df, true_att


@pytest.fixture(scope='module')
def pt_violated():
    df, true_att = dgp.simulate_panel(**SIM_KWARGS, sigma_lambda=1.0)
    return df, true_att


@pytest.fixture(scope='module')
def result_pt_holds(pt_holds):
    df, _ = pt_holds
    return did.twfe_conformal(data=df, data_dict=DATA_DICT)


@pytest.fixture(scope='module')
def result_pt_violated(pt_violated):
    df, _ = pt_violated
    return did.twfe_conformal(data=df, data_dict=DATA_DICT)


# ── Return structure ────────────────────────────────────────────────────────

class TestReturnStructure:
    def test_all_keys_present(self, result_pt_holds):
        expected = {'att', 'pvalue', 'ci', 'ci_lower', 'ci_upper',
                    'counterfactual', 'gap', 'summary'}
        assert expected.issubset(result_pt_holds.keys())

    def test_att_is_float(self, result_pt_holds):
        assert isinstance(result_pt_holds['att'], float)

    def test_pvalue_in_unit_interval(self, result_pt_holds):
        p = result_pt_holds['pvalue']
        assert 0.0 <= p <= 1.0

    def test_ci_lower_lt_upper(self, result_pt_holds):
        r = result_pt_holds
        assert r['ci_lower'] < r['ci_upper']

    def test_ci_tuple_matches_scalar_keys(self, result_pt_holds):
        r = result_pt_holds
        assert r['ci'][0] == r['ci_lower']
        assert r['ci'][1] == r['ci_upper']

    def test_att_within_ci(self, result_pt_holds):
        r = result_pt_holds
        assert r['ci_lower'] <= r['att'] <= r['ci_upper']

    def test_counterfactual_is_series(self, result_pt_holds):
        assert isinstance(result_pt_holds['counterfactual'], pd.Series)

    def test_counterfactual_length(self, result_pt_holds, pt_holds):
        df, _ = pt_holds
        T = df['time'].nunique()
        assert len(result_pt_holds['counterfactual']) == T

    def test_summary_is_dataframe(self, result_pt_holds):
        assert isinstance(result_pt_holds['summary'], pd.DataFrame)


# ── CWZ formula correctness ─────────────────────────────────────────────────

class TestCWZFormula:
    def test_counterfactual_matches_cwz_formula(self, result_pt_holds, pt_holds):
        """Plotting CF = ctrl_avg_t + mean_pre(Y_treat − ctrl_avg) (pre-period baseline only)."""
        df, _ = pt_holds
        T_pre = SIM_KWARGS['T_pre']
        y_treat = (df[df['treated'] == 1]
                   .sort_values('time')['y'].values)
        ctrl_avg = (df[df['treated'] == 0]
                    .groupby('time')['y'].mean()
                    .sort_index().values)
        expected_gap = float(np.mean((y_treat - ctrl_avg)[:T_pre]))
        expected_cf  = ctrl_avg + expected_gap
        np.testing.assert_allclose(
            result_pt_holds['counterfactual'].values, expected_cf, rtol=1e-6)

    def test_gap_matches_formula(self, result_pt_holds, pt_holds):
        df, _ = pt_holds
        y_treat = (df[df['treated'] == 1]
                   .sort_values('time')['y'].values)
        ctrl_avg = (df[df['treated'] == 0]
                    .groupby('time')['y'].mean()
                    .sort_index().values)
        expected_gap = float(np.mean(y_treat - ctrl_avg))
        assert abs(result_pt_holds['gap'] - expected_gap) < 1e-9

    def test_att_equals_standard_did(self, result_pt_holds, pt_holds):
        """ATT = mean_post(Y_treat − ctrl_avg) − mean_pre(Y_treat − ctrl_avg)."""
        df, _ = pt_holds
        T_pre = SIM_KWARGS['T_pre']
        y_treat = (df[df['treated'] == 1]
                   .sort_values('time')['y'].values)
        ctrl_avg = (df[df['treated'] == 0]
                    .groupby('time')['y'].mean()
                    .sort_index().values)
        diff = y_treat - ctrl_avg
        expected_att = float(np.mean(diff[T_pre:])) - float(np.mean(diff[:T_pre]))
        assert abs(result_pt_holds['att'] - expected_att) < 1e-9

    def test_att_close_to_twfe_att(self, pt_holds):
        """CWZ DiD ATT ≈ TWFE ATT (same underlying estimator, minor FE difference)."""
        df, _ = pt_holds
        r_cf   = did.twfe_conformal(data=df, data_dict=DATA_DICT)
        r_twfe = did.twfe(data=df, data_dict=DATA_DICT)
        twfe_att = float(r_twfe['twfe']['coef_'].iloc[0])
        rel_diff = abs(r_cf['att'] - twfe_att) / max(abs(twfe_att), 1e-8)
        assert rel_diff < 0.15, (
            f"CWZ ATT={r_cf['att']:.4f} vs TWFE ATT={twfe_att:.4f}, "
            f"relative diff={rel_diff:.3f} exceeds 0.15")


# ── Inference validity ───────────────────────────────────────────────────────

class TestInference:
    def test_ci_covers_true_att_pt_holds(self, result_pt_holds, pt_holds):
        """Conformal CI must contain the true ATT when PT holds (seed=42)."""
        _, true_att = pt_holds
        r = result_pt_holds
        assert r['ci_lower'] <= true_att <= r['ci_upper'], (
            f"True ATT {true_att:.4f} not in "
            f"[{r['ci_lower']:.4f}, {r['ci_upper']:.4f}]")

    def test_pvalue_significant_pt_holds(self, result_pt_holds):
        """With att_pct=0.15 and noise=0.1 the effect should be detectable."""
        assert result_pt_holds['pvalue'] < 0.20

    def test_no_nan_inf(self, result_pt_holds):
        r = result_pt_holds
        for key in ('att', 'pvalue', 'ci_lower', 'ci_upper', 'gap'):
            val = r[key]
            assert not np.isnan(val),  f"{key} is NaN"
            assert not np.isinf(val),  f"{key} is Inf"

    def test_no_nan_inf_pt_violated(self, result_pt_violated):
        r = result_pt_violated
        for key in ('att', 'pvalue', 'ci_lower', 'ci_upper'):
            val = r[key]
            assert not np.isnan(val),  f"{key} is NaN (PT violated)"
            assert not np.isinf(val),  f"{key} is Inf (PT violated)"


# ── Comparison with OLS SE ───────────────────────────────────────────────────

class TestComparison:
    def test_conformal_ci_valid_width(self, pt_holds):
        """Conformal CI should have positive width (non-degenerate)."""
        df, _ = pt_holds
        r = did.twfe_conformal(data=df, data_dict=DATA_DICT)
        assert (r['ci_upper'] - r['ci_lower']) > 0

    def test_conformal_ci_wider_than_ols_pt_holds(self, pt_holds):
        """Conformal CI >= OLS CI width (conformal trades efficiency for robustness)."""
        df, _ = pt_holds
        r_cf   = did.twfe_conformal(data=df, data_dict=DATA_DICT)
        r_twfe = did.twfe(data=df, data_dict=DATA_DICT)
        se_ols = float(r_twfe['twfe']['se_'].iloc[0])
        ols_width       = 2 * 1.96 * se_ols
        conformal_width = r_cf['ci_upper'] - r_cf['ci_lower']
        assert conformal_width >= ols_width * 0.5, (
            f"Conformal width {conformal_width:.4f} unexpectedly much narrower "
            f"than OLS width {ols_width:.4f}")
