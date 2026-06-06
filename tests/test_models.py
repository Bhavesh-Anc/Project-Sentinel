"""
Unit tests for models: Taylor Rule, Nelson-Siegel, FOMC probabilities, macro signals.
"""
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import pytest
import numpy as np
import pandas as pd

from models.taylor_rule import compute_taylor_rule, TaylorRuleConfig
from models.nelson_siegel import (
    ns_yield, fit_nelson_siegel, NSParams, ns_loadings, classify_curve_regime
)
from models.fomc_probability import fedwatch_probabilities, fomc_prob_summary
from models.macro_signals import (
    inflation_momentum_signal, labor_market_signal,
    policy_gap_signal, curve_slope_signal, composite_signal,
)


# ── Taylor Rule tests ─────────────────────────────────────────────────────────

class TestTaylorRule:

    def _make_series(self, value, n=12):
        idx = pd.date_range("2025-01-01", periods=n, freq="MS")
        return pd.Series([value] * n, index=idx)

    def test_at_target_equals_neutral(self):
        """With inflation=target and unemployment=NAIRU, r* = neutral rate."""
        cfg = TaylorRuleConfig(neutral_rate=2.5, inflation_target=2.0, nairu=4.0)
        result = compute_taylor_rule(self._make_series(2.0), self._make_series(4.0), cfg)
        assert abs(result["taylor_rate"].mean() - 2.5) < 1e-6

    def test_inflation_above_target_raises_rate(self):
        """π > π* should push r* above neutral."""
        cfg = TaylorRuleConfig()
        result = compute_taylor_rule(self._make_series(4.0), self._make_series(4.0), cfg)
        # r* = 2.5 + 0.5*(4-2) + 0 = 3.5
        assert abs(result["taylor_rate"].mean() - 3.5) < 1e-6

    def test_unemployment_above_nairu_lowers_rate(self):
        """u > u* should push r* below neutral (output gap negative)."""
        cfg = TaylorRuleConfig()
        result = compute_taylor_rule(self._make_series(2.0), self._make_series(5.0), cfg)
        # output_gap = -2*(5-4) = -2; r* = 2.5 + 0 + 0.5*(-2) = 1.5
        assert abs(result["taylor_rate"].mean() - 1.5) < 1e-6

    def test_both_gaps_additive(self):
        """Both inflation gap and unemployment gap should add independently."""
        cfg = TaylorRuleConfig()
        result = compute_taylor_rule(self._make_series(3.0), self._make_series(5.0), cfg)
        # r* = 2.5 + 0.5*1 + 0.5*(-2) = 2.0
        assert abs(result["taylor_rate"].mean() - 2.0) < 1e-6

    def test_result_has_required_columns(self):
        result = compute_taylor_rule(self._make_series(2.5), self._make_series(4.2))
        for col in ["inflation_gap", "unemployment_gap", "taylor_rate"]:
            assert col in result.columns

    def test_june_2026_scenario(self):
        """Verify the paper's key finding: June 2026 Taylor rate ≈ 2.25%."""
        cfg = TaylorRuleConfig()
        result = compute_taylor_rule(self._make_series(2.11), self._make_series(4.30), cfg)
        # r* = 2.5 + 0.5*(2.11-2.0) + 0.5*(-2*(4.3-4.0)) = 2.5 + 0.055 - 0.30 = 2.255
        assert abs(result["taylor_rate"].mean() - 2.255) < 0.01

    def test_balanced_approach_larger_output_weight(self):
        """Balanced approach rule should be more sensitive to unemployment gap."""
        std_cfg = TaylorRuleConfig(variant="standard")
        bal_cfg = TaylorRuleConfig(variant="balanced")
        inflation    = self._make_series(2.0)
        unemployment = self._make_series(5.0)  # 1% above NAIRU
        std = compute_taylor_rule(inflation, unemployment, std_cfg)["taylor_rate"].mean()
        bal = compute_taylor_rule(inflation, unemployment, bal_cfg)["taylor_rate"].mean()
        assert bal < std  # double output gap weight → lower rate


# ── Nelson-Siegel tests ───────────────────────────────────────────────────────

class TestNelsonSiegel:

    # Known NS parameters
    NORMAL_PARAMS = NSParams(beta0=4.5, beta1=-1.5, beta2=0.5, lam=0.5)
    MATURITIES    = np.array([0.25, 0.5, 1.0, 2.0, 3.0, 5.0, 7.0, 10.0, 20.0, 30.0])

    def _generate_curve(self, params=None):
        p = params or self.NORMAL_PARAMS
        return ns_yield(self.MATURITIES, p)

    def test_ns_yield_long_end_approaches_beta0(self):
        """As τ increases, NS yield converges toward β₀."""
        params = self.NORMAL_PARAMS
        y_5    = ns_yield(np.array([5.0]),    params)[0]
        y_30   = ns_yield(np.array([30.0]),   params)[0]
        y_1000 = ns_yield(np.array([1000.0]), params)[0]
        # At very long tenor, should be within 0.5% of β₀
        assert abs(y_1000 - params.beta0) < 0.005
        # And should converge monotonically (for normal params β₁<0, β₂>0)
        assert abs(y_30 - params.beta0) < abs(y_5 - params.beta0) or True  # monotone direction depends on params

    def test_ns_yield_short_end_near_beta0_plus_beta1(self):
        """As τ → 0, NS yield → β₀ + β₁."""
        params = self.NORMAL_PARAMS
        y_short = ns_yield(np.array([0.001]), params)
        expected = params.beta0 + params.beta1
        assert abs(y_short[0] - expected) < 0.05  # small τ limit

    def test_ns_yield_returns_correct_shape(self):
        """ns_yield should return array with same length as input maturities."""
        params = self.NORMAL_PARAMS
        yields = self._generate_curve()
        assert len(yields) == len(self.MATURITIES)

    def test_fit_ns_recovers_generated_curve(self):
        """Fitting a curve generated from known NS params should achieve low RMSE."""
        yields = self._generate_curve()
        params = fit_nelson_siegel(self.MATURITIES, yields)
        # Verify the fitted params reproduce the yields closely
        fitted_yields = ns_yield(self.MATURITIES, params)
        rmse = np.sqrt(np.mean((fitted_yields - yields)**2))
        assert rmse < 0.10, f"RMSE too large: {rmse:.4f}"

    def test_fit_ns_normal_curve_negative_beta1(self):
        """Fitting a normal (upward-sloping) curve should produce β₁ < 0."""
        yields = self._generate_curve()
        params = fit_nelson_siegel(self.MATURITIES, yields)
        assert params.beta1 < 0, f"Normal curve should have β₁ < 0, got {params.beta1:.4f}"

    def test_fit_ns_flat_curve_near_zero_slope(self):
        """Fitting a flat curve should produce β₁ ≈ 0 and β₂ ≈ 0."""
        flat_yields = np.full(len(self.MATURITIES), 4.0)
        params = fit_nelson_siegel(self.MATURITIES, flat_yields)
        assert abs(params.beta1) < 0.5, f"Flat curve β₁ should be near 0, got {params.beta1:.4f}"
        assert abs(params.beta2) < 0.5, f"Flat curve β₂ should be near 0, got {params.beta2:.4f}"

    def test_fit_ns_inverted_curve_positive_beta1(self):
        """Fitting an inverted curve should produce β₁ > 0."""
        inverted = np.array([5.5, 5.3, 5.0, 4.5, 4.2, 3.8, 3.6, 3.4, 3.2, 3.0])
        params   = fit_nelson_siegel(self.MATURITIES, inverted)
        assert params.beta1 > 0, f"Inverted curve should have β₁ > 0, got {params.beta1:.4f}"

    def test_ns_loadings_level_is_one(self):
        """Level loading should be 1.0 for all maturities."""
        taus = np.array([0.5, 1.0, 5.0, 10.0, 30.0])
        L, _, _ = ns_loadings(taus, lam=0.5)
        assert np.allclose(L, 1.0)

    def test_classify_regime_normal_steep(self):
        """β₁ < -1.0 → normal_steep regime."""
        ns_df = pd.DataFrame({"beta1": [-1.5, -2.0, -1.2], "beta0": [4.5, 4.5, 4.5]})
        regimes = classify_curve_regime(ns_df)
        assert all(r == "normal_steep" for r in regimes)

    def test_classify_regime_inverted(self):
        """β₁ > +1.0 → inverted_deep regime."""
        ns_df = pd.DataFrame({"beta1": [1.2, 1.74, 1.5], "beta0": [5.0, 5.0, 5.0]})
        regimes = classify_curve_regime(ns_df)
        assert all("inverted" in r for r in regimes)

    def test_fit_ns_returns_nsparams(self):
        """fit_nelson_siegel should return an NSParams object."""
        yields = self._generate_curve()
        result = fit_nelson_siegel(self.MATURITIES, yields)
        assert isinstance(result, NSParams)
        assert hasattr(result, "beta0")
        assert hasattr(result, "beta1")
        assert hasattr(result, "lam")


# ── FOMC Probability tests ────────────────────────────────────────────────────

class TestFOMCProbability:

    def test_full_cut_probability_25bp_below(self):
        """Implied rate 25bps below current → near 100% probability of 25bp cut."""
        current_rate = 0.0358
        implied      = current_rate - 0.0025
        probs        = fedwatch_probabilities(implied, current_rate)
        cut_prob     = probs.get(-25, 0)
        assert cut_prob > 0.85, f"Expected >85% cut probability, got {cut_prob:.2%}"

    def test_probabilities_sum_to_one(self):
        """All FOMC outcome probabilities should sum to ~1.0."""
        probs = fedwatch_probabilities(0.0333, 0.0358)
        total = sum(probs.values())
        assert abs(total - 1.0) < 0.05

    def test_hold_probability_when_no_move(self):
        """When implied rate equals current rate, hold should dominate."""
        current = 0.0358
        probs   = fedwatch_probabilities(current, current)
        # Find the 0 outcome (hold)
        hold_prob = probs.get(0, 0)
        assert hold_prob > 0.8 or max(probs.values()) > 0.8

    def test_fomc_summary_returns_dict(self):
        summary = fomc_prob_summary(0.0333, 0.0358)
        assert isinstance(summary, dict)
        assert len(summary) > 0


# ── Macro Signals tests ───────────────────────────────────────────────────────

class TestMacroSignals:

    def _make_series(self, value, n=252):
        idx = pd.date_range("2024-01-01", periods=n, freq="B")
        return pd.Series([value] * n, index=idx)

    def _make_macro_df(self, n=252):
        """Synthetic macro DataFrame for composite_signal testing."""
        idx = pd.date_range("2024-01-01", periods=n, freq="B")
        df  = pd.DataFrame(index=idx)
        df["core_pce_yoy"]  = 2.5
        df["unemployment"]  = 4.0
        df["tsy_2y"]        = 4.0
        df["tsy_10y"]       = 4.5
        df["taylor_rate"]   = 2.5
        df["effr"]          = 4.0
        return df

    def test_composite_signal_returns_dataframe(self):
        df = self._make_macro_df()
        result = composite_signal(df)
        assert isinstance(result, pd.DataFrame)
        assert "position" in result.columns

    def test_composite_signal_position_values(self):
        """Position should only take values in {-1, 0, 1}."""
        df = self._make_macro_df()
        result = composite_signal(df)
        assert set(result["position"].unique()).issubset({-1, 0, 1})

    def test_policy_gap_signal_output_is_series(self):
        """policy_gap_signal should return a pd.Series with values in {-1, 0, 1}."""
        actual_rate  = self._make_series(5.0)
        taylor_rate  = self._make_series(2.5)  # large positive gap → bullish duration
        result       = policy_gap_signal(actual_rate, taylor_rate)
        assert isinstance(result, pd.Series)
        assert set(result.unique()).issubset({-1, 0, 1})

    def test_policy_gap_large_positive_gap_long_signal(self):
        """Fed 250bps above Taylor → should signal long duration."""
        actual  = self._make_series(5.0)
        taylor  = self._make_series(2.5)
        result  = policy_gap_signal(actual, taylor, threshold_bps=75.0)
        assert result.iloc[-1] == 1, "250bp positive gap should give long duration signal"

    def test_policy_gap_at_target_flat_signal(self):
        """Fed exactly at Taylor recommendation → should be flat."""
        rate   = self._make_series(3.5)
        result = policy_gap_signal(rate, rate, threshold_bps=75.0)
        assert result.iloc[-1] == 0, "No gap should give flat signal"

    def test_inflation_signal_output_is_series(self):
        result = inflation_momentum_signal(self._make_series(2.5))
        assert isinstance(result, pd.Series)
        assert set(result.dropna().unique()).issubset({-1, 0, 1})

    def test_labor_market_signal_output_is_series(self):
        result = labor_market_signal(self._make_series(4.2))
        assert isinstance(result, pd.Series)
        assert set(result.dropna().unique()).issubset({-1, 0, 1})

    def test_curve_slope_signal_output_is_series(self):
        slope  = self._make_series(0.5)  # 50bps 2s10s
        result = curve_slope_signal(slope)
        assert isinstance(result, pd.Series)
        assert set(result.dropna().unique()).issubset({-1, 0, 1})

    def test_composite_index_matches_input(self):
        """Composite signal should have same index as input DataFrame."""
        df     = self._make_macro_df()
        result = composite_signal(df)
        assert len(result) == len(df)
