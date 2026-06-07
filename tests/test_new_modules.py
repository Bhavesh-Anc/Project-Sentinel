"""
Tests for risk analytics, curve strategies, term premium, and regime analysis.
"""
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import pytest
import numpy as np
import pandas as pd
from datetime import date
from dateutil.relativedelta import relativedelta

from sofr_engine import SOFRCurveBootstrapper, SOFRSwap, ScenarioEngine, RiskReport
from models.curve_strategies import (
    DV01NeutralSteepener, DV01NeutralFlattener, Butterfly,
    run_multi_strategy_backtest, compare_strategies,
)
from models.term_premium import fit_ar1, rolling_term_premium, term_premium_summary, AR1Params
from backtesting.regime_analysis import (
    label_macro_regimes, label_curve_regimes,
    performance_by_regime, regime_transition_matrix,
    regime_duration_stats,
)


# ── Helpers ───────────────────────────────────────────────────────────────────

def _flat_curve(rate: float = 0.0450):
    ref = date(2024, 6, 5)
    futures = pd.DataFrame([
        {'expiry': ref + relativedelta(months=3*(i+1)),
         'accrual_end': ref + relativedelta(months=3*(i+2)),
         'implied_rate': rate}
        for i in range(8)
    ])
    ois = [(t, rate) for t in [2, 3, 5, 7, 10, 15, 20, 30]]
    return SOFRCurveBootstrapper.from_market_data(ref, rate, futures, ois)


def _two_swap_portfolio(curve):
    swap_5y  = SOFRSwap(date(2024, 6, 7), date(2029, 6, 7), curve.par_ois_rate(5.0),  10_000_000)
    swap_10y = SOFRSwap(date(2024, 6, 7), date(2034, 6, 7), curve.par_ois_rate(10.0), 10_000_000)
    return RiskReport({"5Y": swap_5y, "10Y": swap_10y}, curve)


def _market_data(n: int = 400) -> pd.DataFrame:
    idx = pd.date_range("2019-01-01", periods=n, freq="B")
    return pd.DataFrame({
        "tsy_2y":  4.5 - np.linspace(0, 2, n) + np.random.RandomState(42).randn(n) * 0.02,
        "tsy_5y":  4.3 - np.linspace(0, 1.5, n) + np.random.RandomState(1).randn(n) * 0.02,
        "tsy_10y": 4.0 - np.linspace(0, 1, n) + np.random.RandomState(2).randn(n) * 0.02,
    }, index=idx)


def _macro_df(n: int = 500) -> pd.DataFrame:
    idx = pd.date_range("2019-01-01", periods=n, freq="B")
    return pd.DataFrame({
        "effr":         [5.0] * 200 + [3.5] * 150 + [2.5] * 150,
        "taylor_rate":  [2.5] * n,
        "core_pce_yoy": [3.5] * 200 + [2.2] * 150 + [2.0] * 150,
    }, index=idx)


# ── ScenarioEngine tests ──────────────────────────────────────────────────────

class TestScenarioEngine:

    def setup_method(self):
        self.curve = _flat_curve(0.045)

    def test_parallel_shift_up_reduces_df(self):
        """Upward shift → all discount factors decrease."""
        shifted = ScenarioEngine.parallel_shift(self.curve, +100)
        for t in [1, 5, 10, 30]:
            assert shifted.df(t) < self.curve.df(t), f"DF({t}Y) should fall on +100bp shift"

    def test_parallel_shift_zero_is_identity(self):
        """Zero shift should not change any discount factors."""
        shifted = ScenarioEngine.parallel_shift(self.curve, 0.0)
        for t in [1, 5, 10]:
            assert abs(shifted.df(t) - self.curve.df(t)) < 1e-10

    def test_parallel_shift_symmetry(self):
        """Up then down shift should recover original DF."""
        c1 = ScenarioEngine.parallel_shift(self.curve, +50)
        c2 = ScenarioEngine.parallel_shift(c1, -50)
        for t in [2, 5, 10]:
            assert abs(c2.df(t) - self.curve.df(t)) < 1e-8

    def test_key_rate_shift_local_only(self):
        """Key-rate shift at 5Y should leave 30Y discount factor nearly unchanged."""
        shifted = ScenarioEngine.key_rate_shift(self.curve, key_tenor=5.0, shift_bps=100, width=1.5)
        assert abs(shifted.df(30) - self.curve.df(30)) < 1e-4

    def test_key_rate_shift_at_centre_equals_full(self):
        """At the key tenor exactly, shift should be at full magnitude."""
        shifted = ScenarioEngine.key_rate_shift(self.curve, key_tenor=5.0, shift_bps=100, width=2.0)
        dz_base  = self.curve.zero_rate(5.0)
        dz_shift = shifted.zero_rate(5.0)
        assert abs((dz_shift - dz_base) * 10000 - 100) < 5  # within 5bps tolerance

    def test_twist_short_end_dominant(self):
        """Twist with large short shift should impact 2Y more than 10Y."""
        shifted = ScenarioEngine.twist_shift(self.curve, short_shift_bps=100, long_shift_bps=0)
        delta_2  = shifted.zero_rate(2) - self.curve.zero_rate(2)
        delta_10 = shifted.zero_rate(10) - self.curve.zero_rate(10)
        assert abs(delta_2) > abs(delta_10)

    def test_butterfly_belly_shifted(self):
        """Butterfly shift should change belly rate by belly_shift_bps (approx)."""
        shifted = ScenarioEngine.butterfly_shift(self.curve, wing_shift_bps=0, belly_shift_bps=50)
        delta_5 = (shifted.zero_rate(5) - self.curve.zero_rate(5)) * 10000
        assert abs(delta_5 - 50) < 10  # within 10bps


# ── RiskReport tests ──────────────────────────────────────────────────────────

class TestRiskReport:

    def setup_method(self):
        self.curve = _flat_curve(0.045)
        self.rpt   = _two_swap_portfolio(self.curve)

    def test_dv01_ladder_shape(self):
        ladder = self.rpt.dv01_ladder()
        assert "tenor_yrs" in ladder.columns
        assert "total_dv01_usd" in ladder.columns
        assert len(ladder) == len([0.25, 0.5, 1.0, 2.0, 3.0, 5.0, 7.0, 10.0, 15.0, 20.0, 30.0])

    def test_dv01_long_end_larger(self):
        """10Y swap should have larger DV01 in long-end buckets than short-end."""
        ladder   = self.rpt.dv01_ladder()
        short_dv = ladder[ladder["tenor_yrs"] <= 2]["10Y"].abs().sum()
        long_dv  = ladder[ladder["tenor_yrs"] >= 7]["10Y"].abs().sum()
        assert long_dv > short_dv

    def test_scenario_grid_shape(self):
        """Scenario grid should be n_short × n_long."""
        grid = self.rpt.scenario_grid([-25, 0, 25], [-25, 0, 25])
        assert grid.shape == (3, 3)

    def test_scenario_grid_zero_shift_zero_pnl(self):
        """Zero shift in all scenarios should give zero P&L."""
        grid = self.rpt.scenario_grid([0], [0])
        assert abs(grid.loc[0, 0]) < 1.0  # within $1

    def test_parallel_pnl_monotone(self):
        """Payer swaps: rising rates → positive P&L (floating leg value increases)."""
        pnl = self.rpt.parallel_pnl([-50, 0, 50])
        pnl_up   = pnl[pnl["shift_bps"] == 50]["pnl_usd"].values[0]
        pnl_down = pnl[pnl["shift_bps"] == -50]["pnl_usd"].values[0]
        assert pnl_up > pnl_down  # payer benefits from rising rates

    def test_stress_test_returns_dataframe(self):
        stress = self.rpt.stress_test()
        assert isinstance(stress, pd.DataFrame)
        assert len(stress) == 7
        assert "pnl_usd" in stress.columns

    def test_historical_var_order(self):
        """VaR at 99% should be more negative than at 95%."""
        pnl    = pd.Series(np.random.RandomState(0).randn(500) * 10)
        var_99 = RiskReport.historical_var(pnl, 0.99)["var_1d_99pct"]
        var_95 = RiskReport.historical_var(pnl, 0.95)[f"var_1d_95pct"]
        assert var_99 <= var_95  # 99% VaR is at least as bad as 95%

    def test_var_1d_99pct_key_present(self):
        pnl  = pd.Series(np.random.RandomState(0).randn(500))
        vr   = RiskReport.historical_var(pnl, 0.99)
        assert "var_1d_99pct" in vr


# ── DV01NeutralSteepener tests ───────────────────────────────────────────────

class TestSteepener:

    def setup_method(self):
        self.data = _market_data()

    def test_backtest_returns_dataframe(self):
        st  = DV01NeutralSteepener()
        res = st.run_backtest(self.data)
        assert isinstance(res, pd.DataFrame)
        assert "daily_pnl_bps" in res.columns
        assert "slope_2s10s_bps" in res.columns

    def test_flat_position_zero_pnl(self):
        """When position is always 0, P&L should be zero."""
        st  = DV01NeutralSteepener()
        pos = pd.Series(0, index=self.data.index)
        res = st.run_backtest(self.data, pos)
        assert res["daily_pnl_bps"].abs().sum() < 1.0  # only txn cost on 0→0 trades

    def test_steepener_profits_from_steepening(self):
        """A monotone-steepening curve should give positive P&L for long steepener."""
        n   = 300
        idx = pd.date_range("2020-01-01", periods=n, freq="B")
        df  = pd.DataFrame({
            "tsy_2y":  np.linspace(4.0, 2.0, n),   # 2Y falls fast
            "tsy_10y": np.linspace(3.5, 3.5, n),    # 10Y stays flat
        }, index=idx)
        # Slope goes from -50bps to +150bps → steepens 200bps → long steepener profits
        st  = DV01NeutralSteepener()
        res = st.run_backtest(df, txn_cost_bps=0)
        assert res["daily_pnl_bps"].sum() > 50  # at least 50bps profit

    def test_flattener_opposite_sign(self):
        """Flattener should produce opposite-sign P&L to steepener."""
        st   = DV01NeutralSteepener()
        fl   = DV01NeutralFlattener()
        res_st = st.run_backtest(self.data, txn_cost_bps=0)
        res_fl = fl.run_backtest(self.data, txn_cost_bps=0)
        np.testing.assert_allclose(
            res_st["daily_pnl_bps"].values,
            -res_fl["daily_pnl_bps"].values,
            atol=1e-8,
        )

    def test_cumulative_is_cumsum_daily(self):
        st  = DV01NeutralSteepener()
        res = st.run_backtest(self.data)
        expected = res["daily_pnl_bps"].cumsum()
        np.testing.assert_allclose(res["cumulative_pnl_bps"].values, expected.values, atol=1e-8)


# ── Butterfly tests ───────────────────────────────────────────────────────────

class TestButterfly:

    def setup_method(self):
        self.data = _market_data()

    def test_backtest_returns_dataframe(self):
        bt  = Butterfly()
        res = bt.run_backtest(self.data)
        assert isinstance(res, pd.DataFrame)
        assert "butterfly_bps" in res.columns

    def test_flat_position_zero_pnl(self):
        pos = pd.Series(0, index=self.data.index)
        bt  = Butterfly()
        res = bt.run_backtest(self.data, pos)
        assert res["daily_pnl_bps"].abs().sum() < 1.0


# ── compare_strategies tests ─────────────────────────────────────────────────

class TestCompareStrategies:

    def test_output_has_sharpe_and_return(self):
        data = _market_data()
        st   = DV01NeutralSteepener()
        bt   = Butterfly()
        cmp  = compare_strategies({
            "steepener": st.run_backtest(data),
            "butterfly": bt.run_backtest(data),
        })
        assert "sharpe" in cmp.columns
        assert "ann_return_bps" in cmp.columns
        assert len(cmp) == 2

    def test_multi_strategy_runs(self):
        data = _market_data(n=500)
        res  = run_multi_strategy_backtest(data)
        assert "daily_pnl_bps" in res.columns
        assert len(res) > 100


# ── Term Premium tests ────────────────────────────────────────────────────────

class TestTermPremium:

    def _short_rate(self, n=500):
        idx = pd.date_range("2015-01-01", periods=n, freq="B")
        return pd.Series(np.random.RandomState(7).randn(n).cumsum() * 0.001 + 0.04, index=idx)

    def test_ar1_fit_returns_params(self):
        r      = self._short_rate()
        params = fit_ar1(r, min_obs=60)
        assert isinstance(params, AR1Params)
        assert 0.0 < params.rho < 1.0
        assert params.sigma > 0

    def test_ar1_rho_clipped(self):
        """rho should always be in [0.80, 0.9999]."""
        r      = self._short_rate()
        params = fit_ar1(r)
        assert 0.80 <= params.rho <= 0.9999

    def test_ar1_raises_on_insufficient_data(self):
        r = self._short_rate(n=20)
        with pytest.raises(ValueError):
            fit_ar1(r, min_obs=60)

    def test_rolling_tp_returns_dataframe(self):
        n   = 400
        idx = pd.date_range("2015-01-01", periods=n, freq="B")
        df  = pd.DataFrame({
            "fed_funds": np.random.RandomState(0).randn(n).cumsum() * 0.05 + 2.0,
            "tsy_10y":   np.random.RandomState(1).randn(n).cumsum() * 0.03 + 3.0,
        }, index=idx)
        tp = rolling_term_premium(df, estimation_window=120)
        assert isinstance(tp, pd.DataFrame)
        assert "term_premium" in tp.columns
        assert len(tp) > 0

    def test_term_premium_has_positive_values(self):
        """When 10Y yield consistently above short rate, term premium should be positive."""
        np.random.seed(99)
        n   = 500
        idx = pd.date_range("2015-01-01", periods=n, freq="B")
        df  = pd.DataFrame({
            "fed_funds": 2.0 + np.random.randn(n) * 0.05,   # short rate ~2%
            "tsy_10y":   3.5 + np.random.randn(n) * 0.03,   # 10Y ~3.5% → TP ≈ +1.5%
        }, index=idx)
        tp = rolling_term_premium(df, estimation_window=120)
        non_nan = tp["term_premium"].dropna()
        assert len(non_nan) > 0
        assert non_nan.mean() > 0

    def test_summary_keys(self):
        n   = 300
        idx = pd.date_range("2015-01-01", periods=n, freq="B")
        df  = pd.DataFrame({
            "fed_funds": np.random.RandomState(3).randn(n).cumsum() * 0.05 + 3.0,
            "tsy_10y":   np.random.RandomState(4).randn(n).cumsum() * 0.03 + 4.0,
        }, index=idx)
        tp   = rolling_term_premium(df, estimation_window=80)
        summ = term_premium_summary(tp)
        for key in ["mean_tp_pct", "current_tp_pct", "max_tp_pct", "min_tp_pct"]:
            assert key in summ

    def test_empty_df_summary(self):
        """term_premium_summary on empty DataFrame returns empty dict."""
        result = term_premium_summary(pd.DataFrame())
        assert result == {}


# ── Regime Analysis tests ─────────────────────────────────────────────────────

class TestRegimeAnalysis:

    def setup_method(self):
        self.macro = _macro_df(500)
        self.n     = 500
        self.idx   = pd.date_range("2019-01-01", periods=self.n, freq="B")

    def test_macro_regime_values(self):
        """label_macro_regimes should produce only valid regime names."""
        regime = label_macro_regimes(self.macro)
        valid  = {"restrictive_hot", "restrictive_cooling", "neutral", "accommodative",
                  "restrictive_neutral"}
        assert set(regime.unique()).issubset(valid)

    def test_restrictive_when_gap_large(self):
        """Fed 500bps above Taylor → restrictive_hot (inflation above target)."""
        df = pd.DataFrame({
            "effr":         [5.5],
            "taylor_rate":  [0.5],
            "core_pce_yoy": [3.0],
        }, index=pd.date_range("2023-01-01", periods=1))
        regime = label_macro_regimes(df, threshold_bps=75)
        assert regime.iloc[0] == "restrictive_hot"

    def test_accommodative_when_gap_negative(self):
        """Fed 500bps below Taylor → accommodative."""
        df = pd.DataFrame({
            "effr":         [0.25],
            "taylor_rate":  [5.5],
            "core_pce_yoy": [2.0],
        }, index=pd.date_range("2021-01-01", periods=1))
        regime = label_macro_regimes(df, threshold_bps=75)
        assert regime.iloc[0] == "accommodative"

    def test_curve_regime_inverted(self):
        """β₁ > 1 → inverted."""
        ns = pd.DataFrame({"beta1": [1.74]}, index=pd.date_range("2023-01-01", periods=1))
        regime = label_curve_regimes(ns)
        assert regime.iloc[0] == "inverted"

    def test_curve_regime_steep(self):
        """β₁ < -1 → steep."""
        ns = pd.DataFrame({"beta1": [-2.0]}, index=pd.date_range("2019-01-01", periods=1))
        regime = label_curve_regimes(ns)
        assert regime.iloc[0] == "steep"

    def test_performance_by_regime_shape(self):
        """Returns a DataFrame with one row per regime."""
        regime  = label_macro_regimes(self.macro)
        pnl_df  = pd.DataFrame({
            "daily_pnl_bps": np.random.randn(self.n),
            "position":      np.sign(np.random.randn(self.n)).astype(int),
        }, index=self.idx)
        result  = performance_by_regime(pnl_df, regime)
        assert isinstance(result, pd.DataFrame)
        assert "sharpe" in result.columns
        assert "hit_rate" in result.columns

    def test_transition_matrix_rows_sum_to_one(self):
        """Every row of the transition matrix should sum to 1.0."""
        regime = label_macro_regimes(self.macro)
        trans  = regime_transition_matrix(regime)
        for row_sum in trans.sum(axis=1):
            assert abs(row_sum - 1.0) < 1e-9

    def test_duration_stats_non_negative(self):
        """All duration statistics should be non-negative."""
        regime = label_macro_regimes(self.macro)
        stats  = regime_duration_stats(regime)
        assert (stats["mean_days"] > 0).all()
        assert (stats["max_days"] > 0).all()
        assert (stats["total_days"] > 0).all()

    def test_duration_total_sums_to_input_length(self):
        """Sum of total_days across all regimes should equal input series length."""
        regime = label_macro_regimes(self.macro)
        stats  = regime_duration_stats(regime)
        assert stats["total_days"].sum() == len(regime)
