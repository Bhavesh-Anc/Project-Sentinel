"""
Unit tests for backtesting framework and performance metrics.
"""
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import pytest
import numpy as np
import pandas as pd

from backtesting.performance import (
    sharpe_ratio, max_drawdown, max_drawdown_duration, calmar_ratio,
    annualized_return, annualized_volatility, hit_rate, win_loss_ratio,
    profit_factor, full_metrics, turnover,
)
from backtesting.signal_backtest import (
    WalkForwardBacktest, treasury_total_return, approximate_10y_duration,
)


# ── Performance Metric tests ──────────────────────────────────────────────────

class TestPerformanceMetrics:

    def _pnl(self, values):
        return pd.Series(values, dtype=float)

    def test_sharpe_constant_returns_nan(self):
        """Constant returns → std = 0 → Sharpe undefined (NaN)."""
        pnl = self._pnl([1.0] * 252)
        assert np.isnan(sharpe_ratio(pnl))

    def test_sharpe_zero_mean_near_zero(self):
        """Zero-mean returns should give Sharpe near 0."""
        np.random.seed(42)
        pnl = self._pnl(np.random.randn(252))
        sr  = sharpe_ratio(pnl)
        assert abs(sr) < 1.0

    def test_sharpe_positive_for_positive_drift(self):
        """Returns with clear positive drift → positive Sharpe."""
        np.random.seed(0)
        pnl = self._pnl(np.random.randn(252) + 1.0)  # mean ≈ 1
        assert sharpe_ratio(pnl) > 0

    def test_max_drawdown_monotone_up(self):
        """Monotonically increasing PnL should have zero max drawdown."""
        cpnl = self._pnl(range(1, 253))
        assert max_drawdown(cpnl) == 0.0

    def test_max_drawdown_always_nonpositive(self):
        """Max drawdown is always ≤ 0 by definition."""
        np.random.seed(0)
        cpnl = self._pnl(np.random.randn(252).cumsum())
        assert max_drawdown(cpnl) <= 0

    def test_max_drawdown_known_value(self):
        """Explicit series: peak at 10, trough at 3 → drawdown = -7."""
        cpnl = self._pnl([0, 5, 10, 8, 3, 6, 9])
        assert abs(max_drawdown(cpnl) - (-7.0)) < 1e-10

    def test_max_drawdown_duration_no_drawdown(self):
        """Monotone increasing series has zero drawdown duration."""
        cpnl = self._pnl([1, 2, 3, 4, 5])
        assert max_drawdown_duration(cpnl) == 0

    def test_max_drawdown_duration_known(self):
        """[0,10,8,6,11]: peak=10 at idx 1; underwater at idx 2,3 (2 days); recovery at 4."""
        cpnl     = self._pnl([0, 10, 8, 6, 11])
        duration = max_drawdown_duration(cpnl)
        assert duration == 2

    def test_hit_rate_all_positive(self):
        """All positive PnL → 100% hit rate."""
        pnl = self._pnl([1.0] * 10)
        assert hit_rate(pnl, active_only=False) == 1.0

    def test_hit_rate_all_negative(self):
        """All negative PnL → 0% hit rate."""
        pnl = self._pnl([-1.0] * 10)
        assert hit_rate(pnl, active_only=False) == 0.0

    def test_hit_rate_half(self):
        """Alternating +/-1 → 50% hit rate."""
        pnl = self._pnl([1.0, -1.0] * 50)
        assert abs(hit_rate(pnl, active_only=False) - 0.5) < 1e-10

    def test_hit_rate_active_only(self):
        """Hit rate with active_only should filter flat (position=0) days."""
        pnl = self._pnl([1.0, 1.0, -1.0, 0.0, 0.0])
        pos = pd.Series([1, 1, -1, 0, 0])
        hr  = hit_rate(pnl, active_only=True, position=pos)
        # Active days: 3 → 2 wins / 3 = 0.667
        assert abs(hr - 2/3) < 1e-10

    def test_win_loss_ratio_equal(self):
        """Equal average wins and losses → WL ratio = 1.0."""
        pnl = self._pnl([2.0, -2.0, 2.0, -2.0])
        assert abs(win_loss_ratio(pnl) - 1.0) < 1e-10

    def test_profit_factor_positive_series(self):
        """Gross profit 4, gross loss 2 → profit factor = 2."""
        pnl = self._pnl([3.0, 1.0, -1.0, -1.0])
        assert abs(profit_factor(pnl) - 2.0) < 1e-10

    def test_annualized_return_scale(self):
        """Annualized return = mean × 252."""
        pnl = self._pnl([2.0] * 252)
        assert abs(annualized_return(pnl) - 504.0) < 1e-10

    def test_annualized_vol_positive(self):
        """Annualized vol should be positive for any non-constant series."""
        np.random.seed(0)
        pnl = self._pnl(np.random.randn(252))
        assert annualized_volatility(pnl) > 0

    def test_annualized_vol_scale(self):
        """Annualized vol ≈ std × sqrt(252)."""
        np.random.seed(0)
        raw = np.random.randn(1000)  # large sample → ddof correction negligible
        pnl = self._pnl(raw)
        expected = raw.std() * np.sqrt(252)
        assert abs(annualized_volatility(pnl) / expected - 1.0) < 0.01

    def test_turnover_no_changes(self):
        """Constant position → 0 trades/yr."""
        pos = pd.Series([1] * 252)
        assert turnover(pos) == 0.0

    def test_full_metrics_keys(self):
        """full_metrics should return all expected keys."""
        np.random.seed(42)
        n    = 252
        pnl  = pd.Series(np.random.randn(n))
        cpnl = pnl.cumsum()
        pos  = pd.Series(np.sign(np.random.randn(n)).astype(int))
        df   = pd.DataFrame({"daily_pnl_bps": pnl, "cumulative_pnl_bps": cpnl, "position": pos})
        metrics = full_metrics(df)
        for key in ["sharpe_ratio", "max_drawdown_bps", "hit_rate",
                    "win_loss_ratio", "annualized_return_bps", "total_pnl_bps"]:
            assert key in metrics, f"Missing key: {key}"

    def test_total_pnl_is_period_gain(self):
        """total_pnl_bps = cumulative_pnl[-1] - cumulative_pnl[0], not the absolute level."""
        pnl  = pd.Series([1.0] * 10)
        cpnl = pnl.cumsum() + 100.0   # starts at 101, ends at 110 → gain = 9
        df   = pd.DataFrame({"daily_pnl_bps": pnl, "cumulative_pnl_bps": cpnl,
                              "position": pd.Series([1] * 10)})
        metrics = full_metrics(df)
        # cpnl.iloc[-1] - cpnl.iloc[0] = 110 - 101 = 9
        assert abs(metrics["total_pnl_bps"] - 9.0) < 1e-10


# ── Treasury Return approximation tests ──────────────────────────────────────

class TestTreasuryReturn:

    def test_no_yield_change_earns_carry(self):
        """Zero yield change and no new trade → return is just carry."""
        ret = treasury_total_return(
            yield_change_bps=0.0, duration=8.0,
            carry_bps=1.5, transaction_cost_bps=0.5, is_new_trade=False,
        )
        assert abs(ret - 1.5) < 1e-10

    def test_new_trade_deducts_txn_cost(self):
        ret_new      = treasury_total_return(0.0, 8.0, 0.0, 0.5, is_new_trade=True)
        ret_existing = treasury_total_return(0.0, 8.0, 0.0, 0.5, is_new_trade=False)
        assert ret_new < ret_existing
        assert abs(ret_existing - ret_new - 0.5) < 1e-10

    def test_yield_rise_hurts_long_position(self):
        """Rising yields → negative price return for duration × Δy."""
        ret = treasury_total_return(
            yield_change_bps=10.0, duration=8.0, carry_bps=0.0, is_new_trade=False,
        )
        assert ret < 0
        assert abs(ret - (-8.0 * 10.0)) < 1e-10

    def test_yield_fall_helps_long_position(self):
        ret = treasury_total_return(-10.0, 8.0, 0.0, is_new_trade=False)
        assert ret > 0


class TestApproxDuration:

    def test_duration_at_5pct_near_8years(self):
        """10Y par bond at 5% should have modified duration ~7.7–8.1 years."""
        dur = approximate_10y_duration(5.0)
        assert 7.0 < dur < 8.5, f"Duration at 5% should be ~7.8Y, got {dur:.2f}"

    def test_duration_higher_at_lower_yield(self):
        """Lower yield → higher modified duration."""
        assert approximate_10y_duration(2.0) > approximate_10y_duration(6.0)

    def test_duration_positive(self):
        for y in [1.0, 3.0, 5.0, 7.0, 10.0]:
            assert approximate_10y_duration(y) > 0


# ── Walk-Forward Backtest tests ───────────────────────────────────────────────

class TestWalkForwardBacktest:

    def _make_data(self, n=800):
        """Synthetic daily data with all required columns."""
        idx = pd.date_range("2018-01-01", periods=n, freq="B")
        df  = pd.DataFrame(index=idx)
        df["tsy_10y"]       = 4.0 + 0.5 * np.sin(np.linspace(0, 6*np.pi, n))
        df["tsy_2y"]        = 3.5 + 0.3 * np.sin(np.linspace(0, 6*np.pi, n))
        df["core_pce_yoy"]  = 2.5
        df["unemployment"]  = 4.2
        df["taylor_rate"]   = 2.5
        df["effr"]          = 4.0
        return df

    def _long_signal(self, train_df):  return pd.Series(1,  index=train_df.index)
    def _flat_signal(self, train_df):  return pd.Series(0,  index=train_df.index)
    def _short_signal(self, train_df): return pd.Series(-1, index=train_df.index)

    def test_result_has_required_columns(self):
        bt     = WalkForwardBacktest(train_window=252, test_window=63, stop_loss_bps=None)
        result = bt.run(self._make_data(), self._long_signal, yield_col="tsy_10y")
        for col in ["daily_pnl_bps", "cumulative_pnl_bps", "position", "is_oos"]:
            assert col in result.columns, f"Missing column: {col}"

    def test_all_results_are_oos(self):
        """Every row in the result should be out-of-sample."""
        bt     = WalkForwardBacktest(train_window=252, test_window=63)
        result = bt.run(self._make_data(), self._long_signal, yield_col="tsy_10y")
        assert result["is_oos"].all()

    def test_flat_signal_zero_positions(self):
        """A constant-zero signal should produce no positions."""
        bt     = WalkForwardBacktest(train_window=252, test_window=63, stop_loss_bps=None)
        result = bt.run(self._make_data(), self._flat_signal, yield_col="tsy_10y")
        assert (result["position"] == 0).all()

    def test_no_look_ahead_bias(self):
        """
        The signal function must only see training-window data.
        We record each roll's max training date and verify it is strictly
        before the corresponding test window start.
        """
        seen_max_dates = []

        def recorder(train_df):
            seen_max_dates.append(train_df.index.max())
            return pd.Series(0, index=train_df.index)

        data = self._make_data(n=600)
        bt   = WalkForwardBacktest(train_window=252, test_window=63, stop_loss_bps=None)
        bt.run(data, recorder, yield_col="tsy_10y")

        for i, max_train_date in enumerate(seen_max_dates):
            test_start_idx = 252 + i * 63
            if test_start_idx < len(data):
                test_start = data.index[test_start_idx]
                assert max_train_date < test_start, (
                    f"Look-ahead at roll {i}: signal saw up to {max_train_date} "
                    f"but test started at {test_start}"
                )

    def test_drawdown_always_nonpositive(self):
        """drawdown_bps column must always be ≤ 0."""
        bt     = WalkForwardBacktest(train_window=252, test_window=63, stop_loss_bps=None)
        result = bt.run(self._make_data(), self._long_signal, yield_col="tsy_10y")
        assert (result["drawdown_bps"] <= 1e-10).all()

    def test_stop_loss_fires(self):
        """A large yield spike should trigger the stop-loss mechanism."""
        data = self._make_data()
        spike_idx = 400
        data.iloc[spike_idx:spike_idx+5, data.columns.get_loc("tsy_10y")] += 1.5
        bt     = WalkForwardBacktest(train_window=252, test_window=63, stop_loss_bps=10.0)
        result = bt.run(data, self._long_signal, yield_col="tsy_10y")
        assert result["stopped_out"].any(), "Stop-loss should fire on large yield spike"

    def test_cumulative_pnl_is_cumsum_of_daily(self):
        """cumulative_pnl_bps must equal the cumulative sum of daily_pnl_bps."""
        bt     = WalkForwardBacktest(train_window=252, test_window=63, stop_loss_bps=None)
        result = bt.run(self._make_data(), self._flat_signal, yield_col="tsy_10y")
        expected = result["daily_pnl_bps"].cumsum()
        assert np.allclose(result["cumulative_pnl_bps"].values, expected.values, atol=1e-8)

    def test_long_higher_pnl_than_short_in_rally(self):
        """
        In a declining-yield environment (bond rally), long should outperform short.
        We construct a monotone falling yield series to create a clear price return.
        """
        n   = 600
        idx = pd.date_range("2018-01-01", periods=n, freq="B")
        df  = pd.DataFrame(index=idx)
        # Monotone falling yields → long duration always profits on price
        df["tsy_10y"]      = np.linspace(5.0, 3.0, n)
        df["tsy_2y"]       = np.linspace(4.5, 2.5, n)
        df["core_pce_yoy"] = 2.5
        df["unemployment"] = 4.2
        df["taylor_rate"]  = 2.5
        df["effr"]         = 4.0
        bt        = WalkForwardBacktest(train_window=252, test_window=63, stop_loss_bps=None, vol_scale=False)
        res_long  = bt.run(df, self._long_signal,  yield_col="tsy_10y")
        res_short = bt.run(df, self._short_signal, yield_col="tsy_10y")
        total_long  = res_long["daily_pnl_bps"].sum()
        total_short = res_short["daily_pnl_bps"].sum()
        assert total_long > total_short, (
            f"In a bond rally, long ({total_long:.1f}bps) should beat short ({total_short:.1f}bps)"
        )
