"""
Walk-forward backtesting framework for US rates signals.

Design principles:
  - NO look-ahead bias: signals use only data available on each decision date
  - Walk-forward: model parameters re-estimated every 252 business days (annual)
  - Transaction costs: configurable bid-ask spread in bps
  - Instrument: 10Y Treasury total return (approximated from yield changes + carry)

Walk-Forward Scheme
-------------------
Training window: 252 business days (1 year)
Test window:     63 business days (1 quarter), rolled quarterly
Re-estimation:   Taylor Rule α,β and Nelson-Siegel λ re-fit each roll

This means signals from 2022 onward use only pre-2022 data in their model
parameters — a genuine out-of-sample test for the 2022–2025 hiking cycle.
"""
from __future__ import annotations
import numpy as np
import pandas as pd
from datetime import date
from typing import Callable


# ── PnL Attribution ───────────────────────────────────────────────────────────

def treasury_total_return(
    yield_change_bps: float,
    duration: float,
    carry_bps: float = 0.0,
    transaction_cost_bps: float = 0.5,
    is_new_trade: bool = False,
) -> float:
    """
    Approximate 1-period total return for a Treasury position.

    Return ≈ -duration × Δy + carry − txn_cost (if position changed)

    Parameters
    ----------
    yield_change_bps     : change in yield from t to t+1 (bps)
    duration             : modified duration of the bond (years)
    carry_bps            : yield earned per period (bps); annualized / 252 for daily
    transaction_cost_bps : one-way bid-ask cost (bps); applied only on trade
    is_new_trade         : True if position changed (incur transaction cost)

    Returns
    -------
    Return in bps
    """
    price_return = -duration * yield_change_bps
    carry        = carry_bps
    txn_cost     = -transaction_cost_bps if is_new_trade else 0.0
    return price_return + carry + txn_cost


def approximate_10y_duration(yield_pct: float, coupon_pct: float = None) -> float:
    """
    Approximate modified duration of a 10Y par Treasury bond.
    For near-par bonds: duration ≈ (1 - (1+y)^{-10}) / y  [Macaulay / (1+y/2)]
    """
    if coupon_pct is None:
        coupon_pct = yield_pct  # Par bond assumption
    y = yield_pct / 100.0 / 2  # semi-annual
    n = 20                      # 10Y × 2 semi-annual periods
    if y < 1e-8:
        mac_dur = n / 2
    else:
        mac_dur = (1 - (1 + y) ** (-n)) / y / 2  # years
    mod_dur = mac_dur / (1 + y)
    return mod_dur


# ── Walk-Forward Engine ───────────────────────────────────────────────────────

class WalkForwardBacktest:
    """
    Walk-forward backtest for rates signals.

    Parameters
    ----------
    train_window : number of business days in the training window
    test_window  : number of business days between re-estimations
    signal_fn    : callable(train_df) → position_series
                   takes a training DataFrame, returns position for test period
    txn_cost_bps : one-way transaction cost in basis points
    duration     : assumed modified duration of the traded instrument
    """

    def __init__(
        self,
        train_window: int = 252,
        test_window: int = 63,
        txn_cost_bps: float = 0.5,
        duration: float = 8.0,
    ):
        self.train_window = train_window
        self.test_window  = test_window
        self.txn_cost_bps = txn_cost_bps
        self.duration     = duration

    def run(
        self,
        data: pd.DataFrame,
        signal_fn: Callable[[pd.DataFrame], pd.Series],
        yield_col:    str = "tsy_10y",
        position_col: str = "position",
    ) -> pd.DataFrame:
        """
        Execute the walk-forward backtest.

        Parameters
        ----------
        data       : full DataFrame (all dates, all columns including yields)
        signal_fn  : callable that receives training-period DataFrame and
                     returns a pd.Series of positions {-1, 0, 1} indexed by date
        yield_col  : column name for the yield used to compute returns

        Returns
        -------
        DataFrame with columns:
            position, yield_change_bps, duration, daily_pnl_bps,
            cumulative_pnl_bps, drawdown_bps, is_oos
        """
        dates = data.index
        n     = len(dates)
        results = []
        prev_position = 0

        for start_idx in range(self.train_window, n, self.test_window):
            train_df = data.iloc[start_idx - self.train_window:start_idx]
            test_end = min(start_idx + self.test_window, n)
            test_df  = data.iloc[start_idx:test_end]

            # Generate signal from training data
            try:
                positions = signal_fn(train_df)
                # Extend signal to test period (use last signal value if shorter)
                positions = positions.reindex(test_df.index, method="ffill")
            except Exception as e:
                print(f"  Signal error at {dates[start_idx]}: {e}")
                positions = pd.Series(0, index=test_df.index)

            # Compute daily P&L for each test day
            for i, (idx, row) in enumerate(test_df.iterrows()):
                pos = int(positions.get(idx, 0))

                if i == 0:
                    # First day of test period: compute yield change from last train day
                    prev_yield = float(data.loc[train_df.index[-1], yield_col]) if yield_col in data.columns else np.nan
                else:
                    prev_yield = float(test_df.iloc[i - 1][yield_col]) if yield_col in test_df.columns else np.nan

                curr_yield = float(row[yield_col]) if yield_col in test_df.columns else np.nan
                yield_change_bps = (curr_yield - prev_yield) * 100 if not np.isnan(curr_yield) and not np.isnan(prev_yield) else 0.0

                # Duration: use yield-dependent estimate
                dur = approximate_10y_duration(curr_yield) if not np.isnan(curr_yield) else self.duration

                # Carry: approximate as daily yield / 252
                carry_bps = curr_yield * 100 / 252 if not np.isnan(curr_yield) else 0.0

                is_new_trade = (pos != prev_position)
                pnl_bps = treasury_total_return(
                    yield_change_bps=yield_change_bps * pos,
                    duration=dur,
                    carry_bps=carry_bps * abs(pos),
                    transaction_cost_bps=self.txn_cost_bps,
                    is_new_trade=is_new_trade,
                )

                results.append({
                    "date":             idx,
                    "position":         pos,
                    "yield_pct":        curr_yield,
                    "yield_change_bps": yield_change_bps,
                    "duration":         dur,
                    "daily_pnl_bps":    pnl_bps,
                    "is_new_trade":     is_new_trade,
                    "is_oos":           True,
                })
                prev_position = pos

        result_df = pd.DataFrame(results).set_index("date")
        result_df = self._add_cumulative_metrics(result_df)
        return result_df

    @staticmethod
    def _add_cumulative_metrics(df: pd.DataFrame) -> pd.DataFrame:
        df["cumulative_pnl_bps"] = df["daily_pnl_bps"].cumsum()
        rolling_max = df["cumulative_pnl_bps"].cummax()
        df["drawdown_bps"] = df["cumulative_pnl_bps"] - rolling_max
        return df


# ── Signal function factory ───────────────────────────────────────────────────

def make_signal_fn(
    taylor_config=None,
    signal_weights: dict | None = None,
) -> Callable:
    """
    Factory that creates a signal_fn compatible with WalkForwardBacktest.
    The returned function re-estimates the Taylor Rule on the training window,
    then generates signals for the test period.
    """
    from models.taylor_rule import compute_taylor_rule, TaylorRuleConfig
    from models.macro_signals import composite_signal

    if taylor_config is None:
        taylor_config = TaylorRuleConfig()

    def signal_fn(train_df: pd.DataFrame) -> pd.Series:
        # Re-estimate Taylor Rule on this training window
        if "core_pce_yoy" in train_df.columns and "unemployment" in train_df.columns:
            taylor_df = compute_taylor_rule(
                train_df["core_pce_yoy"],
                train_df["unemployment"],
                taylor_config,
            )
            train_df = train_df.copy()
            train_df["taylor_rate"] = taylor_df["taylor_rate"]

        sig_df = composite_signal(train_df, weights=signal_weights)
        return sig_df["position"]

    return signal_fn


if __name__ == "__main__":
    print("Walk-forward backtester loaded. Run via notebooks/main_analysis.ipynb")
    print("or import WalkForwardBacktest and make_signal_fn in your analysis script.")
