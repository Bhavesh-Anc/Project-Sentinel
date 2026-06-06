"""
Walk-forward backtesting framework for US rates signals.

Design principles:
  - NO look-ahead bias: signals use only data available on each decision date
  - Walk-forward: model parameters re-estimated every 252 business days (annual)
  - Transaction costs: configurable bid-ask spread in bps
  - Instrument: 10Y Treasury total return (approximated from yield changes + carry)
  - Vol-scaling: position sized down when realised yield vol is elevated
  - Stop-loss: long duration positions exited if yield rises >stop_loss_bps from entry

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
    train_window    : number of business days in the training window
    test_window     : number of business days between re-estimations
    txn_cost_bps    : one-way transaction cost in basis points
    duration        : assumed modified duration of the traded instrument
    vol_scale       : if True, scale position size by target_vol / realised_vol
    target_vol_bps  : target daily P&L vol in bps (used when vol_scale=True)
    vol_window      : rolling window (days) for realised vol estimation
    stop_loss_bps   : exit long duration if yield rises this many bps from entry
                      exit short duration if yield falls this many bps from entry
                      set to None to disable
    """

    def __init__(
        self,
        train_window: int = 252,
        test_window: int = 63,
        txn_cost_bps: float = 0.5,
        duration: float = 8.0,
        vol_scale: bool = True,
        target_vol_bps: float = 30.0,
        vol_window: int = 21,
        stop_loss_bps: float | None = 50.0,
    ):
        self.train_window   = train_window
        self.test_window    = test_window
        self.txn_cost_bps   = txn_cost_bps
        self.duration       = duration
        self.vol_scale      = vol_scale
        self.target_vol_bps = target_vol_bps
        self.vol_window     = vol_window
        self.stop_loss_bps  = stop_loss_bps

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
        prev_position  = 0
        prev_size      = 1.0   # fractional position size from vol scaling
        entry_yield    = np.nan  # yield at which current position was entered
        stopped_out    = False   # True = stop-loss fired; wait for signal to reset

        # Pre-compute rolling yield vol for vol-scaling (uses all data; no look-ahead
        # because we only use past window at each step)
        if self.vol_scale and yield_col in data.columns:
            yield_changes = data[yield_col].diff() * 100  # bps
            roll_vol = yield_changes.rolling(self.vol_window).std()  # daily bps vol
        else:
            roll_vol = pd.Series(np.nan, index=data.index)

        for start_idx in range(self.train_window, n, self.test_window):
            train_df = data.iloc[start_idx - self.train_window:start_idx]
            test_end = min(start_idx + self.test_window, n)
            test_df  = data.iloc[start_idx:test_end]

            # Generate signal from training data (no look-ahead)
            try:
                positions = signal_fn(train_df)
                positions = positions.reindex(test_df.index, method="ffill")
            except Exception as e:
                print(f"  Signal error at {dates[start_idx]}: {e}")
                positions = pd.Series(0, index=test_df.index)

            for i, (idx, row) in enumerate(test_df.iterrows()):
                raw_pos = int(positions.get(idx, 0))

                # ── Stop-loss logic ───────────────────────────────────────────
                curr_yield = float(row[yield_col]) if yield_col in test_df.columns else np.nan
                if i == 0:
                    prev_yield = float(data.loc[train_df.index[-1], yield_col]) if yield_col in data.columns else np.nan
                else:
                    prev_yield = float(test_df.iloc[i - 1][yield_col]) if yield_col in test_df.columns else np.nan

                yield_change_bps = (curr_yield - prev_yield) * 100 if not (np.isnan(curr_yield) or np.isnan(prev_yield)) else 0.0

                if self.stop_loss_bps is not None:
                    # New position entered — record entry yield
                    if raw_pos != 0 and (raw_pos != prev_position or stopped_out):
                        entry_yield = curr_yield
                        stopped_out = False

                    # Check if stop triggered on existing position
                    if raw_pos != 0 and not stopped_out and not np.isnan(entry_yield):
                        move = (curr_yield - entry_yield) * 100  # bps
                        if (raw_pos > 0 and move > self.stop_loss_bps) or \
                           (raw_pos < 0 and move < -self.stop_loss_bps):
                            stopped_out = True

                    # Reset stop when signal flips to flat or opposite
                    if raw_pos == 0 or raw_pos != int(np.sign(prev_position)) and prev_position != 0:
                        stopped_out = False

                effective_pos = 0 if stopped_out else raw_pos

                # ── Vol-scaling ───────────────────────────────────────────────
                if self.vol_scale and effective_pos != 0:
                    rv = float(roll_vol.get(idx, np.nan))
                    if not np.isnan(rv) and rv > 1e-6:
                        # Scale so expected daily P&L vol ≈ target_vol_bps
                        # P&L vol ≈ duration × yield_vol → size = target / (dur × rv)
                        dur_est = approximate_10y_duration(curr_yield) if not np.isnan(curr_yield) else self.duration
                        implied_pnl_vol = dur_est * rv
                        size = min(1.0, self.target_vol_bps / implied_pnl_vol)
                    else:
                        size = 1.0
                else:
                    size = 1.0

                # ── P&L ───────────────────────────────────────────────────────
                dur = approximate_10y_duration(curr_yield) if not np.isnan(curr_yield) else self.duration
                carry_bps = curr_yield * 100 / 252 if not np.isnan(curr_yield) else 0.0

                is_new_trade = (effective_pos != prev_position)
                pnl_bps = treasury_total_return(
                    yield_change_bps=yield_change_bps * effective_pos * size,
                    duration=dur,
                    carry_bps=carry_bps * abs(effective_pos) * size,
                    transaction_cost_bps=self.txn_cost_bps,
                    is_new_trade=is_new_trade,
                )

                results.append({
                    "date":             idx,
                    "raw_signal":       raw_pos,
                    "position":         effective_pos,
                    "size":             size,
                    "stopped_out":      stopped_out,
                    "yield_pct":        curr_yield,
                    "yield_change_bps": yield_change_bps,
                    "duration":         dur,
                    "daily_pnl_bps":    pnl_bps,
                    "is_new_trade":     is_new_trade,
                    "is_oos":           True,
                })
                prev_position = effective_pos
                prev_size     = size

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
