"""
Multi-leg curve trading strategies for US rates.

Strategies
----------
DV01NeutralSteepener  : Long 10Y / Short 2Y, DV01-neutral — profits from 2s10s steepening
DV01NeutralFlattener  : Short 10Y / Long 2Y, DV01-neutral — profits from flattening
Butterfly             : Long belly (5Y), short wings (2Y & 10Y), DV01-neutral
RiskReversalSteepener : Leveraged steepener with stop-loss

WalkForwardCurveBacktest : Extend WalkForwardBacktest to multi-leg strategies
"""
from __future__ import annotations
import numpy as np
import pandas as pd
from dataclasses import dataclass
from typing import Callable

from backtesting.performance import full_metrics, sharpe_ratio, max_drawdown, annualized_return


# ── Duration helpers ──────────────────────────────────────────────────────────

def _modified_duration(ytm_pct: float, tenor_years: float, coupon_freq: int = 2) -> float:
    """
    Approximate modified duration for a par bond.
    Exact Macaulay duration / (1 + y/m) where m = coupon payments/year.
    """
    y = ytm_pct / 100.0
    m = coupon_freq
    c = y / m          # coupon per period
    n = int(round(tenor_years * m))
    if n == 0 or c < 1e-12:
        return tenor_years
    # Macaulay duration via closed form for par bond
    mac_dur = (1 + 1/m) / y - 1 / (m * (np.power(1 + y/m, n) - 1) * (1 + y/m)**(-1) + y)
    return mac_dur / (1 + y / m)


def _dv01_per_notional(ytm_pct: float, tenor_years: float) -> float:
    """DV01 per $1 notional = modified_duration × price / 10_000 (at par price ≈ 1)."""
    return _modified_duration(ytm_pct, tenor_years) / 10_000.0


# ── 2s10s Steepener ───────────────────────────────────────────────────────────

class DV01NeutralSteepener:
    """
    DV01-neutral 2s10s steepener.

    Position
    --------
    Long  10Y Treasuries (receive fixed on 10Y OIS swap) — benefits when 10Y falls
    Short 2Y  Treasuries (pay fixed on 2Y OIS swap)      — benefits when 2Y rises

    Sizing
    ------
    To be DV01-neutral: notional_2y × DV01_2y = notional_10y × DV01_10y
    We fix notional_10y = 10M and size the 2Y leg accordingly.

    P&L per day (in bps of 10Y DV01 equivalent)
    -------------------------------------------
    pnl = -DV01_10y × Δy10 + DV01_2y × Δy2
        = DV01 × (Δy2 - Δy10)   [since DV01_10y = DV01_2y by construction]
        = DV01 × Δ(slope)         where slope = y2 - y10

    Carry: net carry = (y10 - y2) × fraction_of_year
           (receive 10Y coupon, pay 2Y coupon; positive when curve is normal)
    """

    def __init__(self, dv01_usd: float = 10_000.0, carry_bps_per_year: float | None = None):
        self.dv01_usd   = dv01_usd
        self._carry_bps = carry_bps_per_year

    def daily_pnl_bps(
        self,
        dy2_bps: float,
        dy10_bps: float,
        y2_pct: float = 4.0,
        y10_pct: float = 4.5,
    ) -> float:
        """
        Daily P&L in bps (normalised to DV01 = 1bp per unit).

        pnl = Δ(2s10s) = Δy2 - Δy10
        Carry = (y10 - y2) / 252   (daily accrual)
        """
        price_pnl = dy2_bps - dy10_bps         # change in slope
        carry     = (y10_pct - y2_pct) / 252.0  # daily carry in %
        return price_pnl + carry * 100           # convert carry to bps

    def run_backtest(
        self,
        data: pd.DataFrame,
        position_signal: pd.Series | None = None,
        txn_cost_bps: float = 0.5,
    ) -> pd.DataFrame:
        """
        Run backtest of the steepener strategy.

        Parameters
        ----------
        data : DataFrame with columns tsy_2y, tsy_10y (yields in %)
        position_signal : +1 = long steepener, -1 = short, 0 = flat
                          defaults to always long (+1)
        txn_cost_bps : round-trip transaction cost

        Returns
        -------
        DataFrame with daily_pnl_bps, cumulative_pnl_bps, position, slope_2s10s
        """
        df   = data.copy()
        cols = ["tsy_2y", "tsy_10y"]
        df.dropna(subset=cols, inplace=True)

        dy2  = df["tsy_2y"].diff() * 100    # in bps
        dy10 = df["tsy_10y"].diff() * 100
        slope = (df["tsy_10y"] - df["tsy_2y"]) * 100   # in bps

        if position_signal is None:
            pos = pd.Series(1, index=df.index)
        else:
            pos = position_signal.reindex(df.index).fillna(0)

        # Detect trades (position changes)
        trades = pos.diff().abs() > 0.5

        # Steepener = pay fixed 10Y + receive fixed 2Y (DV01-neutral)
        # Price P&L = Δy10 - Δy2 = change in slope  (positive when curve steepens)
        # Carry     = receive y2 coupon - pay y10 coupon (negative when curve is normal)
        pnl_raw = dy10 - dy2 + (df["tsy_2y"] - df["tsy_10y"]) / 252.0 * 100
        pnl     = pos.shift(1) * pnl_raw
        pnl[trades] -= txn_cost_bps

        result = pd.DataFrame({
            "daily_pnl_bps":      pnl,
            "cumulative_pnl_bps": pnl.cumsum(),
            "position":           pos,
            "slope_2s10s_bps":    slope,
            "dy2_bps":            dy2,
            "dy10_bps":           dy10,
        }).dropna()

        return result


# ── 2s10s Flattener ───────────────────────────────────────────────────────────

class DV01NeutralFlattener:
    """
    DV01-neutral flattener — mirror image of the steepener.
    Long 2Y / Short 10Y. Profits when curve flattens (Δy2 - Δy10 < 0).
    """

    def run_backtest(
        self,
        data: pd.DataFrame,
        txn_cost_bps: float = 0.5,
    ) -> pd.DataFrame:
        st   = DV01NeutralSteepener()
        res  = st.run_backtest(data, txn_cost_bps=txn_cost_bps)
        res["daily_pnl_bps"]      *= -1
        res["cumulative_pnl_bps"]  = res["daily_pnl_bps"].cumsum()
        res["position"]            = -res["position"]
        return res


# ── 2s5s10s Butterfly ─────────────────────────────────────────────────────────

class Butterfly:
    """
    2s5s10s DV01-neutral butterfly.

    Structure
    ---------
    Long belly (5Y): receive fixed 5Y
    Short wings: pay fixed 2Y and 10Y

    Wing sizing (DV01-neutral):
      DV01_5y = DV01_2y × w2 + DV01_10y × w10
      w2 + w10 = 1  (equal wing split)
      → w2 = w10 = 0.5 × DV01_5y / (0.5×DV01_2y + 0.5×DV01_10y)

    Curvature trade:
      pnl ≈ Δy5 - 0.5×Δy2 - 0.5×Δy10
      Carry: positive if curve has positive curvature (belly cheap to wings)
    """

    def run_backtest(
        self,
        data: pd.DataFrame,
        position_signal: pd.Series | None = None,
        txn_cost_bps: float = 0.75,
    ) -> pd.DataFrame:
        """
        Backtest the butterfly strategy.

        Parameters
        ----------
        data : DataFrame requiring tsy_2y, tsy_5y, tsy_10y columns
        """
        cols = ["tsy_2y", "tsy_5y", "tsy_10y"]
        df   = data.copy().dropna(subset=cols)

        dy2  = df["tsy_2y"].diff()  * 100
        dy5  = df["tsy_5y"].diff()  * 100
        dy10 = df["tsy_10y"].diff() * 100

        # Approximate DV01 weights via modified duration
        y2  = df["tsy_2y"].mean()
        y5  = df["tsy_5y"].mean()
        y10 = df["tsy_10y"].mean()
        d2  = _modified_duration(y2,  2.0)
        d5  = _modified_duration(y5,  5.0)
        d10 = _modified_duration(y10, 10.0)

        # DV01-neutral wing weights
        w2  = 0.5 * d5 / d2  if d2 > 0 else 0.5
        w10 = 0.5 * d5 / d10 if d10 > 0 else 0.5

        # Butterfly P&L: long 5Y belly, short wings
        # Daily carry = curvature of the par yield curve
        curvature_carry = (df["tsy_5y"] - 0.5 * df["tsy_2y"] - 0.5 * df["tsy_10y"]) / 252 * 100
        price_pnl = dy5 - w2 * dy2 - w10 * dy10

        if position_signal is None:
            pos = pd.Series(1, index=df.index)
        else:
            pos = position_signal.reindex(df.index).fillna(0)

        trades = pos.diff().abs() > 0.5
        pnl    = pos.shift(1) * (price_pnl + curvature_carry)
        pnl[trades] -= txn_cost_bps

        result = pd.DataFrame({
            "daily_pnl_bps":      pnl,
            "cumulative_pnl_bps": pnl.cumsum(),
            "position":           pos,
            "butterfly_bps":      (df["tsy_5y"] - 0.5*df["tsy_2y"] - 0.5*df["tsy_10y"]) * 100,
            "dy5_bps":            dy5,
        }).dropna()

        return result


# ── Multi-strategy portfolio ──────────────────────────────────────────────────

def run_multi_strategy_backtest(
    data: pd.DataFrame,
    steepener_signal: pd.Series | None = None,
    butterfly_signal: pd.Series | None = None,
    duration_signal:  pd.Series | None = None,
    weights: dict[str, float] | None = None,
    txn_cost_bps: float = 0.5,
) -> pd.DataFrame:
    """
    Combine three strategies into a portfolio with specified weights.

    Parameters
    ----------
    data              : market data DataFrame
    steepener_signal  : position signal for 2s10s steepener
    butterfly_signal  : position signal for 2s5s10s butterfly
    duration_signal   : position signal for outright 10Y duration
    weights           : {'steepener': w1, 'butterfly': w2, 'duration': w3}
                        default equal-weight

    Returns
    -------
    DataFrame with per-strategy and aggregate P&L
    """
    if weights is None:
        weights = {"steepener": 1/3, "butterfly": 1/3, "duration": 1/3}

    results = {}

    if "tsy_2y" in data.columns and "tsy_10y" in data.columns:
        st  = DV01NeutralSteepener()
        res = st.run_backtest(data, steepener_signal, txn_cost_bps)
        results["steepener"] = res["daily_pnl_bps"]

    if all(c in data.columns for c in ["tsy_2y", "tsy_5y", "tsy_10y"]):
        bt  = Butterfly()
        res = bt.run_backtest(data, butterfly_signal, txn_cost_bps)
        results["butterfly"] = res["daily_pnl_bps"]

    if "tsy_10y" in data.columns and duration_signal is not None:
        # Simple duration strategy: position × daily yield change × duration
        dy10   = data["tsy_10y"].diff() * 100
        dur    = 8.0  # approximate 10Y duration
        pos    = duration_signal.reindex(data.index).fillna(0)
        trades = pos.diff().abs() > 0.5
        pnl    = -pos.shift(1) * dur * dy10
        pnl[trades] -= txn_cost_bps
        results["duration"] = pnl

    if not results:
        raise ValueError("No strategies could be constructed from the provided data")

    common_idx = pd.concat(list(results.values()), axis=1).dropna().index
    portfolio_pnl = sum(
        weights.get(name, 0.0) * pnl.reindex(common_idx)
        for name, pnl in results.items()
    )

    out = pd.DataFrame({"daily_pnl_bps": portfolio_pnl})
    out["cumulative_pnl_bps"] = out["daily_pnl_bps"].cumsum()
    for name, pnl in results.items():
        out[f"{name}_pnl_bps"] = pnl.reindex(common_idx)
    return out


# ── Performance comparison ────────────────────────────────────────────────────

def compare_strategies(results: dict[str, pd.DataFrame]) -> pd.DataFrame:
    """
    Compare performance metrics across multiple strategy backtests.

    Parameters
    ----------
    results : dict of name → DataFrame with daily_pnl_bps column
    """
    rows = []
    for name, df in results.items():
        pnl  = df["daily_pnl_bps"].dropna()
        cpnl = pnl.cumsum()
        n    = len(pnl)
        rows.append({
            "strategy":          name,
            "sharpe":            sharpe_ratio(pnl),
            "ann_return_bps":    annualized_return(pnl),
            "max_drawdown_bps":  max_drawdown(cpnl),
            "total_pnl_bps":     cpnl.iloc[-1] if n > 0 else np.nan,
            "hit_rate":          (pnl > 0).mean(),
            "n_days":            n,
        })
    return pd.DataFrame(rows).set_index("strategy").round(3)
