"""
Performance metrics for the rates trading strategy backtest.

Standard quant metrics:
  - Annualized Sharpe Ratio (primary)
  - Maximum Drawdown (absolute and duration)
  - Calmar Ratio (annualized return / max drawdown)
  - Hit Rate (% profitable trading days / signal periods)
  - Win/Loss Ratio
  - Annualized Return (in bps and %)
  - Volatility (annualized daily P&L std)
  - Regime-conditional performance (bull/bear/sideways rates)

All bps-denominated (basis points of notional).
To convert to dollar P&L: multiply by notional / 10_000.
"""
from __future__ import annotations
import numpy as np
import pandas as pd
from typing import Literal


TRADING_DAYS_PER_YEAR = 252


def sharpe_ratio(
    daily_pnl: pd.Series,
    risk_free_rate_annual: float = 0.0,
    annualize: bool = True,
) -> float:
    """
    Annualized Sharpe ratio.
    daily_pnl: daily P&L in bps (or any consistent units).
    risk_free_rate_annual: annualized risk-free rate in same units.
    """
    daily_rf = risk_free_rate_annual / TRADING_DAYS_PER_YEAR
    excess   = daily_pnl - daily_rf
    if excess.std() < 1e-10:
        return np.nan
    sr = excess.mean() / excess.std()
    return float(sr * np.sqrt(TRADING_DAYS_PER_YEAR) if annualize else sr)


def max_drawdown(cumulative_pnl: pd.Series) -> float:
    """Maximum peak-to-trough drawdown in the same units as cumulative_pnl."""
    rolling_max = cumulative_pnl.cummax()
    drawdown    = cumulative_pnl - rolling_max
    return float(drawdown.min())


def max_drawdown_duration(cumulative_pnl: pd.Series) -> int:
    """Length of the longest drawdown period in business days."""
    rolling_max = cumulative_pnl.cummax()
    underwater  = cumulative_pnl < rolling_max

    max_dur = 0
    curr_dur = 0
    for uw in underwater:
        if uw:
            curr_dur += 1
            max_dur   = max(max_dur, curr_dur)
        else:
            curr_dur  = 0
    return max_dur


def calmar_ratio(daily_pnl: pd.Series, cumulative_pnl: pd.Series) -> float:
    """Annualized return / |max drawdown|. Higher is better."""
    ann_return = annualized_return(daily_pnl)
    mdd        = abs(max_drawdown(cumulative_pnl))
    if mdd < 1e-10:
        return np.nan
    return float(ann_return / mdd)


def annualized_return(daily_pnl: pd.Series) -> float:
    """Annualized P&L (bps per year) = mean daily P&L × 252."""
    return float(daily_pnl.mean() * TRADING_DAYS_PER_YEAR)


def annualized_volatility(daily_pnl: pd.Series) -> float:
    """Annualized daily P&L standard deviation = σ × √252."""
    return float(daily_pnl.std() * np.sqrt(TRADING_DAYS_PER_YEAR))


def hit_rate(daily_pnl: pd.Series, active_only: bool = True, position: pd.Series | None = None) -> float:
    """
    Fraction of days (or signal-active days) with positive P&L.
    If position is provided and active_only=True, only count days with non-zero position.
    """
    if active_only and position is not None:
        pnl = daily_pnl[position != 0]
    else:
        pnl = daily_pnl
    if len(pnl) == 0:
        return np.nan
    return float((pnl > 0).mean())


def win_loss_ratio(daily_pnl: pd.Series) -> float:
    """Average win / |average loss|. >1 means average win exceeds average loss."""
    wins   = daily_pnl[daily_pnl > 0]
    losses = daily_pnl[daily_pnl < 0]
    if len(losses) == 0 or len(wins) == 0:
        return np.nan
    return float(wins.mean() / abs(losses.mean()))


def profit_factor(daily_pnl: pd.Series) -> float:
    """Gross profits / |gross losses|."""
    gross_profit = daily_pnl[daily_pnl > 0].sum()
    gross_loss   = abs(daily_pnl[daily_pnl < 0].sum())
    if gross_loss < 1e-10:
        return np.nan
    return float(gross_profit / gross_loss)


def turnover(position: pd.Series) -> float:
    """
    Annualized turnover: number of position changes per year.
    Each change from -1→0, 0→1, 1→-1, etc. counts as 1 trade.
    """
    changes = (position.diff().abs() > 0).sum()
    years   = len(position) / TRADING_DAYS_PER_YEAR
    return float(changes / years) if years > 0 else np.nan


def full_metrics(result_df: pd.DataFrame, position_col: str = "position") -> dict:
    """
    Compute the full suite of performance metrics from a backtest result DataFrame.
    Expects columns: daily_pnl_bps, cumulative_pnl_bps, position.
    """
    pnl  = result_df["daily_pnl_bps"]
    cpnl = result_df["cumulative_pnl_bps"]
    pos  = result_df[position_col] if position_col in result_df.columns else None

    return {
        "annualized_return_bps":  annualized_return(pnl),
        "annualized_vol_bps":     annualized_volatility(pnl),
        "sharpe_ratio":           sharpe_ratio(pnl),
        "max_drawdown_bps":       max_drawdown(cpnl),
        "max_dd_duration_days":   max_drawdown_duration(cpnl),
        "calmar_ratio":           calmar_ratio(pnl, cpnl),
        "hit_rate":               hit_rate(pnl, active_only=True, position=pos),
        "win_loss_ratio":         win_loss_ratio(pnl),
        "profit_factor":          profit_factor(pnl),
        "total_pnl_bps":          float(cpnl.iloc[-1]) if len(cpnl) > 0 else np.nan,
        "n_trading_days":         len(pnl),
        "active_days":            int((pos != 0).sum()) if pos is not None else len(pnl),
        "turnover_per_year":      turnover(pos) if pos is not None else np.nan,
        "long_pct":               float((pos == 1).mean()) if pos is not None else np.nan,
        "short_pct":              float((pos == -1).mean()) if pos is not None else np.nan,
    }


def print_tearsheet(metrics: dict, title: str = "Strategy Performance") -> None:
    """Pretty-print the performance metrics tearsheet."""
    print(f"\n{'=' * 55}")
    print(f"  {title}")
    print(f"{'=' * 55}")
    print(f"  {'Annualized Return':30s}: {metrics['annualized_return_bps']:>8.1f} bps/yr")
    print(f"  {'Annualized Volatility':30s}: {metrics['annualized_vol_bps']:>8.1f} bps/yr")
    print(f"  {'Sharpe Ratio':30s}: {metrics['sharpe_ratio']:>8.3f}")
    print(f"  {'Max Drawdown':30s}: {metrics['max_drawdown_bps']:>8.1f} bps")
    print(f"  {'Max DD Duration':30s}: {metrics['max_dd_duration_days']:>8d} days")
    print(f"  {'Calmar Ratio':30s}: {metrics['calmar_ratio']:>8.3f}")
    print(f"  {'Hit Rate (active days)':30s}: {metrics['hit_rate']:>8.1%}")
    print(f"  {'Win/Loss Ratio':30s}: {metrics['win_loss_ratio']:>8.2f}")
    print(f"  {'Profit Factor':30s}: {metrics['profit_factor']:>8.2f}")
    print(f"  {'Total P&L':30s}: {metrics['total_pnl_bps']:>8.1f} bps")
    print(f"  {'Turnover (trades/yr)':30s}: {metrics['turnover_per_year']:>8.1f}")
    print(f"  {'Long %':30s}: {metrics['long_pct']:>8.1%}")
    print(f"  {'Short %':30s}: {metrics['short_pct']:>8.1%}")
    print(f"{'=' * 55}\n")


def regime_performance(
    result_df: pd.DataFrame,
    regime_col: str,
) -> pd.DataFrame:
    """
    Compute Sharpe ratio and mean daily P&L conditioned on market regime.
    Useful for understanding when the strategy works vs. fails.
    """
    if regime_col not in result_df.columns:
        raise ValueError(f"Column '{regime_col}' not in result_df")

    rows = []
    for regime, grp in result_df.groupby(regime_col):
        pnl = grp["daily_pnl_bps"]
        rows.append({
            "regime":         regime,
            "n_days":         len(grp),
            "mean_pnl_bps":   float(pnl.mean()),
            "sharpe":         sharpe_ratio(pnl),
            "hit_rate":       hit_rate(pnl),
            "total_pnl_bps":  float(pnl.sum()),
        })
    return pd.DataFrame(rows).set_index("regime")


if __name__ == "__main__":
    # Synthetic sanity check
    np.random.seed(42)
    n = 252 * 3  # 3 years
    pnl = pd.Series(np.random.randn(n) * 3 + 0.5)
    cpnl = pnl.cumsum()
    pos  = pd.Series(np.sign(np.random.randn(n)).astype(int))

    df = pd.DataFrame({"daily_pnl_bps": pnl, "cumulative_pnl_bps": cpnl, "position": pos})
    metrics = full_metrics(df)
    print_tearsheet(metrics, "Synthetic Strategy")
