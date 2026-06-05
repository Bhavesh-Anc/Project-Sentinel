"""
Macro signal construction for US rates trading strategy.

Signals built from FRED macro data:
1. Inflation momentum  — core PCE 3-month annualized change
2. Labor market        — unemployment gap, JOLTS / payrolls momentum
3. Fed policy gap      — actual EFFR vs. Taylor Rule implied rate
4. Curve regime        — Nelson-Siegel slope / curvature factors
5. Financial conditions— VIX, credit spreads, SOFR-EFFR basis

Composite signal:
    Long 10Y Treasuries (steepener) when:
      - Fed is above Taylor Rule (tight: policy gap > +75bps)
      - Inflation is decelerating (3m PCE mom < 6m PCE mom)
      - Unemployment is rising (gap widening)
    Short (flattener / outright short) when opposite conditions.
"""
from __future__ import annotations
import numpy as np
import pandas as pd
from typing import Literal


SIGNAL_THRESHOLD_BPS = 75.0   # Policy gap threshold for signal activation


# ── Individual signal builders ────────────────────────────────────────────────

def inflation_momentum_signal(
    core_pce: pd.Series,
    window_short: int = 63,    # ~3 months
    window_long:  int = 126,   # ~6 months
) -> pd.Series:
    """
    Signal = +1 (bullish rates / long bonds) when inflation is decelerating.
    Measure: 3-month annualized core PCE < 6-month annualized core PCE.

    Returns a Series of {-1, 0, +1}.
    """
    pce_3m = core_pce.pct_change(window_short) * (252 / window_short) * 100
    pce_6m = core_pce.pct_change(window_long)  * (252 / window_long)  * 100

    signal = pd.Series(0, index=core_pce.index, dtype=int)
    signal[pce_3m < pce_6m] =  1   # decelerating → bullish rates (long bonds)
    signal[pce_3m > pce_6m] = -1   # accelerating → bearish rates (short bonds)
    return signal.rename("inflation_signal")


def labor_market_signal(
    unemployment: pd.Series,
    nairu: float = 4.0,
    momentum_window: int = 63,
) -> pd.Series:
    """
    Signal = +1 when unemployment is rising above NAIRU (slack building → dovish Fed).
    Signal = -1 when unemployment is falling below NAIRU (tight labor → hawkish Fed).
    """
    gap        = unemployment - nairu
    gap_change = gap.diff(momentum_window)

    signal = pd.Series(0, index=unemployment.index, dtype=int)
    signal[(gap > 0) & (gap_change > 0)] =  1   # rising slack → bullish rates
    signal[(gap < 0) & (gap_change < 0)] = -1   # tightening labor → bearish
    return signal.rename("labor_signal")


def policy_gap_signal(
    actual_rate: pd.Series,
    taylor_rate: pd.Series,
    threshold_bps: float = SIGNAL_THRESHOLD_BPS,
) -> pd.Series:
    """
    Signal based on distance between actual Fed Funds and Taylor Rule.
    +1 when Fed is materially above Taylor Rule (overtightening → eventual cuts)
    -1 when Fed is materially below Taylor Rule (behind the curve → eventual hikes)
    """
    threshold = threshold_bps / 100.0
    gap = actual_rate - taylor_rate

    signal = pd.Series(0, index=actual_rate.index, dtype=int)
    signal[gap >  threshold] =  1   # over-tightened → eventual cuts → long bonds
    signal[gap < -threshold] = -1   # behind curve   → eventual hikes → short bonds
    return signal.rename("policy_gap_signal")


def curve_slope_signal(
    slope_2s10s: pd.Series,
    lookback_pct: float = 0.15,
    window: int = 252,
) -> pd.Series:
    """
    Signal based on yield curve slope percentile.
    +1 when curve is unusually flat/inverted (historically precedes steepening)
    -1 when curve is unusually steep (historically precedes flattening)

    lookback_pct: enter signal when slope below this percentile (for +1 case)
    """
    rolling_pct = slope_2s10s.rolling(window).rank(pct=True)

    signal = pd.Series(0, index=slope_2s10s.index, dtype=int)
    signal[rolling_pct < lookback_pct]         =  1   # unusually flat → buy steepener
    signal[rolling_pct > (1 - lookback_pct)]   = -1   # unusually steep → sell / flattener
    return signal.rename("curve_slope_signal")


def financial_conditions_signal(
    vix: pd.Series,
    credit_spread_hy: pd.Series | None = None,
    vix_threshold: float = 25.0,
) -> pd.Series:
    """
    Simple financial stress signal.
    +1 when VIX > threshold (flight to quality → rates rally / yields fall)
    0  otherwise (don't use FCI as a standalone fade signal — too noisy)
    """
    signal = pd.Series(0, index=vix.index, dtype=int)
    signal[vix > vix_threshold] = 1
    if credit_spread_hy is not None:
        # Additional confirmation: IG spread widening
        ig_widening = credit_spread_hy.diff(21) > 0.5  # 50bps in a month
        signal[(vix > vix_threshold) & ~ig_widening] = 0  # require confirmation
    return signal.rename("fci_signal")


# ── Composite signal ──────────────────────────────────────────────────────────

def composite_signal(
    df: pd.DataFrame,
    weights: dict[str, float] | None = None,
) -> pd.DataFrame:
    """
    Combine individual signals into a composite score.
    df must have columns for the inputs needed by each sub-signal.
    weights: signal_name → weight in composite (default: equal weight)

    Returns a DataFrame with individual signals + composite + position.
    """
    out = df.copy()

    # Build each signal where data available
    sig_cols = []

    if "core_pce" in df.columns:
        out["inflation_signal"] = inflation_momentum_signal(df["core_pce"])
        sig_cols.append("inflation_signal")

    if "unemployment" in df.columns:
        out["labor_signal"] = labor_market_signal(df["unemployment"])
        sig_cols.append("labor_signal")

    if all(c in df.columns for c in ["fed_funds", "taylor_rate"]):
        out["policy_gap_signal"] = policy_gap_signal(df["fed_funds"], df["taylor_rate"])
        sig_cols.append("policy_gap_signal")

    if "slope_2s10s" in df.columns:
        out["curve_slope_signal"] = curve_slope_signal(df["slope_2s10s"])
        sig_cols.append("curve_slope_signal")

    if "vix" in df.columns:
        out["fci_signal"] = financial_conditions_signal(df["vix"])
        sig_cols.append("fci_signal")

    if not sig_cols:
        raise ValueError("No signal components could be computed from the provided DataFrame")

    # Default: equal weights
    if weights is None:
        w = {col: 1.0 / len(sig_cols) for col in sig_cols}
    else:
        w = weights

    # Composite score in [-1, +1]
    out["composite_score"] = sum(w.get(col, 0) * out[col] for col in sig_cols)

    # Discrete position: threshold at 0.33 (majority of signals must agree)
    out["position"] = 0
    out.loc[out["composite_score"] >  0.33, "position"] =  1   # long duration
    out.loc[out["composite_score"] < -0.33, "position"] = -1   # short duration

    # Signal quality: how many sub-signals agree
    out["signal_agreement"] = out[sig_cols].apply(
        lambda row: (row > 0).sum() - (row < 0).sum(), axis=1
    )

    return out


def regime_filter(
    signal_df: pd.DataFrame,
    ns_df: pd.DataFrame | None = None,
    min_vol_percentile: float = 0.05,
    max_vol_percentile: float = 0.95,
) -> pd.DataFrame:
    """
    Apply regime filters to avoid trading in extreme volatility environments.
    Masks signals when VIX is in extreme tails (too noisy or crowded trade).
    """
    df = signal_df.copy()

    if "vix" in df.columns:
        vix_pct = df["vix"].rolling(252).rank(pct=True)
        # Mask positions during extreme VIX spikes (above 95th pct) — liquidity breaks down
        extreme_vol = vix_pct > max_vol_percentile
        df.loc[extreme_vol, "position"] = 0

    return df


if __name__ == "__main__":
    # Synthetic demo
    dates = pd.bdate_range("2018-01-01", "2024-12-31")
    np.random.seed(42)
    n = len(dates)

    df = pd.DataFrame({
        "core_pce":    100 * np.cumprod(1 + np.random.randn(n) * 0.001),
        "unemployment": 4.0 + np.random.randn(n) * 0.3,
        "fed_funds":   np.clip(2.5 + np.random.randn(n).cumsum() * 0.01, 0, 6),
        "taylor_rate": np.clip(2.0 + np.random.randn(n).cumsum() * 0.01, 0, 7),
        "slope_2s10s": np.random.randn(n) * 0.5,
        "vix":         20 + np.abs(np.random.randn(n)) * 5,
    }, index=dates)

    result = composite_signal(df)
    pos_dist = result["position"].value_counts()
    print("Position distribution:")
    print(pos_dist)
    print(f"\nSignal coverage: {(result['position'] != 0).mean():.1%} of days with active position")
