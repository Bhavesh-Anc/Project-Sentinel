"""
Regime-conditional performance attribution for backtests.

Functions
---------
label_macro_regimes     : classify each date into a macro regime
label_curve_regimes     : classify using Nelson-Siegel β₁ factor
performance_by_regime   : compute full metrics split by regime
regime_transition_matrix: compute regime transition probabilities
plot_regime_attribution : chart cumulative P&L coloured by regime
"""
from __future__ import annotations
import numpy as np
import pandas as pd
from typing import Sequence

from .performance import sharpe_ratio, max_drawdown, annualized_return, hit_rate


# ── Macro regime labels ───────────────────────────────────────────────────────

def label_macro_regimes(
    macro_df: pd.DataFrame,
    fed_funds_col:  str = "effr",
    taylor_col:     str = "taylor_rate",
    inflation_col:  str = "core_pce_yoy",
    inflation_target: float = 2.0,
    threshold_bps:  float = 75.0,
) -> pd.Series:
    """
    Classify each date into one of four macro regimes based on Fed policy stance.

    Regimes
    -------
    'restrictive_hot'    : EFFR >> Taylor AND inflation > target
    'restrictive_cooling': EFFR >> Taylor AND inflation <= target
    'neutral'            : EFFR ≈ Taylor (within threshold)
    'accommodative'      : EFFR << Taylor

    Returns pd.Series of regime labels with same index as macro_df.
    """
    thresh = threshold_bps / 100.0   # to same units as rates in %

    gap    = macro_df[fed_funds_col] - macro_df[taylor_col]
    infl   = macro_df[inflation_col]

    regime = pd.Series("neutral", index=macro_df.index, name="macro_regime")
    regime[gap >  thresh]                               = "restrictive_neutral"
    regime[(gap > thresh) & (infl > inflation_target)]  = "restrictive_hot"
    regime[(gap > thresh) & (infl <= inflation_target)] = "restrictive_cooling"
    regime[gap < -thresh]                               = "accommodative"
    return regime


def label_curve_regimes(
    ns_factors: pd.DataFrame,
    beta1_col:  str = "beta1",
    steep_threshold: float = -1.0,
    flat_threshold:  float = -0.3,
    inv_threshold:   float = 0.5,
) -> pd.Series:
    """
    Classify yield curve regime using Nelson-Siegel β₁ (slope) factor.

    Regimes
    -------
    'steep'      : β₁ < steep_threshold    (very normal/steep)
    'normal'     : steep_threshold ≤ β₁ < flat_threshold
    'flat'       : flat_threshold ≤ β₁ < inv_threshold
    'inverted'   : β₁ ≥ inv_threshold
    """
    b1     = ns_factors[beta1_col]
    regime = pd.Series("normal", index=ns_factors.index, name="curve_regime")
    regime[b1 < steep_threshold]  = "steep"
    regime[b1 >= inv_threshold]   = "inverted"
    regime[(b1 >= flat_threshold) & (b1 < inv_threshold)] = "flat"
    return regime


# ── Performance by regime ─────────────────────────────────────────────────────

def performance_by_regime(
    backtest_df: pd.DataFrame,
    regime_series: pd.Series,
    pnl_col: str = "daily_pnl_bps",
    pos_col: str = "position",
) -> pd.DataFrame:
    """
    Compute performance metrics split by regime.

    Parameters
    ----------
    backtest_df   : output of WalkForwardBacktest.run() (or similar)
    regime_series : pd.Series of regime labels (aligned by date)
    pnl_col       : column name for daily P&L
    pos_col       : column name for position

    Returns
    -------
    DataFrame indexed by regime name with columns:
      sharpe, ann_return_bps, max_drawdown_bps, hit_rate,
      total_pnl_bps, n_days, pct_time
    """
    merged = backtest_df[[pnl_col, pos_col]].copy()
    merged["regime"] = regime_series.reindex(merged.index, method="ffill")
    merged.dropna(subset=["regime"], inplace=True)

    total_days = len(merged)
    rows = []
    for regime, grp in merged.groupby("regime"):
        pnl  = grp[pnl_col].dropna()
        cpnl = pnl.cumsum()
        n    = len(pnl)
        if n < 5:
            continue
        rows.append({
            "regime":           regime,
            "sharpe":           sharpe_ratio(pnl),
            "ann_return_bps":   annualized_return(pnl),
            "max_drawdown_bps": max_drawdown(cpnl),
            "hit_rate":         (pnl > 0).mean(),
            "total_pnl_bps":    cpnl.iloc[-1] if n > 0 else np.nan,
            "n_days":           n,
            "pct_time":         n / total_days,
        })

    df = pd.DataFrame(rows).set_index("regime")
    return df.round(3) if not df.empty else df


# ── Regime transition matrix ──────────────────────────────────────────────────

def regime_transition_matrix(regime_series: pd.Series) -> pd.DataFrame:
    """
    Compute empirical transition probability matrix between regimes.

    Returns DataFrame where entry [i, j] is P(next=j | current=i).
    """
    regimes = regime_series.dropna()
    states  = sorted(regimes.unique())
    counts  = pd.DataFrame(0, index=states, columns=states, dtype=float)
    for t in range(len(regimes) - 1):
        cur  = regimes.iloc[t]
        nxt  = regimes.iloc[t + 1]
        counts.loc[cur, nxt] += 1
    totals = counts.sum(axis=1)
    probs  = counts.div(totals, axis=0).fillna(0.0)
    probs.index.name   = "from"
    probs.columns.name = "to"
    return probs.round(3)


# ── Regime duration stats ─────────────────────────────────────────────────────

def regime_duration_stats(regime_series: pd.Series) -> pd.DataFrame:
    """
    For each regime, compute average and max duration of continuous episodes.

    Returns DataFrame with columns: regime, n_episodes, mean_days, max_days, total_days.
    """
    regimes = regime_series.dropna()
    rows    = []
    prev    = None
    run_len = 0
    episodes: dict[str, list[int]] = {}

    for val in regimes:
        if val == prev:
            run_len += 1
        else:
            if prev is not None:
                episodes.setdefault(prev, []).append(run_len)
            prev    = val
            run_len = 1
    if prev is not None:
        episodes.setdefault(prev, []).append(run_len)

    for regime, runs in sorted(episodes.items()):
        rows.append({
            "regime":      regime,
            "n_episodes":  len(runs),
            "mean_days":   np.mean(runs),
            "max_days":    max(runs),
            "total_days":  sum(runs),
        })
    return pd.DataFrame(rows).set_index("regime").round(1)


# ── Consolidated attribution report ──────────────────────────────────────────

def attribution_report(
    backtest_df: pd.DataFrame,
    macro_df: pd.DataFrame | None = None,
    ns_factors: pd.DataFrame | None = None,
) -> dict[str, pd.DataFrame]:
    """
    Generate a complete attribution report.

    Returns dict with keys:
      'by_macro_regime'  (if macro_df provided)
      'by_curve_regime'  (if ns_factors provided)
      'macro_transitions'
      'curve_transitions'
      'macro_durations'
      'curve_durations'
    """
    out = {}

    if macro_df is not None:
        macro_regime = label_macro_regimes(macro_df)
        out["by_macro_regime"]   = performance_by_regime(backtest_df, macro_regime)
        out["macro_transitions"] = regime_transition_matrix(macro_regime)
        out["macro_durations"]   = regime_duration_stats(macro_regime)

    if ns_factors is not None:
        curve_regime = label_curve_regimes(ns_factors)
        out["by_curve_regime"]   = performance_by_regime(backtest_df, curve_regime)
        out["curve_transitions"] = regime_transition_matrix(curve_regime)
        out["curve_durations"]   = regime_duration_stats(curve_regime)

    return out
