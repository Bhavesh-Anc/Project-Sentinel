"""
Term Premium Decomposition for US Treasury yields.

Methodology
-----------
We use a simplified ACM-style (Adrian, Crump & Moench 2013) decomposition:

  y(t, n) = E_t[avg short rate over n years] + Term Premium(t, n)

The short-rate expectations path is estimated via an AR(1) model:
  r_{t+1} = ρ r_t + (1-ρ) μ + ε_t

The expected 10Y average short rate is the geometric series:
  E[avg_r] = μ + (r_0 - μ) × (1 - ρ^n) / (n × (1 - ρ))

Term premium = observed 10Y yield − E[avg_r]

A positive term premium compensates for duration risk (inflation uncertainty,
supply/demand, liquidity). ACM (2013) found the 10Y TP averaged ~1.5% pre-2008
and turned negative during QE before recovering post-2022.

Functions
---------
fit_ar1               : fit AR(1) to short rate series
short_rate_expectations: simulate expected short rate path using AR(1)
term_premium          : compute TP for a given tenor
rolling_term_premium  : apply over a historical dataset
"""
from __future__ import annotations
import numpy as np
import pandas as pd
from dataclasses import dataclass


@dataclass
class AR1Params:
    """AR(1) model parameters: r_t+1 = mu*(1-rho) + rho*r_t + eps."""
    mu:  float   # long-run mean
    rho: float   # persistence (0 < rho < 1)
    sigma: float # residual std dev


def fit_ar1(short_rate: pd.Series, min_obs: int = 60) -> AR1Params:
    """
    Fit AR(1) model to a short-rate series by OLS.

    Returns AR1Params. Clips rho to [0.80, 0.9999] for stability.
    """
    r = short_rate.dropna()
    if len(r) < min_obs:
        raise ValueError(f"Need at least {min_obs} observations, got {len(r)}")
    y = r.iloc[1:].values
    x = r.iloc[:-1].values
    # OLS: y = a + b*x + eps
    xm   = x.mean()
    ym   = y.mean()
    denom = np.dot(x - xm, x - xm)
    b    = np.dot(x - xm, y - ym) / denom if denom > 1e-20 else 0.0
    rho  = float(np.clip(b, 0.80, 0.9999))
    # Long-run mean = sample mean (exact for stationary AR(1), avoids mu=a/(1-rho)
    # inconsistency when b is clipped away from its OLS estimate)
    mu   = float(r.mean())
    resid = y - (mu * (1 - rho) + rho * x)
    return AR1Params(mu=mu, rho=rho, sigma=float(resid.std()))


def short_rate_expectations(
    current_rate: float,
    params: AR1Params,
    horizon_years: float = 10.0,
    freq: int = 12,
) -> np.ndarray:
    """
    Simulate the AR(1) expected path of the short rate.

    Parameters
    ----------
    current_rate   : today's short rate (decimal, e.g. 0.0433)
    params         : AR1Params from fit_ar1
    horizon_years  : forecast horizon
    freq           : steps per year (12 = monthly)

    Returns array of expected rates at each step.
    """
    n_steps = int(horizon_years * freq)
    path    = np.zeros(n_steps)
    # Monthly persistence = rho^(1/freq) to match freq
    rho_step = params.rho ** (1.0 / freq)
    r = current_rate
    for i in range(n_steps):
        r = params.mu + rho_step * (r - params.mu)
        path[i] = r
    return path


def expected_avg_short_rate(
    current_rate: float,
    params: AR1Params,
    horizon_years: float = 10.0,
) -> float:
    """
    Expected average short rate over the next horizon_years.
    Uses closed-form AR(1) average expectation.
    """
    rho = params.rho
    mu  = params.mu
    n   = horizon_years

    if abs(1 - rho) < 1e-8:  # random walk limit
        return current_rate

    # E[avg_r over n years] = mu + (r0 - mu) × (1 - rho^n) / (n*(1-rho))
    # rho here is annualized persistence
    return mu + (current_rate - mu) * (1 - rho**n) / (n * (1 - rho))


def term_premium(
    yield_10y: float,
    current_short_rate: float,
    params: AR1Params,
    horizon_years: float = 10.0,
) -> dict[str, float]:
    """
    Compute term premium for a single observation.

    Returns
    -------
    dict with:
      yield_10y          : observed 10Y yield
      expected_avg_rate  : E[avg short rate over 10Y]
      term_premium       : yield_10y - expected_avg_rate
      ar1_mu             : AR(1) long-run mean
      ar1_rho            : AR(1) persistence
    """
    exp_avg = expected_avg_short_rate(current_short_rate, params, horizon_years)
    return {
        "yield_10y":         yield_10y,
        "expected_avg_rate": exp_avg,
        "term_premium":      yield_10y - exp_avg,
        "ar1_mu":            params.mu,
        "ar1_rho":           params.rho,
    }


def rolling_term_premium(
    macro_df: pd.DataFrame,
    short_rate_col: str = "fed_funds",
    yield_10y_col:  str = "tsy_10y",
    estimation_window: int = 252 * 3,
    horizon_years: float = 10.0,
) -> pd.DataFrame:
    """
    Compute rolling term premium using an expanding/rolling AR(1) window.

    Parameters
    ----------
    macro_df           : DataFrame with short rate and 10Y yield columns (in %)
    short_rate_col     : column name for short rate (e.g. Fed Funds Rate)
    yield_10y_col      : column name for 10Y Treasury yield
    estimation_window  : days of data used for AR(1) estimation (default 3Y)
    horizon_years      : term premium horizon (default 10Y)

    Returns
    -------
    DataFrame with columns:
      yield_10y, expected_avg_rate, term_premium, ar1_mu, ar1_rho
    """
    cols = [short_rate_col, yield_10y_col]
    df   = macro_df[cols].dropna().copy()

    rows = []
    for i in range(estimation_window, len(df)):
        window = df.iloc[max(0, i - estimation_window):i]
        try:
            # Fit AR(1) on the estimation window
            params = fit_ar1(window[short_rate_col] / 100.0, min_obs=60)
            r0     = df[short_rate_col].iloc[i] / 100.0
            y10    = df[yield_10y_col].iloc[i] / 100.0
            tp     = term_premium(y10, r0, params, horizon_years)
            tp["date"] = df.index[i]
            rows.append(tp)
        except Exception:
            continue

    if not rows:
        return pd.DataFrame()

    result = pd.DataFrame(rows).set_index("date")
    # Convert to percentage points
    for col in ["yield_10y", "expected_avg_rate", "term_premium", "ar1_mu"]:
        result[col] *= 100.0
    return result


def term_premium_summary(tp_df: pd.DataFrame) -> dict:
    """Summary statistics for a term premium time series."""
    if tp_df.empty:
        return {}
    tp = tp_df["term_premium"].dropna()
    return {
        "mean_tp_pct":      tp.mean(),
        "std_tp_pct":       tp.std(),
        "current_tp_pct":   tp.iloc[-1],
        "max_tp_pct":       tp.max(),
        "min_tp_pct":       tp.min(),
        "pct_positive":     (tp > 0).mean(),
        "date_max":         tp.idxmax(),
        "date_min":         tp.idxmin(),
    }
