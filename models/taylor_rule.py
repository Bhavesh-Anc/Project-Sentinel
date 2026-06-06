"""
Taylor Rule model for Fed Funds Rate forecasting.

The classic Taylor (1993) Rule:
    r* = r_neutral + α(π - π*) + β(y - y*)

where:
    r_neutral = 2.5%    (Fed's long-run neutral rate estimate)
    π         = core PCE inflation (12-month trailing)
    π*        = 2.0%    (Fed's symmetric inflation target)
    y - y*    = output gap (≈ Okun's law: -2 × unemployment gap)
    α = β     = 0.5     (standard Rudebusch 2001 calibration)

We also implement:
    Inertial Taylor Rule (rho × r_{t-1} + (1-rho) × r*_t)
    Balanced Approach Rule (Bernanke et al. 2019 — double weight on output gap)
    OLS estimation of α, β on pre-ZLB data (2015–2019)
"""
from __future__ import annotations
import numpy as np
import pandas as pd
from dataclasses import dataclass
from typing import Literal


NAIRU     = 4.0   # Natural rate of unemployment (CBO midpoint)
NEUTRAL_R = 2.5   # Long-run neutral nominal rate (Fed "dot", r-star + π*)
PI_STAR   = 2.0   # Fed's symmetric inflation target (core PCE %)


@dataclass
class TaylorRuleConfig:
    neutral_rate:     float = NEUTRAL_R
    inflation_target: float = PI_STAR
    nairu:            float = NAIRU
    alpha:            float = 0.5   # inflation gap weight
    beta:             float = 0.5   # output gap weight
    okun_multiplier:  float = -2.0  # maps unemployment gap → output gap
    inertia:          float = 0.0   # rho in inertial rule (0 = no inertia)
    variant: Literal["standard", "balanced", "inertial"] = "standard"


def compute_taylor_rule(
    inflation: pd.Series,
    unemployment: pd.Series,
    config: TaylorRuleConfig | None = None,
    lagged_rate: pd.Series | None = None,
) -> pd.DataFrame:
    """
    Compute Taylor Rule implied Fed Funds Rate.

    Parameters
    ----------
    inflation    : core PCE or core CPI, year-over-year %, same index as unemployment
    unemployment : U-3 unemployment rate %
    config       : TaylorRuleConfig (uses standard calibration if None)
    lagged_rate  : actual Fed Funds Rate lagged 1 period (for inertial rule)

    Returns
    -------
    DataFrame with columns:
        inflation_gap, unemployment_gap, output_gap_proxy,
        taylor_rate, [taylor_inertial], policy_deviation
    """
    if config is None:
        config = TaylorRuleConfig()

    df = pd.DataFrame(index=inflation.index)
    df["inflation"]       = inflation
    df["unemployment"]    = unemployment

    df["inflation_gap"]    = df["inflation"] - config.inflation_target
    df["unemployment_gap"] = df["unemployment"] - config.nairu
    df["output_gap_proxy"] = config.okun_multiplier * df["unemployment_gap"]

    if config.variant == "balanced":
        # Bernanke 2019: double weight on output gap
        alpha = config.alpha
        beta  = config.beta * 2
    else:
        alpha = config.alpha
        beta  = config.beta

    df["taylor_rate"] = (
        config.neutral_rate
        + alpha * df["inflation_gap"]
        + beta  * df["output_gap_proxy"]
    )

    # Inertial version: smooth the rate recommendation
    if config.inertia > 0 and lagged_rate is not None:
        rho = config.inertia
        df["taylor_inertial"] = rho * lagged_rate + (1 - rho) * df["taylor_rate"]

    return df


def estimate_taylor_rule(
    actual_rate: pd.Series,
    inflation: pd.Series,
    unemployment: pd.Series,
    neutral_rate: float = NEUTRAL_R,
    inflation_target: float = PI_STAR,
    nairu: float = NAIRU,
    okun_multiplier: float = -2.0,
) -> dict:
    """
    Estimate Taylor Rule coefficients via OLS.
    Regresses: r_t = const + α × inflation_gap_t + β × output_gap_t

    Use on pre-ZLB sample (e.g., 2015-2019) to avoid bias from
    the effective lower bound period (2020-2022).

    Returns
    -------
    dict with keys: alpha_hat, beta_hat, const_hat, r_squared, summary
    """
    from statsmodels.regression.linear_model import OLS
    from statsmodels.tools import add_constant

    idx = inflation.index.intersection(unemployment.index).intersection(actual_rate.index)
    inf_gap  = (inflation.loc[idx] - inflation_target).values
    unemp    = unemployment.loc[idx]
    out_gap  = (okun_multiplier * (unemp - nairu)).values
    y        = (actual_rate.loc[idx] - neutral_rate).values

    X = add_constant(np.column_stack([inf_gap, out_gap]))
    model = OLS(y, X).fit()

    return {
        "const_hat":  model.params[0],
        "alpha_hat":  model.params[1],
        "beta_hat":   model.params[2],
        "r_squared":  model.rsquared,
        "summary":    model.summary(),
    }


def taylor_policy_gap(
    actual_rate: pd.Series,
    taylor_df: pd.DataFrame,
) -> pd.Series:
    """
    Compute the policy gap: actual Fed Funds Rate minus Taylor Rule implied rate.
    Positive = Fed is above Taylor Rule (tight policy)
    Negative = Fed is below Taylor Rule (accommodative)
    """
    aligned = taylor_df["taylor_rate"].reindex(actual_rate.index, method="ffill")
    return actual_rate - aligned


def classify_policy_stance(gap: pd.Series, threshold_bps: float = 75.0) -> pd.Series:
    """
    Classify Fed policy stance based on Taylor gap.
    Returns a Series of: 'tight', 'neutral', 'accommodative'
    """
    threshold = threshold_bps / 100.0
    conditions = [
        gap >  threshold,
        gap < -threshold,
    ]
    choices = ["tight", "accommodative"]
    return pd.Series(
        np.select(conditions, choices, default="neutral"),
        index=gap.index,
    )


if __name__ == "__main__":
    # Quick demonstration with synthetic data
    dates = pd.date_range("2018-01-01", "2024-12-31", freq="MS")
    np.random.seed(42)

    inflation    = pd.Series(2.5 + np.random.randn(len(dates)) * 0.5, index=dates)
    unemployment = pd.Series(4.5 + np.random.randn(len(dates)) * 0.3, index=dates)

    result = compute_taylor_rule(inflation, unemployment)
    print("Taylor Rule (standard):")
    print(result[["inflation_gap", "unemployment_gap", "taylor_rate"]].tail(12).round(3))
