"""
FOMC meeting probability model.

Two approaches:

1. Market-implied (from SOFR futures / OIS):
   Extract the probability distribution of Fed Funds Rate outcomes
   at each FOMC meeting date from the SOFR futures curve.
   This replicates the CME FedWatch methodology.

2. Statistical model (logistic regression):
   Predict P(hike), P(hold), P(cut) from macro variables:
   - Taylor policy gap (actual rate vs. Taylor Rule)
   - CPI/PCE momentum (3-month annualized change)
   - Unemployment gap
   - Financial conditions index proxy (VIX, credit spreads)

FOMC Meeting Dates (2024-2026):
The FOMC meets ~8 times per year. We store a schedule and can extend.

Market-implied probability calculation:
    At each meeting, the Fed can move in 25bp increments (or hold).
    Using the SR1 (1-month SOFR futures) for the meeting month:
        implied_rate_after_meeting ≈ (100 - SR1_price) / 100
        P(hike) = max(0, implied_rate - current_rate) / 0.25
        P(cut)  = max(0, current_rate - implied_rate) / 0.25
        P(hold) = 1 - P(hike) - P(cut)
    More precise: fit the full probability distribution across possible outcomes.
"""
from __future__ import annotations
import numpy as np
import pandas as pd
from datetime import date
from typing import Literal


# ── FOMC meeting schedule ─────────────────────────────────────────────────────

FOMC_DATES_2024_2026 = [
    # 2024
    date(2024, 1, 31),
    date(2024, 3, 20),
    date(2024, 5, 1),
    date(2024, 6, 12),
    date(2024, 7, 31),
    date(2024, 9, 18),
    date(2024, 11, 7),
    date(2024, 12, 18),
    # 2025
    date(2025, 1, 29),
    date(2025, 3, 19),
    date(2025, 5, 7),
    date(2025, 6, 18),
    date(2025, 7, 30),
    date(2025, 9, 17),
    date(2025, 11, 5),
    date(2025, 12, 17),
    # 2026
    date(2026, 1, 28),
    date(2026, 3, 18),
    date(2026, 5, 6),
    date(2026, 6, 17),
    date(2026, 7, 29),
    date(2026, 9, 16),
    date(2026, 11, 4),
    date(2026, 12, 16),
]


def next_fomc(as_of: date, n: int = 1) -> list[date]:
    """Return the next n FOMC meeting dates after as_of."""
    upcoming = [d for d in FOMC_DATES_2024_2026 if d > as_of]
    return upcoming[:n]


# ── Market-implied probabilities from SOFR futures ────────────────────────────

def implied_rate_from_sr1(
    sr1_price: float,
    meeting_day: int,
    days_in_month: int,
    rate_before_meeting: float,
) -> float:
    """
    Extract the post-meeting Fed Funds rate implied by an SR1 futures contract.

    SR1 settles to the arithmetic average SOFR in the delivery month.
    If the meeting is on day D of an M-day month:
        SR1_implied_avg = (D/M) × r_before + ((M-D)/M) × r_after
        r_after = (SR1_avg - (D/M) × r_before) × M / (M - D)

    Parameters
    ----------
    sr1_price           : SR1 futures settlement price (e.g. 94.70)
    meeting_day         : calendar day of FOMC announcement in the month
    days_in_month       : total calendar days in the delivery month
    rate_before_meeting : current Fed Funds rate before the meeting (decimal)

    Returns
    -------
    Implied post-meeting rate (decimal)
    """
    sr1_implied_avg = (100.0 - sr1_price) / 100.0
    weight_before   = meeting_day / days_in_month
    weight_after    = 1.0 - weight_before

    if weight_after < 1e-6:
        return sr1_implied_avg  # meeting is last day; no post-meeting days

    r_after = (sr1_implied_avg - weight_before * rate_before_meeting) / weight_after
    return r_after


def fedwatch_probabilities(
    implied_rate_after: float,
    current_rate: float,
    step_size: float = 0.0025,
    max_moves: int = 3,
) -> dict[int, float]:
    """
    Compute CME FedWatch-style probability distribution over possible Fed outcomes.

    Maps the continuous implied rate to a discrete distribution over
    {-max_moves*step, ..., -step, 0, +step, ..., +max_moves*step} moves.

    Uses linear interpolation between adjacent 25bp outcomes.

    Parameters
    ----------
    implied_rate_after : post-meeting rate implied by futures
    current_rate       : current Federal Funds target (upper bound) decimal
    step_size          : standard move size (0.0025 = 25bps)
    max_moves          : max moves in either direction to consider

    Returns
    -------
    dict mapping move_in_bps (int) → probability (float), sums to ~1.0
    """
    rate_change = implied_rate_after - current_rate
    change_in_steps = rate_change / step_size

    # Bound the implied change
    change_in_steps = np.clip(change_in_steps, -max_moves, max_moves)

    # Two adjacent integer outcomes that bracket the continuous implied change
    lower_n = int(np.floor(change_in_steps))
    upper_n = int(np.ceil(change_in_steps))

    if lower_n == upper_n:
        return {int(lower_n * step_size * 10_000): 1.0}

    # Linear interpolation
    p_upper = change_in_steps - lower_n
    p_lower = 1.0 - p_upper

    probs = {}
    if abs(p_lower) > 1e-6:
        probs[int(lower_n * step_size * 10_000)] = p_lower
    if abs(p_upper) > 1e-6:
        probs[int(upper_n * step_size * 10_000)] = p_upper

    return probs


def fomc_prob_summary(
    implied_rate_after: float,
    current_rate: float,
    step_size: float = 0.0025,
) -> dict[str, float]:
    """
    Aggregate distribution into P(hike), P(hold), P(cut).
    """
    dist = fedwatch_probabilities(implied_rate_after, current_rate, step_size)
    p_hike = sum(p for move, p in dist.items() if move > 0)
    p_cut  = sum(p for move, p in dist.items() if move < 0)
    p_hold = dist.get(0, 0.0)
    return {"p_hike": p_hike, "p_hold": p_hold, "p_cut": p_cut}


# ── Statistical FOMC probability model ───────────────────────────────────────

def build_fomc_features(
    macro_df: pd.DataFrame,
    fomc_dates: list[date],
) -> pd.DataFrame:
    """
    Build a feature matrix for statistical FOMC prediction.
    Aligns macro data to FOMC meeting dates (using most recent prior data).

    Features:
        - taylor_gap : actual rate - Taylor Rule implied rate
        - core_pce_3m: 3-month annualized core PCE (momentum)
        - unemp_gap  : unemployment - NAIRU
        - vix        : VIX level (financial conditions)
        - slope_2s10s: 2s10s slope (market expectation)
        - sofr_effr_spread: SOFR-EFFR basis (funding stress)
    """
    feature_cols = [
        "taylor_gap", "core_pce_3m_ann", "unemployment_gap",
        "vix", "slope_2s10s", "sofr_effr_spread",
    ]
    available = [c for c in feature_cols if c in macro_df.columns]

    rows = []
    for fomc_date in fomc_dates:
        prior_data = macro_df.loc[:fomc_date.strftime("%Y-%m-%d")]
        if prior_data.empty:
            continue
        row = prior_data[available].iloc[-1].to_dict()
        row["fomc_date"] = fomc_date
        rows.append(row)

    return pd.DataFrame(rows).set_index("fomc_date")


def fit_fomc_logit(
    features: pd.DataFrame,
    outcomes: pd.Series,
    outcome_type: Literal["hike", "cut", "change"] = "hike",
) -> object:
    """
    Fit a logistic regression model to predict FOMC outcomes.

    Parameters
    ----------
    features    : DataFrame of feature values at each FOMC date
    outcomes    : Binary Series: 1 = outcome occurred, 0 = did not
    outcome_type: what we're predicting (for labeling)

    Returns
    -------
    Fitted sklearn LogisticRegression model
    """
    from sklearn.linear_model import LogisticRegression
    from sklearn.preprocessing import StandardScaler
    from sklearn.pipeline import Pipeline

    valid_idx = features.index.intersection(outcomes.index)
    X = features.loc[valid_idx].dropna()
    y = outcomes.loc[X.index]

    pipe = Pipeline([
        ("scaler", StandardScaler()),
        ("logit",  LogisticRegression(C=1.0, random_state=42)),
    ])
    pipe.fit(X, y)
    return pipe


def extract_cumulative_cuts(
    fomc_dates: list[date],
    implied_rates: pd.Series,
    current_rate: float,
    step_size: float = 0.0025,
) -> pd.DataFrame:
    """
    Build a table of cumulative expected cuts/hikes at each FOMC meeting.
    implied_rates: Series indexed by date with implied post-meeting SOFR rate.
    """
    rows = []
    for d in fomc_dates:
        if d not in implied_rates.index:
            continue
        impl = implied_rates[d]
        total_move_bps = (impl - current_rate) * 10_000
        probs = fomc_prob_summary(impl, current_rate, step_size)
        rows.append({
            "fomc_date":            d,
            "implied_rate_pct":     impl * 100,
            "cumulative_move_bps":  total_move_bps,
            **probs,
        })
    return pd.DataFrame(rows).set_index("fomc_date")


if __name__ == "__main__":
    # Demo: FedWatch-style probability extraction
    current_rate = 0.0530   # 5.30% (current upper bound)
    implied_rate = 0.0505   # 5.05% implied by futures → ~1 cut priced in

    summary = fomc_prob_summary(implied_rate, current_rate)
    print("FOMC Meeting Probabilities:")
    print(f"  P(25bp cut) = {summary['p_cut']:.1%}")
    print(f"  P(hold)     = {summary['p_hold']:.1%}")
    print(f"  P(25bp hike)= {summary['p_hike']:.1%}")

    print("\nNext FOMC dates:")
    for d in next_fomc(date.today(), n=4):
        print(f"  {d}")
