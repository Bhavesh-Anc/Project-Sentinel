"""
Day count conventions used in US rates markets.

SOFR instruments:  ACT/360  (ISDA standard)
US Treasuries:     ACT/ACT  (ICMA)
Fed Funds OIS:     ACT/360
Fixed leg (swaps): 30/360 or ACT/360 depending on convention
"""
from datetime import date
import numpy as np


def act360(start: date, end: date) -> float:
    """ACT/360: actual calendar days / 360. Standard for SOFR, Fed Funds, LIBOR."""
    return (end - start).days / 360.0


def act365(start: date, end: date) -> float:
    """ACT/365 Fixed: actual calendar days / 365. Used in some GBP markets."""
    return (end - start).days / 365.0


def act_act_isma(start: date, end: date, coupon_freq: int = 2) -> float:
    """
    ACT/ACT (ICMA/ISMA): used for US Treasury bonds.
    Year fraction = actual days / (frequency × days in coupon period).
    For zero-coupon purposes, approximated as ACT/365.25.
    """
    return (end - start).days / 365.25


def thirty_360(start: date, end: date) -> float:
    """
    30/360 (Bond Basis): used for fixed leg of many USD swap conventions.
    Formula per ISDA: days = 360*(Y2-Y1) + 30*(M2-M1) + (D2-D1)
    where D1 = min(D1, 30), D2 = min(D2, 30) if D1 == 30 or 31.
    """
    y1, m1, d1 = start.year, start.month, start.day
    y2, m2, d2 = end.year, end.month, end.day

    if d1 == 31:
        d1 = 30
    if d2 == 31 and d1 == 30:
        d2 = 30

    return (360 * (y2 - y1) + 30 * (m2 - m1) + (d2 - d1)) / 360.0


def year_frac(start: date, end: date, convention: str = "ACT360") -> float:
    """
    Dispatch year fraction calculation by convention name.
    Supported: 'ACT360', 'ACT365', 'ACT_ACT', '30_360'
    """
    convention = convention.upper().replace("/", "").replace("-", "")
    dispatch = {
        "ACT360":   act360,
        "ACT365":   act365,
        "ACTACT":   act_act_isma,
        "30360":    thirty_360,
    }
    if convention not in dispatch:
        raise ValueError(f"Unknown day count convention: {convention}")
    return dispatch[convention](start, end)


def compound_sofr_fixings(
    fixings: list[float],
    day_counts: list[int],
    day_count_basis: int = 360,
) -> float:
    """
    Compound a sequence of overnight SOFR fixings over a period.

    Parameters
    ----------
    fixings      : daily SOFR rates as decimals (e.g. [0.0530, 0.0530, ...])
    day_counts   : number of calendar days each fixing applies for
                   (1 for Mon–Thu; 3 for Fri covering Fri/Sat/Sun)
    day_count_basis : 360 (SOFR standard)

    Returns
    -------
    Compounded annualized rate for the period (decimal)
    """
    if len(fixings) != len(day_counts):
        raise ValueError("fixings and day_counts must have equal length")

    factor = 1.0
    total_days = 0
    for r, d in zip(fixings, day_counts):
        factor *= (1.0 + r * d / day_count_basis)
        total_days += d

    return (factor - 1.0) * day_count_basis / total_days


def bday_weights(dates: list[date]) -> list[int]:
    """
    Compute day-count weights for a series of business days.
    Friday gets weight 3 (covers Fri/Sat/Sun), Mon–Thu get weight 1.
    Holidays are not handled here — use a calendar library if needed.
    """
    weights = []
    for i, d in enumerate(dates[:-1]):
        gap = (dates[i + 1] - d).days
        weights.append(gap)
    return weights
