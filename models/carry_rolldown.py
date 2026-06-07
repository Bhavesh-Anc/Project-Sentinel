"""
Carry and Roll-Down analytics for SOFR / Treasury positions.

Definitions
-----------
Carry       : income earned from holding a fixed-income position for dt,
              assuming no change in the yield curve (coupon income - financing cost)

Roll-Down   : P&L from the bond "aging" (rolling) along the yield curve.
              A 10Y bond becomes a 9Y11M bond after one month.
              If the curve is upward-sloping, the rolled bond has a *lower* yield
              → positive price return = positive roll-down.

Total Return = Carry + Roll-Down  (in a static curve environment)
             = the "pull to par" + "ride the curve" components

This is the core concept behind carry-arbitrage strategies. A steepener position
with positive carry + roll-down can survive modest adverse yield moves.

Functions
---------
carry_bps             : daily/annual carry for a zero-coupon position
rolldown_bps          : roll-down over a horizon for a given tenor
carry_rolldown_table  : full table across tenors
breakeven_yield_move  : maximum adverse yield move before carry is wiped out
carry_rolldown_matrix : 2D grid: tenor × horizon
"""
from __future__ import annotations
import numpy as np
import pandas as pd
from typing import Sequence

from sofr_engine.curve import DiscountCurve


# ── Single-point analytics ────────────────────────────────────────────────────

def carry_bps(
    curve: DiscountCurve,
    tenor_years: float,
    dt_years: float = 1 / 252,
    financing_rate: float | None = None,
) -> float:
    """
    Carry (in bps) for holding a zero-coupon bond of tenor_years for dt_years.

    Carry = (coupon/par rate at tenor) - (financing rate)
    For SOFR OIS: financing rate = overnight SOFR (par rate at near zero tenor)

    Parameters
    ----------
    curve           : discount curve for yield computation
    tenor_years     : bond/swap tenor
    dt_years        : holding period (default 1 business day)
    financing_rate  : overnight financing cost; defaults to curve.zero_rate(1/252)

    Returns
    -------
    Carry in bps per period dt_years.
    """
    par_rate   = curve.par_ois_rate(tenor_years)                 # yield on bond
    fin_rate   = financing_rate if financing_rate is not None else curve.zero_rate(1 / 252)
    carry_ann  = (par_rate - fin_rate) * 100 * 100               # convert to bps/year
    return carry_ann * dt_years                                   # bps over dt


def rolldown_bps(
    curve: DiscountCurve,
    tenor_years: float,
    dt_years: float = 1 / 12,
    duration_approx: bool = True,
) -> float:
    """
    Roll-down P&L (in bps) from aging from tenor_years to (tenor_years - dt_years).

    As a bond ages by dt, its residual maturity shrinks. If the curve slopes up,
    the new (shorter) yield is lower → positive price appreciation.

    Roll-down ≈ -duration × (y(T - dt) - y(T))   [via dP/P ≈ -D × dy]

    Parameters
    ----------
    curve         : current discount curve
    tenor_years   : starting tenor
    dt_years      : aging period (default 1 month)
    duration_approx : use modified duration approximation (True) or exact (False)

    Returns
    -------
    Roll-down in bps (positive = gain from aging along upward-sloping curve).
    """
    t_start = tenor_years
    t_end   = max(tenor_years - dt_years, 1e-6)

    # Use zero rates (continuously compounded) — they are smooth across the entire curve
    # and avoid the par-rate instability near grid anchor points.
    y_start = curve.zero_rate(t_start) * 100   # zero yield at start tenor (pct)
    y_end   = curve.zero_rate(t_end)   * 100   # zero yield at aged tenor (pct)

    yield_pickup = y_start - y_end   # positive if curve is upward-sloping (yield falls as we age)

    if duration_approx:
        # Modified duration for a zero-coupon bond: duration = T (Macaulay = modified for c=0)
        # For a par bond approximation: duration ≈ (1 - e^{-y*T}) / y  (continuous coupon)
        y = y_start / 100
        if y > 1e-8:
            duration = (1 - np.exp(-y * t_start)) / y
        else:
            duration = t_start
        return duration * yield_pickup * 100  # in bps

    # Exact: price change = DF(t_end) at old vs new yield
    p_old = 1.0            # par bond at t_start
    p_new = curve.df(t_end) / curve.df(t_start)   # aged bond PV ratio (approx)
    return (p_new - p_old) * 10_000                # bps


def total_return_bps(
    curve: DiscountCurve,
    tenor_years: float,
    dt_years: float = 1 / 12,
    financing_rate: float | None = None,
) -> dict[str, float]:
    """
    Total return breakdown: carry + roll-down for a given tenor and horizon.
    """
    c  = carry_bps(curve, tenor_years, dt_years, financing_rate)
    rd = rolldown_bps(curve, tenor_years, dt_years)
    return {
        "tenor_years":    tenor_years,
        "carry_bps":      c,
        "rolldown_bps":   rd,
        "total_return_bps": c + rd,
        "dt_years":       dt_years,
    }


# ── Full carry/roll-down table ────────────────────────────────────────────────

STANDARD_TENORS = [0.5, 1.0, 2.0, 3.0, 5.0, 7.0, 10.0, 15.0, 20.0, 30.0]


def carry_rolldown_table(
    curve: DiscountCurve,
    tenors: Sequence[float] = STANDARD_TENORS,
    dt_years: float = 1 / 12,
    financing_rate: float | None = None,
) -> pd.DataFrame:
    """
    Carry + roll-down table across standard tenors.

    Returns DataFrame with columns:
      tenor_yrs, yield_pct, par_rate_pct, carry_bps, rolldown_bps,
      total_return_bps, breakeven_move_bps, modified_duration
    """
    fin_rate = financing_rate if financing_rate is not None else curve.zero_rate(1 / 252)
    rows     = []
    for t in tenors:
        if t >= curve._times[-1] - 0.5:
            continue
        par_r = curve.par_ois_rate(t)
        zero  = curve.zero_rate(t)
        c     = carry_bps(curve, t, dt_years, fin_rate)
        rd    = rolldown_bps(curve, t, dt_years)
        total = c + rd

        # Breakeven yield move: how many bps can rates rise before losing total return?
        y  = par_r
        dur = (1 - np.exp(-y * t)) / y if y > 1e-8 else t
        be  = total / (dur * 100) if dur > 0 else 0.0  # in bps

        rows.append({
            "tenor_yrs":          t,
            "zero_rate_pct":      zero * 100,
            "par_rate_pct":       par_r * 100,
            "carry_bps":          c,
            "rolldown_bps":       rd,
            "total_return_bps":   total,
            "breakeven_move_bps": be,
            "modified_duration":  dur,
        })
    return pd.DataFrame(rows).set_index("tenor_yrs").round(4)


# ── Breakeven yield move ──────────────────────────────────────────────────────

def breakeven_yield_move(
    curve: DiscountCurve,
    tenor_years: float,
    dt_years: float = 1 / 12,
) -> float:
    """
    Maximum adverse yield move (in bps) that carry+rolldown can absorb before
    total return turns negative.

    breakeven = (carry + rolldown) / modified_duration
    """
    tot = total_return_bps(curve, tenor_years, dt_years)["total_return_bps"]
    par_r = curve.par_ois_rate(tenor_years)
    y     = par_r if par_r > 1e-8 else 0.03
    dur   = (1 - np.exp(-y * tenor_years)) / y
    return tot / (dur * 100) if dur > 0 else 0.0


# ── 2D matrix: tenor × horizon ────────────────────────────────────────────────

def carry_rolldown_matrix(
    curve: DiscountCurve,
    tenors:   Sequence[float] = [2.0, 5.0, 10.0, 30.0],
    horizons: Sequence[float] = [1/52, 1/12, 3/12, 6/12, 1.0],
    component: str = "total_return_bps",
) -> pd.DataFrame:
    """
    2D matrix of carry+rolldown across tenors (rows) and holding horizons (cols).

    Parameters
    ----------
    tenors    : list of bond tenors in years
    horizons  : list of holding periods in years (e.g. 1/52 = 1 week)
    component : 'carry_bps', 'rolldown_bps', or 'total_return_bps'
    """
    horizon_labels = {1/52: "1W", 1/12: "1M", 3/12: "3M", 6/12: "6M", 1.0: "1Y"}
    data = {}
    for h in horizons:
        col  = horizon_labels.get(h, f"{h*12:.0f}M")
        vals = []
        for t in tenors:
            if t < curve._times[-1]:
                tr  = total_return_bps(curve, t, h)
                vals.append(tr[component])
            else:
                vals.append(np.nan)
        data[col] = vals

    df = pd.DataFrame(data, index=[f"{t}Y" for t in tenors])
    df.index.name = "tenor"
    return df.round(2)


# ── Steepener carry decomposition ─────────────────────────────────────────────

def steepener_carry(
    curve: DiscountCurve,
    short_tenor: float = 2.0,
    long_tenor: float = 10.0,
    dt_years: float = 1 / 252,
) -> dict[str, float]:
    """
    Net carry and roll-down for a DV01-neutral 2s10s steepener.

    The steepener pays fixed 10Y and receives fixed 2Y.
    Net carry = carry(2Y leg received) + carry(10Y leg paid, negative sign)

    Returns dict with:
      carry_2y_bps, carry_10y_bps, net_carry_bps,
      rolldown_2y_bps, rolldown_10y_bps, net_rolldown_bps,
      net_total_bps
    """
    fin = curve.zero_rate(1 / 252)
    c2  = carry_bps(curve, short_tenor, dt_years, fin)   # receive fixed 2Y
    c10 = carry_bps(curve, long_tenor,  dt_years, fin)   # pay fixed 10Y (negative)
    r2  = rolldown_bps(curve, short_tenor, dt_years)
    r10 = rolldown_bps(curve, long_tenor,  dt_years)

    # DV01-neutral: we receive 2Y (long 2Y) and pay 10Y (short 10Y)
    # Net carry = receive 2Y coupon - pay 10Y coupon
    # Net rolldown: 2Y rolls to shorter, 10Y rolls to shorter (opposite sign since short 10Y)
    net_carry    = c2 - c10
    net_rolldown = r2 - r10   # short 10Y means we don't benefit from 10Y rolldown
    return {
        "carry_2y_bps":      c2,
        "carry_10y_bps":     c10,
        "net_carry_bps":     net_carry,
        "rolldown_2y_bps":   r2,
        "rolldown_10y_bps":  r10,
        "net_rolldown_bps":  net_rolldown,
        "net_total_bps":     net_carry + net_rolldown,
    }
