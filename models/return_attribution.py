"""
Fixed-Income Return Attribution — Campisi (2000) / GRAP Framework.

Decomposes the realized P&L of a rate/swap position into five orthogonal
components so that portfolio managers can understand *why* a position earned
(or lost) money over a holding period.

Framework
---------
Total Return = Carry + Roll-Down + Duration Effect + Convexity Effect + Residual

Components
----------
Carry        : net coupon income minus financing cost over the period.
               Income side assumes the bond earns its par rate; financing cost
               is the overnight SOFR (or a user-supplied rate).

Roll-Down    : price appreciation from the bond aging along the curve.
               As time passes the bond has a shorter residual maturity; if the
               curve is upward-sloping that shorter point has a lower yield,
               so the price rises ("ride the yield curve").

Duration     : first-order sensitivity to a parallel shift in yield levels.
               −ModDur × Δy (in bps of price return for a 1-decimal yield move).

Convexity    : second-order Taylor correction.
               ½ × Convexity × Δy²  (always non-negative, benefits the holder
               regardless of direction of rate move).

Residual     : unexplained P&L — curve-shape twist, basis, model error.
               Residual = total_actual − total_approx.

The "actual" total return uses exact repricing from the end curve so that
the five components always sum to the observed P&L by construction.

Reference
---------
Campisi, S. (2000). "Primer on Fixed Income Performance Attribution."
Journal of Performance Measurement, 4(4), 14–25.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from dataclasses import dataclass, field
from datetime import date
from typing import Sequence

from sofr_engine.curve import DiscountCurve


# ── Bond analytics helpers ────────────────────────────────────────────────────

def _mod_duration(yield_decimal: float, tenor: float, freq: int = 2) -> float:
    """
    Modified duration of a par bond with semi-annual (default) coupons.

    Closed-form for a par bond (coupon rate = yield):
      MacD = (1 + y/m) / y × (1 − 1/(1 + y/m)^n)
      ModD = MacD / (1 + y/m) = (1/y) × (1 − 1/(1 + y/m)^n)

    where n = tenor × freq is the number of coupon periods.

    Parameters
    ----------
    yield_decimal : semi-annual par yield as a decimal (e.g. 0.05 for 5%)
    tenor         : maturity in years
    freq          : coupon frequency per year (2 = semi-annual)

    Returns
    -------
    Modified duration in years.
    """
    if tenor <= 0:
        return 0.0
    y = yield_decimal
    m = freq
    n = tenor * m

    if y < 1e-8:
        return tenor * (1 - 1.0 / (n + 1))  # limit as y→0

    ym = y / m
    discount_factor_n = (1 + ym) ** (-n)

    mod_duration = (1.0 / y) * (1.0 - discount_factor_n)
    return mod_duration


def _convexity(yield_decimal: float, tenor: float, freq: int = 2) -> float:
    """
    Convexity of a par bond using a numerical bump approach.

    C = (P+ + P- − 2 × P) / (P × dy²)

    The coupon is fixed at yield_decimal (par bond at inception, price = 1).
    The bump shifts only the discount yield, not the coupon.

    Parameters
    ----------
    yield_decimal : par yield as a decimal (coupon rate of the par bond)
    tenor         : maturity in years
    freq          : coupon frequency per year

    Returns
    -------
    Convexity in years² (positive for a standard bond).
    """
    if tenor <= 0:
        return 0.0

    dy = 1e-4  # 1bp bump
    coupon = yield_decimal

    def _price(discount_y: float) -> float:
        m = freq
        n_periods = max(1, int(round(tenor * m)))
        c = coupon / m
        price = sum(c / (1 + discount_y / m) ** k for k in range(1, n_periods + 1))
        price += 1.0 / (1 + discount_y / m) ** n_periods
        return price

    y = yield_decimal
    p_mid = _price(y)
    p_up  = _price(y + dy)
    p_dn  = _price(y - dy)

    return (p_up + p_dn - 2 * p_mid) / (p_mid * dy ** 2)


def _par_bond_clean_price(
    coupon: float, discount_yield: float, tenor: float, freq: int = 2
) -> float:
    """
    Clean price of a fixed-coupon bond at a given discount yield.

    The bond is assumed to be priced exactly at a coupon date boundary
    (no accrued).  This is the standard flat-yield pricing formula.

    Parameters
    ----------
    coupon         : annual coupon rate (fixed at inception par yield)
    discount_yield : semi-annual yield to maturity (decimal) for discounting
    tenor          : remaining maturity in years
    freq           : coupon frequency per year (default 2 = semi-annual)

    Returns
    -------
    Clean price per unit face value (= 1.0 when coupon == discount_yield).
    """
    m = freq
    n_periods = max(1, int(round(tenor * m)))
    c_per = coupon / m
    ym = discount_yield / m

    price = sum(c_per / (1 + ym) ** k for k in range(1, n_periods + 1))
    price += 1.0 / (1 + ym) ** n_periods
    return price


# ── Core attribution dataclass ────────────────────────────────────────────────

@dataclass
class AttributionResult:
    """
    Full Campisi attribution for a single holding period.

    All monetary components are expressed in basis points of price return
    (i.e. bps × 10,000 × P₀ gives dollar P&L per unit of face value).
    """
    tenor_years:       float
    dt_years:          float
    yield_start_pct:   float   # par yield at start, in percent
    yield_end_pct:     float   # par yield at end (aged tenor), in percent
    delta_y_bps:       float   # actual yield change (end minus start) in bps

    carry_bps:         float   # net coupon − financing over dt
    rolldown_bps:      float   # P&L from aging along the curve
    duration_bps:      float   # −ModDur × delta_y
    convexity_bps:     float   # ½ × Convexity × delta_y² (in bps of price)
    total_approx_bps:  float   # carry + rolldown + duration + convexity
    total_actual_bps:  float   # exact repricing P&L
    residual_bps:      float   # total_actual − total_approx

    modified_duration: float   # modified duration in years
    convexity_years2:  float   # convexity in years²


# ── Single-period attribution ─────────────────────────────────────────────────

def attribute_single_period(
    curve_start: DiscountCurve,
    curve_end: DiscountCurve,
    tenor_years: float,
    dt_years: float = 1 / 252,
    financing_rate: float | None = None,
) -> AttributionResult:
    """
    Attribute the P&L of holding a par bond at tenor_years for dt_years.

    The position is a receiver (long duration): we hold a par bond worth 1.0
    at the start of the period.  After dt_years the bond has aged to residual
    maturity (tenor_years − dt_years) and is repriced on curve_end.

    Actual total return (exact repricing)
    --------------------------------------
    new_price = curve_end.df(T − dt) / curve_start.df(T)
    total_actual_bps = (new_price − 1 + carry_income_decimal) × 10_000

    Approximated total return (Taylor expansion)
    --------------------------------------------
    total_approx_bps = carry_bps + rolldown_bps + duration_bps + convexity_bps

    Residual
    --------
    residual_bps = total_actual_bps − total_approx_bps

    Parameters
    ----------
    curve_start     : discount curve at start of period
    curve_end       : discount curve at end of period
    tenor_years     : initial tenor of the par bond in years
    dt_years        : holding period length in years (default 1 business day)
    financing_rate  : overnight financing rate (decimal); defaults to the
                      overnight SOFR read from curve_start

    Returns
    -------
    AttributionResult with all components populated.
    """
    t_start = tenor_years
    t_end   = max(t_start - dt_years, 1e-6)

    # ── Financing rate ────────────────────────────────────────────────────────
    if financing_rate is None:
        financing_rate = curve_start.zero_rate(1 / 252)

    # ── Yields ───────────────────────────────────────────────────────────────
    # Semi-annual par rates are used throughout for consistency with the
    # semi-annual coupon bond pricing in _mod_duration, _convexity, and
    # _reprice_par_bond_on_curve (all using freq=2).
    # Yield change is measured at the original tenor on both curves to isolate
    # the level-shift component cleanly.
    y_start   = curve_start.par_ois_rate(t_start, payment_freq=2)  # semi-annual par
    y_end_lvl = curve_end.par_ois_rate(t_start,   payment_freq=2)  # end curve, same tenor

    yield_start_pct = y_start   * 100
    yield_end_pct   = y_end_lvl * 100
    delta_y_bps     = (y_end_lvl - y_start) * 10_000  # parallel shift at tenor T

    # ── Carry component ───────────────────────────────────────────────────────
    carry_income_decimal = (y_start - financing_rate) * dt_years
    carry = carry_income_decimal * 10_000            # bps

    # ── Bond analytics ───────────────────────────────────────────────────────
    mod_dur  = _mod_duration(y_start, t_start)
    conv     = _convexity(y_start, t_start)

    # ── Roll-down component ───────────────────────────────────────────────────
    # Roll-down measures the P&L from the bond aging along the START curve only.
    # Using zero rates (smooth, no boundary issues): the bond's yield changes
    # from zero_rate(T) to zero_rate(T−dt) on the same curve.
    # Positive when the curve is upward-sloping (yield falls as tenor shortens).
    z_start        = curve_start.zero_rate(t_start)
    z_aged         = curve_start.zero_rate(t_end)
    yield_roll_chg = z_aged - z_start              # change in zero rate from aging
    rolldown = -mod_dur * yield_roll_chg * 10_000  # bps

    dy_decimal    = (y_end_lvl - y_start)              # yield change in decimal
    duration_bps  = -mod_dur * dy_decimal * 10_000     # bps
    convexity_bps = 0.5 * conv * dy_decimal ** 2 * 10_000  # bps

    # ── Actual total return (exact repricing of the par bond) ────────────────
    # The par bond (coupon = y_start) is repriced at the semi-annual par yield
    # given by curve_end at the original tenor T (avoids boundary issues at
    # non-integer tenors).  The aged tenor (T−dt) is close to T so this is
    # a very small approximation.
    # total_actual = (new_clean_price − 1.0 + net_carry) × 10000
    new_price        = _par_bond_clean_price(y_start, y_end_lvl, t_start)
    total_actual_bps = (new_price - 1.0 + carry_income_decimal) * 10_000

    # ── Taylor approximation total ────────────────────────────────────────────
    total_approx_bps = carry + rolldown + duration_bps + convexity_bps

    # ── Residual ──────────────────────────────────────────────────────────────
    residual_bps = total_actual_bps - total_approx_bps

    return AttributionResult(
        tenor_years       = tenor_years,
        dt_years          = dt_years,
        yield_start_pct   = yield_start_pct,
        yield_end_pct     = yield_end_pct,
        delta_y_bps       = delta_y_bps,
        carry_bps         = carry,
        rolldown_bps      = rolldown,
        duration_bps      = duration_bps,
        convexity_bps     = convexity_bps,
        total_approx_bps  = total_approx_bps,
        total_actual_bps  = total_actual_bps,
        residual_bps      = residual_bps,
        modified_duration = mod_dur,
        convexity_years2  = conv,
    )


# ── History attribution ───────────────────────────────────────────────────────

def attribute_history(
    curve_series: list[tuple[date, DiscountCurve]],
    tenor_years: float,
    financing_col: float | None = None,
) -> pd.DataFrame:
    """
    Attribute returns over a sequence of (date, DiscountCurve) pairs.

    Each consecutive pair of curves forms one holding period.  The same fixed
    tenor_years is used for every period (constant-maturity analysis — the
    position is conceptually rolled back to its original tenor each day).

    Parameters
    ----------
    curve_series    : list of (date, DiscountCurve) in chronological order;
                      must contain at least two elements
    tenor_years     : fixed tenor for the constant-maturity position
    financing_col   : constant financing rate override (decimal); None = use
                      overnight SOFR from each start curve

    Returns
    -------
    DataFrame with one row per period and columns:
        date, tenor_years, dt_years, yield_start_pct, yield_end_pct,
        delta_y_bps, carry_bps, rolldown_bps, duration_bps, convexity_bps,
        total_approx_bps, total_actual_bps, residual_bps,
        modified_duration, convexity_years2,
        cumulative_carry_bps, cumulative_rolldown_bps, cumulative_duration_bps,
        cumulative_convexity_bps, cumulative_total_actual_bps
    """
    if len(curve_series) < 2:
        raise ValueError("curve_series must have at least two (date, curve) entries")

    rows: list[dict] = []

    for i in range(len(curve_series) - 1):
        date_start, c_start = curve_series[i]
        date_end,   c_end   = curve_series[i + 1]

        dt_days  = (date_end - date_start).days
        dt_years = dt_days / 365.25

        result = attribute_single_period(
            curve_start    = c_start,
            curve_end      = c_end,
            tenor_years    = tenor_years,
            dt_years       = dt_years,
            financing_rate = financing_col,
        )

        row = {
            "date":               date_end,
            "tenor_years":        result.tenor_years,
            "dt_years":           result.dt_years,
            "yield_start_pct":    result.yield_start_pct,
            "yield_end_pct":      result.yield_end_pct,
            "delta_y_bps":        result.delta_y_bps,
            "carry_bps":          result.carry_bps,
            "rolldown_bps":       result.rolldown_bps,
            "duration_bps":       result.duration_bps,
            "convexity_bps":      result.convexity_bps,
            "total_approx_bps":   result.total_approx_bps,
            "total_actual_bps":   result.total_actual_bps,
            "residual_bps":       result.residual_bps,
            "modified_duration":  result.modified_duration,
            "convexity_years2":   result.convexity_years2,
        }
        rows.append(row)

    df = pd.DataFrame(rows).set_index("date")

    cumulative_cols = [
        "carry_bps", "rolldown_bps", "duration_bps",
        "convexity_bps", "total_actual_bps",
    ]
    for col in cumulative_cols:
        df[f"cumulative_{col}"] = df[col].cumsum()

    return df


# ── Portfolio attribution ─────────────────────────────────────────────────────

@dataclass
class PortfolioAttribution:
    """
    Attribution for a multi-position portfolio of rate/swap positions.

    Each position is described by a dict with keys:
        tenor_years  : float — maturity of the position
        notional     : float — face value (default 1.0)
        dv01_sign    : +1 for receiver (long duration), −1 for payer (short)
        label        : str  — optional display label

    DV01 sign convention
    --------------------
    +1 (receiver) : receives fixed, pays floating.  Benefits when rates fall.
                    Duration contribution is positive; net duration_bps = +|D×Δy|.
    −1 (payer)    : pays fixed, receives floating.  Benefits when rates rise.
                    Flips the sign of all price return components.
    """
    positions: list[dict]

    def attribute_single_period(
        self,
        curve_start: DiscountCurve,
        curve_end: DiscountCurve,
        dt_years: float = 1 / 252,
        financing_rate: float | None = None,
    ) -> pd.DataFrame:
        """
        Attribution for each position and total portfolio.

        Returns
        -------
        DataFrame with index = position labels + 'Total', columns = all
        AttributionResult fields (in bps, scaled by notional and dv01_sign).
        """
        rows: list[dict] = []

        for pos in self.positions:
            t         = pos["tenor_years"]
            notional  = pos.get("notional", 1.0)
            dv01_sign = pos.get("dv01_sign", 1)
            label     = pos.get("label", f"{t}Y {'recv' if dv01_sign > 0 else 'pay'}")

            result = attribute_single_period(
                curve_start    = curve_start,
                curve_end      = curve_end,
                tenor_years    = t,
                dt_years       = dt_years,
                financing_rate = financing_rate,
            )

            sign = float(dv01_sign)
            row = {
                "label":             label,
                "tenor_years":       t,
                "notional":          notional,
                "dv01_sign":         dv01_sign,
                "yield_start_pct":   result.yield_start_pct,
                "yield_end_pct":     result.yield_end_pct,
                "delta_y_bps":       result.delta_y_bps,
                "carry_bps":         sign * result.carry_bps         * notional,
                "rolldown_bps":      sign * result.rolldown_bps      * notional,
                "duration_bps":      sign * result.duration_bps      * notional,
                "convexity_bps":     sign * result.convexity_bps     * notional,
                "total_approx_bps":  sign * result.total_approx_bps  * notional,
                "total_actual_bps":  sign * result.total_actual_bps  * notional,
                "residual_bps":      sign * result.residual_bps      * notional,
                "modified_duration": result.modified_duration,
                "convexity_years2":  result.convexity_years2,
            }
            rows.append(row)

        df = pd.DataFrame(rows).set_index("label")

        bps_cols = [
            "carry_bps", "rolldown_bps", "duration_bps", "convexity_bps",
            "total_approx_bps", "total_actual_bps", "residual_bps",
        ]
        total_row = {col: df[col].sum() for col in bps_cols}
        total_row.update({
            "tenor_years":       float("nan"),
            "notional":          df["notional"].sum(),
            "dv01_sign":         float("nan"),
            "yield_start_pct":   float("nan"),
            "yield_end_pct":     float("nan"),
            "delta_y_bps":       float("nan"),
            "modified_duration": float("nan"),
            "convexity_years2":  float("nan"),
        })
        total_df = pd.DataFrame([total_row], index=["Total"])
        return pd.concat([df, total_df])

    def summary(
        self,
        curve_start: DiscountCurve,
        curve_end: DiscountCurve,
        dt_years: float = 1 / 252,
    ) -> dict:
        """
        Net P&L summary and dominant attribution component.

        Returns
        -------
        Dict with keys:
            net_pnl_bps, carry_bps, rolldown_bps, duration_bps, convexity_bps,
            residual_bps, dominant_component
        """
        attr_df = self.attribute_single_period(curve_start, curve_end, dt_years)
        total   = attr_df.loc["Total"]

        components = {
            "carry_bps":     float(total["carry_bps"]),
            "rolldown_bps":  float(total["rolldown_bps"]),
            "duration_bps":  float(total["duration_bps"]),
            "convexity_bps": float(total["convexity_bps"]),
            "residual_bps":  float(total["residual_bps"]),
        }
        dominant = max(components, key=lambda k: abs(components[k]))

        return {
            "net_pnl_bps":        float(total["total_actual_bps"]),
            **components,
            "dominant_component": dominant,
        }


# ── Steepener attribution ─────────────────────────────────────────────────────

def steepener_attribution(
    curve_start: DiscountCurve,
    curve_end: DiscountCurve,
    short_tenor: float = 2.0,
    long_tenor: float = 10.0,
    dt_years: float = 1 / 252,
) -> dict:
    """
    Full attribution for a DV01-neutral 2s10s steepener.

    Position
    --------
    Receive fixed short_tenor (long duration, dv01_sign = +1).
    Pay fixed long_tenor   (short duration, dv01_sign = −1).

    The long leg notional is set to 1.0; the short leg notional is scaled so
    that the two legs have equal and opposite DV01:

        N_short × DV01(short_tenor) = N_long × DV01(long_tenor)
        N_short = DV01(long_tenor) / DV01(short_tenor)

    DV01 is approximated as modified_duration × price × notional / 10_000,
    where price = 1.0 (par bond at inception), so N_short = dur_long / dur_short.

    Parameters
    ----------
    curve_start  : discount curve at start of period
    curve_end    : discount curve at end of period
    short_tenor  : short leg tenor (default 2Y)
    long_tenor   : long leg tenor (default 10Y)
    dt_years     : holding period in years

    Returns
    -------
    Dict with keys:
        short_leg, long_leg          — AttributionResult for each leg
        net                          — PortfolioAttribution.summary() dict
        dv01_ratio                   — N_short / N_long
        net_carry_bps, net_rolldown_bps, net_duration_bps,
        net_convexity_bps, net_total_bps
    """
    y_short = curve_start.par_ois_rate(short_tenor)
    y_long  = curve_start.par_ois_rate(long_tenor)

    dur_short = _mod_duration(y_short, short_tenor)
    dur_long  = _mod_duration(y_long,  long_tenor)

    dv01_ratio = dur_long / dur_short if dur_short > 1e-8 else 1.0

    short_result = attribute_single_period(
        curve_start, curve_end, short_tenor, dt_years
    )
    long_result = attribute_single_period(
        curve_start, curve_end, long_tenor, dt_years
    )

    portfolio = PortfolioAttribution(positions=[
        {"tenor_years": short_tenor, "notional": dv01_ratio, "dv01_sign": +1,
         "label": f"recv_{short_tenor}Y"},
        {"tenor_years": long_tenor,  "notional": 1.0,        "dv01_sign": -1,
         "label": f"pay_{long_tenor}Y"},
    ])

    net = portfolio.summary(curve_start, curve_end, dt_years)

    return {
        "short_leg":          short_result,
        "long_leg":           long_result,
        "net":                net,
        "dv01_ratio":         dv01_ratio,
        "net_carry_bps":      net["carry_bps"],
        "net_rolldown_bps":   net["rolldown_bps"],
        "net_duration_bps":   net["duration_bps"],
        "net_convexity_bps":  net["convexity_bps"],
        "net_total_bps":      net["net_pnl_bps"],
    }
