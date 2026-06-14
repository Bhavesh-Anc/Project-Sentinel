"""
Jarrow-Yildirim Inflation Model — simplified implementation.

Provides:
  - InflationCurve: piecewise-flat forward CPI growth rates
  - ZCInflationSwap / YoYInflationSwap / InflationCapFloor: product data classes
  - Pricing functions: zc_inflation_pv, yoy_inflation_pv, inflation_cap_floor_pv
  - Utilities: breakeven_inflation, calibrate_inflation_curve

References:
  Jarrow, R. & Yildirim, Y. (2003). Pricing Treasury Inflation Protected
  Securities and Related Derivatives using an HJM Model. JFQA.
"""
from __future__ import annotations

import numpy as np
from dataclasses import dataclass, field
from typing import List
from scipy.stats import norm

from sofr_engine.curve import DiscountCurve


# ── Inflation Curve ───────────────────────────────────────────────────────────

@dataclass
class InflationCurve:
    """
    Piecewise-flat forward CPI growth curve.

    Parameters
    ----------
    times : (M,) array of pillar times in years, strictly increasing, > 0
    rates : (M,) array of annualised expected CPI growth rates,
            e.g. 0.025 = 2.5 %

    The curve is piecewise-flat: rate[k] applies to the interval
    (times[k-1], times[k]] where times[-1] = 0 by convention.
    """

    times: np.ndarray
    rates: np.ndarray

    # ── constructor helpers ───────────────────────────────────────────────────

    @classmethod
    def flat(cls, rate: float, max_tenor: float = 30.0) -> "InflationCurve":
        """Create a single-pillar flat inflation curve."""
        return cls(times=np.array([max_tenor]), rates=np.array([rate]))

    def __post_init__(self) -> None:
        self.times = np.asarray(self.times, dtype=float)
        self.rates = np.asarray(self.rates, dtype=float)

        if len(self.times) != len(self.rates):
            raise ValueError(
                f"times and rates must have the same length, got "
                f"{len(self.times)} vs {len(self.rates)}"
            )
        if len(self.times) == 0:
            raise ValueError("InflationCurve must have at least one pillar")
        if np.any(self.times <= 0):
            raise ValueError("All pillar times must be strictly positive")
        if np.any(np.diff(self.times) <= 0):
            raise ValueError("Pillar times must be strictly increasing")
        if np.any(self.rates < -0.20) or np.any(self.rates > 0.50):
            raise ValueError(
                "Rates must be in [-0.20, 0.50]; got values outside range"
            )

    # ── interpolation / analytics ─────────────────────────────────────────────

    def forward_inflation(self, T: float) -> float:
        """
        Piecewise-flat forward inflation rate at time T.
        Returns the rate of the first pillar with time >= T,
        or the last rate if T is beyond all pillars.
        """
        for t, r in zip(self.times, self.rates):
            if T <= t:
                return float(r)
        return float(self.rates[-1])

    def cpi_ratio(self, T: float) -> float:
        """
        E[CPI(T) / CPI(0)] under the piecewise-flat forward rate model.

        Integrates segment-by-segment:
          result = Prod_k (1 + r_k)^{delta_t_k}
        where delta_t_k is the portion of segment k that falls within [0, T].
        """
        if T <= 0:
            return 1.0
        result = 1.0
        prev_t = 0.0
        for t, r in zip(self.times, self.rates):
            if T <= t:
                dt = T - prev_t
                result *= (1 + r) ** dt
                return result
            else:
                dt = t - prev_t
                result *= (1 + r) ** dt
                prev_t = t
        # T is beyond the last pillar — use the last rate
        dt = T - prev_t
        result *= (1 + self.rates[-1]) ** dt
        return result

    def yoy_forward(self, T_start: float, T_end: float) -> float:
        """
        Year-on-year forward inflation for the period [T_start, T_end].

        Returns E[CPI(T_end)/CPI(T_start)] - 1.
        """
        return self.cpi_ratio(T_end) / self.cpi_ratio(T_start) - 1.0

    def par_fixed_rate(self, T: float) -> float:
        """
        Par fixed rate K such that a zero-coupon inflation swap is at-par.
        K = CPI_ratio(T)^(1/T) - 1
        """
        cr = self.cpi_ratio(T)
        return cr ** (1.0 / T) - 1.0


# ── Product data classes ──────────────────────────────────────────────────────

@dataclass
class ZCInflationSwap:
    """Zero-coupon inflation swap."""
    maturity: float
    fixed_rate: float
    notional: float = 1_000_000.0
    receive_inflation: bool = True


@dataclass
class YoYInflationSwap:
    """Year-on-year inflation swap."""
    payment_dates: list
    fixed_rate: float
    notional: float = 1_000_000.0
    receive_inflation: bool = True


@dataclass
class InflationCapFloor:
    """Inflation cap or floor."""
    payment_dates: list
    strike: float
    vol: float
    notional: float = 1_000_000.0
    is_cap: bool = True


# ── Result data classes ───────────────────────────────────────────────────────

@dataclass
class ZCInflationResult:
    """Result of zero-coupon inflation swap pricing."""
    pv: float
    float_leg_pv: float
    fixed_leg_pv: float
    par_rate: float
    cpi_ratio: float
    breakeven_bps: float  # (par_rate - fixed_rate) * 10_000


@dataclass
class YoYResult:
    """Result of year-on-year inflation swap pricing."""
    pv: float
    float_leg_pv: float
    fixed_leg_pv: float
    n_periods: int


@dataclass
class InflationCapResult:
    """Result of inflation cap/floor pricing."""
    pv: float
    caplet_pvs: list
    is_cap: bool


# ── Pricing functions ─────────────────────────────────────────────────────────

def zc_inflation_pv(
    nominal_curve: DiscountCurve,
    infl_curve: InflationCurve,
    swap: ZCInflationSwap,
) -> ZCInflationResult:
    """
    Price a zero-coupon inflation swap.

    Float leg: N * P(T) * (CPI(T)/CPI(0) - 1)  [expected value]
    Fixed leg: N * P(T) * ((1+K)^T - 1)
    PV = float_leg - fixed_leg  (if receive_inflation)
    """
    T = swap.maturity
    N = swap.notional
    K = swap.fixed_rate

    P = nominal_curve.df(T)
    cr = infl_curve.cpi_ratio(T)

    float_leg_pv = N * P * (cr - 1)
    fixed_leg_pv = N * P * ((1 + K) ** T - 1)

    pv = (float_leg_pv - fixed_leg_pv) if swap.receive_inflation else (fixed_leg_pv - float_leg_pv)

    par_rate = infl_curve.par_fixed_rate(T)
    breakeven_bps = (par_rate - K) * 10_000

    return ZCInflationResult(
        pv=pv,
        float_leg_pv=float_leg_pv,
        fixed_leg_pv=fixed_leg_pv,
        par_rate=par_rate,
        cpi_ratio=cr,
        breakeven_bps=breakeven_bps,
    )


def yoy_inflation_pv(
    nominal_curve: DiscountCurve,
    infl_curve: InflationCurve,
    swap: YoYInflationSwap,
) -> YoYResult:
    """
    Price a year-on-year inflation swap.

    Each period [T_{k-1}, T_k] contributes:
      Float: N * alpha_k * P(T_k) * YoY(T_{k-1}, T_k)
      Fixed: N * alpha_k * P(T_k) * K
    where alpha_k = T_k - T_{k-1}.
    """
    dates = sorted(swap.payment_dates)
    N = swap.notional
    K = swap.fixed_rate

    T_prev = 0.0
    float_leg_pv = 0.0
    fixed_leg_pv = 0.0

    for T_k in dates:
        alpha = T_k - T_prev
        P = nominal_curve.df(T_k)
        yoy = infl_curve.yoy_forward(T_prev, T_k)
        float_leg_pv += N * alpha * P * yoy
        fixed_leg_pv += N * alpha * P * K
        T_prev = T_k

    pv = (float_leg_pv - fixed_leg_pv) if swap.receive_inflation else (fixed_leg_pv - float_leg_pv)

    return YoYResult(
        pv=pv,
        float_leg_pv=float_leg_pv,
        fixed_leg_pv=fixed_leg_pv,
        n_periods=len(dates),
    )


def inflation_caplet_pv(
    nominal_curve: DiscountCurve,
    infl_curve: InflationCurve,
    T_start: float,
    T_end: float,
    strike: float,
    vol: float,
    notional: float,
    is_cap: bool,
) -> float:
    """
    Price a single inflation caplet or floorlet using a Black-style formula.

    The caplet pays max(CPI(T_end)/CPI(T_start) - 1 - strike, 0) at T_end.
    We model the YoY ratio F = CPI(T_end)/CPI(T_start) as log-normal with
    vol driven by T_start (the reset date).

    For T_start ~ 0 or vol ~ 0 we return the intrinsic (discounted) value.
    """
    # Forward CPI ratio for the period
    if T_start > 1e-9:
        F = infl_curve.cpi_ratio(T_end) / infl_curve.cpi_ratio(T_start)
    else:
        F = infl_curve.cpi_ratio(T_end)

    alpha = T_end - T_start
    # Convert the annualised strike to a gross ratio over the period
    K_gross = (1 + strike) ** alpha
    P = nominal_curve.df(T_end)

    if T_start < 1e-9 or vol < 1e-10:
        # Intrinsic value only
        if is_cap:
            payoff = max(F - K_gross, 0.0)
        else:
            payoff = max(K_gross - F, 0.0)
        return notional * P * payoff

    sqrtT = np.sqrt(T_start)
    d1 = (np.log(F / K_gross) + 0.5 * vol ** 2 * T_start) / (vol * sqrtT)
    d2 = d1 - vol * sqrtT

    if is_cap:
        payoff = F * norm.cdf(d1) - K_gross * norm.cdf(d2)
    else:
        payoff = K_gross * norm.cdf(-d2) - F * norm.cdf(-d1)

    return notional * P * payoff


def inflation_cap_floor_pv(
    nominal_curve: DiscountCurve,
    infl_curve: InflationCurve,
    cap_floor: InflationCapFloor,
) -> InflationCapResult:
    """
    Price an inflation cap or floor as a strip of caplets/floorlets.
    """
    dates = sorted(cap_floor.payment_dates)
    T_prev = 0.0
    caplet_pvs = []

    for T_k in dates:
        cpv = inflation_caplet_pv(
            nominal_curve,
            infl_curve,
            T_prev,
            T_k,
            cap_floor.strike,
            cap_floor.vol,
            cap_floor.notional,
            cap_floor.is_cap,
        )
        caplet_pvs.append(cpv)
        T_prev = T_k

    total_pv = sum(caplet_pvs)
    return InflationCapResult(pv=total_pv, caplet_pvs=caplet_pvs, is_cap=cap_floor.is_cap)


def inflation_cap_floor_parity(
    nominal_curve: DiscountCurve,
    infl_curve: InflationCurve,
    payment_dates: list,
    strike: float,
    vol: float,
    notional: float,
) -> float:
    """
    Verify put-call parity for inflation caps and floors.

    Cap - Floor = YoY Swap (receive inflation, fixed = strike)

    Returns the difference (cap_pv - floor_pv) - yoy_swap_pv.
    Should be near zero for consistent inputs.
    """
    cap = InflationCapFloor(
        payment_dates=payment_dates,
        strike=strike,
        vol=vol,
        notional=notional,
        is_cap=True,
    )
    floor = InflationCapFloor(
        payment_dates=payment_dates,
        strike=strike,
        vol=vol,
        notional=notional,
        is_cap=False,
    )
    cap_result = inflation_cap_floor_pv(nominal_curve, infl_curve, cap)
    floor_result = inflation_cap_floor_pv(nominal_curve, infl_curve, floor)

    yoy_swap = YoYInflationSwap(
        payment_dates=payment_dates,
        fixed_rate=strike,
        notional=notional,
        receive_inflation=True,
    )
    yoy_result = yoy_inflation_pv(nominal_curve, infl_curve, yoy_swap)

    difference = (cap_result.pv - floor_result.pv) - yoy_result.pv
    return difference


def breakeven_inflation(
    nominal_curve: DiscountCurve,
    infl_curve: InflationCurve,
    T: float,
) -> dict:
    """
    Compute breakeven inflation, nominal yield, and implied real yield.

    Uses the Fisher approximation: real ~ nominal - breakeven_inflation.
    Nominal yield is the continuously-compounded zero rate: -ln(DF(T))/T.
    """
    bei = infl_curve.par_fixed_rate(T)
    P = nominal_curve.df(T)
    r_nominal = -np.log(P) / T  # continuously compounded zero rate
    r_real = r_nominal - bei    # Fisher approximation
    return {
        "breakeven_inflation": bei,
        "nominal_yield": r_nominal,
        "real_yield": r_real,
    }


# ── Calibration ───────────────────────────────────────────────────────────────

def calibrate_inflation_curve(
    nominal_curve: DiscountCurve,
    maturities: list,
    zc_par_rates: list,
) -> InflationCurve:
    """
    Bootstrap a piecewise-flat InflationCurve from zero-coupon inflation par rates.

    Each par rate K_i implies CPI_ratio(T_i) = (1 + K_i)^{T_i}.
    The forward rate for segment [T_{i-1}, T_i] is derived as:
      f_i = (CPI_ratio(T_i) / CPI_ratio(T_{i-1}))^{1/(T_i - T_{i-1})} - 1

    Parameters
    ----------
    nominal_curve  : not used in bootstrap arithmetic, kept for API consistency
    maturities     : sorted list of maturity times (years)
    zc_par_rates   : list of zero-coupon par inflation rates (decimal)

    Returns
    -------
    InflationCurve with times = maturities, rates = bootstrapped forwards
    """
    if len(maturities) != len(zc_par_rates):
        raise ValueError("maturities and zc_par_rates must have the same length")

    maturities = list(maturities)
    zc_par_rates = list(zc_par_rates)

    forward_rates = []
    prev_T = 0.0
    prev_cpi_ratio = 1.0

    for T, par in zip(maturities, zc_par_rates):
        cpi_ratio_T = (1 + par) ** T
        dt = T - prev_T
        fwd = (cpi_ratio_T / prev_cpi_ratio) ** (1.0 / dt) - 1.0
        forward_rates.append(fwd)
        prev_T = T
        prev_cpi_ratio = cpi_ratio_T

    return InflationCurve(
        times=np.array(maturities),
        rates=np.array(forward_rates),
    )
