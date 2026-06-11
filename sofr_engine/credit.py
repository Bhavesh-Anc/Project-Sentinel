"""
Credit Default Swap (CDS) Pricing Module
=========================================
Implements single-name CDS pricing under the standard ISDA model:
- Piecewise-constant hazard rate (intensity) curve bootstrapped from par spreads
- Fee leg (premium) and protection leg present values
- Par spread (breakeven spread), CS01, DV01, recovery-adjusted analytics

Model
-----
The survival probability from today to t:
    Q(t) = exp(-∫₀ᵗ λ(u) du)

For piecewise-constant hazard rates on a grid (T₀=0, T₁, T₂, ...):
    Q(t) = Q(Tᵢ) × exp(-λᵢ₊₁ × (t - Tᵢ))   for Tᵢ ≤ t < Tᵢ₊₁

The CDS fee leg pays the spread s on surviving notional each quarter:
    PV_fee = N × s × Σᵢ αᵢ × DF(Tᵢ) × Q(Tᵢ)   (survival-weighted cashflows)

The protection leg pays (1 - R) × notional on default:
    PV_prot = N × (1-R) × ∫₀ᵀ DF(t) × (-dQ/dt) dt
            ≈ N × (1-R) × Σᵢ DF(tᵢ₊½) × [Q(Tᵢ) - Q(Tᵢ₊₁)]

References
----------
O'Kane, D. & Turnbull, S. (2003) "Valuation of Credit Default Swaps",
Lehman Brothers QFA whitepaper.
ISDA CDS Standard Model documentation, 2009.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Sequence

import numpy as np
from scipy.optimize import brentq

from .curve import DiscountCurve

__all__ = [
    "HazardRateCurve",
    "CDSContract",
    "CDSResult",
    "bootstrap_hazard_curve",
    "cds_pv",
    "cds_par_spread",
    "cds_cs01",
    "cds_dv01",
    "risky_annuity",
]


# ── Hazard Rate Curve ─────────────────────────────────────────────────────────

@dataclass
class HazardRateCurve:
    """
    Piecewise-constant hazard rate (default intensity) curve.

    Parameters
    ----------
    times    : pillar maturities in years [T₁, T₂, ...] (T₀=0 implied)
    hazards  : hazard rates λᵢ > 0 for each interval (0, T₁], (T₁, T₂], ...
    recovery : assumed recovery rate R (default 0.40)
    """
    times:    np.ndarray
    hazards:  np.ndarray
    recovery: float = 0.40

    def __post_init__(self) -> None:
        self.times   = np.asarray(self.times,   dtype=float)
        self.hazards = np.asarray(self.hazards, dtype=float)
        if len(self.times) != len(self.hazards):
            raise ValueError("times and hazards must have the same length")
        if np.any(self.hazards < 0):
            raise ValueError("all hazard rates must be non-negative")
        if not 0.0 <= self.recovery < 1.0:
            raise ValueError("recovery must be in [0, 1)")

    def survival(self, t: float) -> float:
        """Q(t) = P(no default before t)."""
        if t <= 0:
            return 1.0
        cum_haz = 0.0
        t_prev  = 0.0
        for T_k, lam_k in zip(self.times, self.hazards):
            if t <= T_k:
                cum_haz += lam_k * (t - t_prev)
                return math.exp(-cum_haz)
            cum_haz += lam_k * (T_k - t_prev)
            t_prev   = T_k
        # Beyond last pillar: extend flat
        lam_last = float(self.hazards[-1])
        cum_haz += lam_last * (t - t_prev)
        return math.exp(-cum_haz)

    def hazard_at(self, t: float) -> float:
        """Instantaneous hazard rate λ(t)."""
        for T_k, lam_k in zip(self.times, self.hazards):
            if t <= T_k:
                return float(lam_k)
        return float(self.hazards[-1])

    def default_prob(self, t1: float, t2: float) -> float:
        """Marginal default probability P(t1 < τ ≤ t2)."""
        return self.survival(t1) - self.survival(t2)

    @classmethod
    def flat(cls, hazard: float, maturities: Sequence[float],
             recovery: float = 0.40) -> "HazardRateCurve":
        """Construct a flat hazard rate curve."""
        times   = np.asarray(maturities, dtype=float)
        hazards = np.full(len(times), hazard)
        return cls(times=times, hazards=hazards, recovery=recovery)


# ── CDS Contract ──────────────────────────────────────────────────────────────

@dataclass
class CDSContract:
    """
    Standard CDS contract specification.

    Parameters
    ----------
    maturity_years : contract maturity in years from today
    coupon         : running coupon (e.g. 0.01 = 100bps standard)
    notional       : face value
    recovery       : assumed recovery on default (typically 0.40)
    freq           : coupon payment frequency per year (4 = quarterly)
    buy_protection : True = fee payer / protection buyer; False = fee receiver
    """
    maturity_years: float
    coupon:         float   = 0.01     # 100bps standard
    notional:       float   = 10_000_000.0
    recovery:       float   = 0.40
    freq:           int     = 4
    buy_protection: bool    = True

    def __post_init__(self) -> None:
        if self.maturity_years <= 0:
            raise ValueError("maturity_years must be positive")
        if not 0.0 <= self.recovery < 1.0:
            raise ValueError("recovery must be in [0, 1)")
        if self.freq <= 0:
            raise ValueError("freq must be positive")

    def _coupon_dates(self) -> list[float]:
        """Quarterly (or freq) coupon payment dates."""
        dt  = 1.0 / self.freq
        eps = 1e-9
        t   = dt
        dates: list[float] = []
        while t <= self.maturity_years + eps:
            dates.append(round(t, 8))
            t += dt
        return dates


# ── CDS result ────────────────────────────────────────────────────────────────

@dataclass
class CDSResult:
    pv:            float
    fee_leg_pv:    float
    prot_leg_pv:   float
    par_spread_bps: float
    risky_annuity: float
    cs01:          float    # sensitivity to 1bp upward shift in all hazard rates
    dv01:          float    # sensitivity to 1bp parallel shift in risk-free curve
    upfront:       float    # upfront payment = PV(protection) - PV(fee at coupon)


# ── Risky annuity ─────────────────────────────────────────────────────────────

def risky_annuity(
    discount_curve: DiscountCurve,
    hazard_curve:   HazardRateCurve,
    maturity:       float,
    freq:           int = 4,
    n_accrual:      int = 4,
) -> float:
    """
    Compute the risky annuity (present value of $1 per year paid until default or maturity).

        A = Σᵢ αᵢ × DF(Tᵢ) × Q(Tᵢ) + accrual_on_default

    The accrual-on-default term approximates the partial coupon owed on default mid-period.

    Parameters
    ----------
    maturity  : annuity maturity in years
    freq      : coupon frequency per year
    n_accrual : sub-steps per coupon period for accrual integration
    """
    dt     = 1.0 / freq
    t_prev = 0.0
    annuity = 0.0

    t_pay = dt
    while t_pay <= maturity + 1e-9:
        alpha = dt  # coupon period length (ACT/360 simplified)
        Q_pay = hazard_curve.survival(t_pay)
        DF_pay = float(discount_curve.df(t_pay))
        annuity += alpha * DF_pay * Q_pay

        # Accrual-on-default: ∫_{t_prev}^{t_pay} accrual(t) × DF(t) × (-dQ/dt) dt
        for k in range(n_accrual):
            t_mid  = t_prev + (k + 0.5) * dt / n_accrual
            accrual = (t_mid - t_prev) / alpha  # accrued fraction
            DF_mid  = float(discount_curve.df(t_mid))
            # Approximate (-dQ/dt) × dt ≈ Q(t_mid-δ/2) - Q(t_mid+δ/2)
            dq      = hazard_curve.default_prob(
                t_prev + k * dt / n_accrual,
                t_prev + (k + 1) * dt / n_accrual,
            )
            annuity += accrual * alpha * DF_mid * dq / (dt / n_accrual) * (dt / n_accrual)

        t_prev  = t_pay
        t_pay  += dt

    return annuity


# ── Protection leg ────────────────────────────────────────────────────────────

def _protection_leg_pv(
    discount_curve: DiscountCurve,
    hazard_curve:   HazardRateCurve,
    maturity:       float,
    n_steps:        int = 200,
) -> float:
    """
    PV of the protection leg (pays 1-R on default).

        PV_prot = (1-R) × ∫₀ᵀ DF(t) × (-dQ/dt) dt
               ≈ (1-R) × Σᵢ DF(tᵢ₊½) × [Q(tᵢ) - Q(tᵢ₊₁)]

    Uses a fine grid for accuracy.
    """
    R   = hazard_curve.recovery
    dt  = maturity / n_steps
    pv  = 0.0
    Q_prev = 1.0
    for i in range(n_steps):
        t_mid = (i + 0.5) * dt
        t_end = (i + 1.0) * dt
        Q_end = hazard_curve.survival(t_end)
        dQ    = Q_prev - Q_end
        pv   += float(discount_curve.df(t_mid)) * dQ
        Q_prev = Q_end
    return (1.0 - R) * pv


# ── CDS PV ────────────────────────────────────────────────────────────────────

def cds_pv(
    discount_curve: DiscountCurve,
    hazard_curve:   HazardRateCurve,
    cds:            CDSContract,
) -> CDSResult:
    """
    Full CDS valuation: fee leg, protection leg, par spread, CS01, DV01.

    Returns CDSResult with all analytics.
    """
    N  = cds.notional
    s  = cds.coupon
    T  = cds.maturity_years
    R  = hazard_curve.recovery
    freq = cds.freq

    ann  = risky_annuity(discount_curve, hazard_curve, T, freq)
    prot = _protection_leg_pv(discount_curve, hazard_curve, T)

    fee_pv  = N * s * ann
    prot_pv = N * prot

    # From protection buyer's perspective: pay fee, receive prot
    if cds.buy_protection:
        pv = prot_pv - fee_pv
    else:
        pv = fee_pv - prot_pv

    par_s    = prot / ann if ann > 1e-14 else float("nan")
    upfront  = (prot_pv - fee_pv) if cds.buy_protection else (fee_pv - prot_pv)

    cs01_val = _cs01(discount_curve, hazard_curve, cds, ann, prot_pv)
    dv01_val = _dv01(discount_curve, hazard_curve, cds)

    return CDSResult(
        pv=round(pv, 2),
        fee_leg_pv=round(fee_pv, 2),
        prot_leg_pv=round(prot_pv, 2),
        par_spread_bps=round(par_s * 10_000, 4),
        risky_annuity=round(ann, 6),
        cs01=round(cs01_val, 2),
        dv01=round(dv01_val, 2),
        upfront=round(upfront, 2),
    )


def cds_par_spread(
    discount_curve: DiscountCurve,
    hazard_curve:   HazardRateCurve,
    maturity:       float,
    freq:           int = 4,
) -> float:
    """Par spread s* such that CDS PV = 0 (decimal)."""
    ann  = risky_annuity(discount_curve, hazard_curve, maturity, freq)
    prot = _protection_leg_pv(discount_curve, hazard_curve, maturity)
    return prot / ann if ann > 1e-14 else float("nan")


# ── Risk sensitivities ────────────────────────────────────────────────────────

def _cs01(
    discount_curve: DiscountCurve,
    hazard_curve:   HazardRateCurve,
    cds:            CDSContract,
    ann:            float,
    prot_pv:        float,
) -> float:
    """
    CS01: change in PV for 1bp parallel upward shift in all hazard rates.
    Uses central finite difference.
    """
    bump  = 1e-4  # 1bp
    h_up  = HazardRateCurve(hazard_curve.times, hazard_curve.hazards + bump,
                             hazard_curve.recovery)
    h_dn  = HazardRateCurve(hazard_curve.times, hazard_curve.hazards - bump,
                             hazard_curve.recovery)

    N, s, T, freq = cds.notional, cds.coupon, cds.maturity_years, cds.freq
    ann_up  = risky_annuity(discount_curve, h_up, T, freq)
    ann_dn  = risky_annuity(discount_curve, h_dn, T, freq)
    prot_up = _protection_leg_pv(discount_curve, h_up, T)
    prot_dn = _protection_leg_pv(discount_curve, h_dn, T)

    pv_up = N * (prot_up - s * ann_up) if cds.buy_protection else N * (s * ann_up - prot_up)
    pv_dn = N * (prot_dn - s * ann_dn) if cds.buy_protection else N * (s * ann_dn - prot_dn)
    return (pv_up - pv_dn) / 2.0


def cds_cs01(
    discount_curve: DiscountCurve,
    hazard_curve:   HazardRateCurve,
    cds:            CDSContract,
) -> float:
    """CS01 standalone: PV change for +1bp in all hazard rates."""
    ann  = risky_annuity(discount_curve, hazard_curve, cds.maturity_years, cds.freq)
    prot = _protection_leg_pv(discount_curve, hazard_curve, cds.maturity_years)
    return _cs01(discount_curve, hazard_curve, cds, ann, prot)


def _dv01(
    discount_curve: DiscountCurve,
    hazard_curve:   HazardRateCurve,
    cds:            CDSContract,
) -> float:
    """DV01: change in CDS PV for +1bp parallel shift in risk-free curve."""
    bump  = 1e-4
    times = discount_curve._times
    logdf = discount_curve._log_df

    curve_up = DiscountCurve(discount_curve.ref_date, times, np.exp(logdf - bump * times))
    curve_dn = DiscountCurve(discount_curve.ref_date, times, np.exp(logdf + bump * times))

    N, s, T, freq = cds.notional, cds.coupon, cds.maturity_years, cds.freq

    ann_up  = risky_annuity(curve_up, hazard_curve, T, freq)
    ann_dn  = risky_annuity(curve_dn, hazard_curve, T, freq)
    prot_up = _protection_leg_pv(curve_up, hazard_curve, T)
    prot_dn = _protection_leg_pv(curve_dn, hazard_curve, T)

    pv_up = N * (prot_up - s * ann_up) if cds.buy_protection else N * (s * ann_up - prot_up)
    pv_dn = N * (prot_dn - s * ann_dn) if cds.buy_protection else N * (s * ann_dn - prot_dn)
    return (pv_up - pv_dn) / 2.0


def cds_dv01(
    discount_curve: DiscountCurve,
    hazard_curve:   HazardRateCurve,
    cds:            CDSContract,
) -> float:
    """DV01 standalone."""
    return _dv01(discount_curve, hazard_curve, cds)


# ── Hazard rate bootstrap ─────────────────────────────────────────────────────

def bootstrap_hazard_curve(
    discount_curve: DiscountCurve,
    maturities:     Sequence[float],
    par_spreads:    Sequence[float],
    recovery:       float = 0.40,
    freq:           int   = 4,
) -> HazardRateCurve:
    """
    Bootstrap a piecewise-constant hazard rate curve from CDS par spreads.

    For each maturity Tᵢ (in order), finds λᵢ such that the par CDS spread
    matches the market quote, holding previous hazard rates fixed.

    Parameters
    ----------
    maturities  : CDS maturities in years [T₁, T₂, ...] (must be ascending)
    par_spreads : market par spreads in decimal [s₁, s₂, ...] (e.g. 0.01 = 100bps)
    recovery    : assumed recovery rate
    freq        : coupon frequency (default 4 = quarterly)

    Returns
    -------
    HazardRateCurve with bootstrapped hazard rates
    """
    maturities  = list(maturities)
    par_spreads = list(par_spreads)
    if len(maturities) != len(par_spreads):
        raise ValueError("maturities and par_spreads must have the same length")
    if sorted(maturities) != maturities:
        raise ValueError("maturities must be in ascending order")

    bootstrapped_times   = []
    bootstrapped_hazards = []

    for i, (T_i, s_i) in enumerate(zip(maturities, par_spreads)):
        def par_spread_error(lam_i: float) -> float:
            # Build curve with current pillar set to lam_i
            hazards = bootstrapped_hazards + [lam_i]
            times   = bootstrapped_times   + [T_i]
            hc = HazardRateCurve(times=np.array(times), hazards=np.array(hazards),
                                  recovery=recovery)
            s_model = cds_par_spread(discount_curve, hc, T_i, freq)
            return s_model - s_i

        # Search for λᵢ in a reasonable range
        try:
            lam_sol = brentq(par_spread_error, 1e-8, 0.50, xtol=1e-10)
        except ValueError:
            # If brentq fails (e.g. spread too high), fall back to implied
            lam_sol = s_i / (1.0 - recovery) if (1.0 - recovery) > 0 else 0.01

        bootstrapped_times.append(T_i)
        bootstrapped_hazards.append(lam_sol)

    return HazardRateCurve(
        times    = np.array(bootstrapped_times),
        hazards  = np.array(bootstrapped_hazards),
        recovery = recovery,
    )
