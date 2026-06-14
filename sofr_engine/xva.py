"""
XVA — Credit, Debt, and Funding Valuation Adjustments
======================================================
Implements the three main bilateral OTC derivative valuation adjustments
under a G2++ interest-rate model for exposure simulation:

CVA (Credit Valuation Adjustment)
    Cost of counterparty default on an OTC derivative:

        CVA = (1-R) × Σᵢ DF(0,tᵢ) × EPE(tᵢ) × PD(tᵢ₋₁, tᵢ)

DVA (Debt Valuation Adjustment)
    Benefit from our own potential default:

        DVA = (1-R_own) × Σᵢ DF(0,tᵢ) × ENE(tᵢ) × PD_own(tᵢ₋₁, tᵢ)

FVA (Funding Valuation Adjustment)
    Simplified funding cost/benefit from the net uncollateralised exposure:

        FVA = -s_f × Σᵢ DF(0,tᵢ) × (EPE(tᵢ) − ENE(tᵢ)) × Δtᵢ

    where s_f is the funding spread.

Exposure Simulation
-------------------
The mark-to-market of a SOFR OIS swap at simulation time tᵢ is computed
analytically from the G2++ state (x(tᵢ), y(tᵢ)) using:

    Floating leg value ≈ DF(tᵢ, T_start) − DF(tᵢ, T_end)
    Fixed leg value    = Σₖ αₖ × K × DF(tᵢ, Tₖ)

where DF(tᵢ, Tₖ) = g2pp_zcb(curve, params, tᵢ, Tₖ, x, y) for each path.

References
----------
Gregory, J. (2012) "Counterparty Credit Risk and Credit Value Adjustment",
Wiley Finance.

Brigo, D. & Mercurio, F. (2006) "Interest Rate Models — Theory and Practice",
Springer.

Piterbarg, V. (2010) "Funding beyond discounting: collateral agreements and
derivatives pricing", Risk Magazine.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Sequence

import numpy as np

from .curve import DiscountCurve
from .credit import HazardRateCurve
from .g2pp import G2ppParams, G2ppSimResult, simulate_g2pp, g2pp_zcb

__all__ = [
    "XVAParams",
    "EPEProfile",
    "XVAResult",
    "compute_epe_profile",
    "cva",
    "dva",
    "fva",
    "full_xva",
    "cva_sensitivity",
]


# ── Parameter and result dataclasses ──────────────────────────────────────────

@dataclass
class XVAParams:
    """
    Configuration for XVA Monte Carlo simulation and funding.

    Attributes
    ----------
    n_paths       : number of simulation paths (must be even for antithetic)
    n_steps       : number of time steps along the horizon
    seed          : RNG seed for reproducibility
    funding_spread: annual funding spread s_f (e.g. 0.005 = 50bps)
    """
    n_paths:        int   = 2_000
    n_steps:        int   = 50
    seed:           int   = 42
    funding_spread: float = 0.005  # 50bps funding cost

    def __post_init__(self) -> None:
        if self.n_paths < 2:
            raise ValueError("n_paths must be at least 2")
        if self.n_steps < 1:
            raise ValueError("n_steps must be at least 1")
        if self.funding_spread < 0:
            raise ValueError("funding_spread must be non-negative")


@dataclass
class EPEProfile:
    """
    Exposure profile computed along simulation time steps.

    Attributes
    ----------
    times         : simulation time grid, shape (n_steps+1,)
    epe           : Expected Positive Exposure E[max(V,0)], shape (n_steps+1,)
    ene           : Expected Negative Exposure E[max(-V,0)] (positive number),
                    shape (n_steps+1,)
    mean_exposure : mean signed exposure E[V], shape (n_steps+1,)
    """
    times:         np.ndarray  # (n_steps+1,)
    epe:           np.ndarray  # (n_steps+1,)  E[max(V,0)]
    ene:           np.ndarray  # (n_steps+1,)  E[max(-V,0)], stored positive
    mean_exposure: np.ndarray  # (n_steps+1,)  E[V]


@dataclass
class XVAResult:
    """
    Full XVA results for an OTC derivative position.

    Attributes
    ----------
    cva           : Credit Valuation Adjustment (positive = cost)
    dva           : Debt Valuation Adjustment (positive = benefit)
    fva           : Funding Valuation Adjustment (negative = cost)
    total_xva     : cva - dva + fva  (add to clean price for inclusive value)
    epe_profile   : underlying exposure profile
    cva_by_period : CVA contribution per time bucket, shape (n_steps,)
    """
    cva:           float
    dva:           float
    fva:           float
    total_xva:     float        # cva - dva + fva
    epe_profile:   EPEProfile
    cva_by_period: np.ndarray   # (n_steps,)


# ── Internal helpers ───────────────────────────────────────────────────────────

def _payment_times(maturity: float, freq: int) -> list[float]:
    """Generate evenly-spaced payment times at 1/freq spacing up to maturity."""
    dt = 1.0 / freq
    times: list[float] = []
    t = dt
    while t <= maturity + 1e-9:
        times.append(round(t, 10))
        t += dt
    return times


def _swap_mtm_at_step(
    sim:          G2ppSimResult,
    step:         int,
    pay_times:    list[float],
    dt_fixed:     float,
    fixed_rate:   float,
    notional:     float,
    pay_fixed:    bool,
) -> np.ndarray:
    """
    Compute path-wise swap mark-to-market at simulation step ``step``.

    For a SOFR OIS swap observed at time t = sim.t_grid[step]:
      - Remaining payments are those at T_k > t
      - Floating leg value = DF(t, T_first_remaining) - DF(t, T_end)
        (OIS floating leg is worth par minus the terminal discount factor,
         from the first remaining reset date to maturity)
      - Fixed leg value   = Σ_k αₖ × K × DF(t, T_k)

    Uses sim.zcb_prices_at for vectorised path computation.

    Returns
    -------
    mtm : ndarray of shape (n_paths,), positive = asset to us (if pay_fixed)
    """
    t = float(sim.t_grid[step])

    # Filter payment dates that are strictly after the current time
    remaining = [T for T in pay_times if T > t + 1e-10]

    if len(remaining) == 0:
        # Swap has matured — zero exposure
        return np.zeros(sim.n_paths)

    # Compute ZCB prices P(t, T_k) for all remaining payment dates
    # zcb_prices_at returns shape (n_paths, len(remaining))
    p_mat = sim.zcb_prices_at(step, remaining)  # (n_paths, len(remaining))

    # Floating leg (OIS approximation):
    #   PV_float = DF(t, T_first) - DF(t, T_end)
    # For the very first step (t=0) or when the first payment is in the future,
    # this is exact for a spot-starting OIS.
    p_first = p_mat[:, 0]   # DF(t, T_first_remaining), shape (n_paths,)
    p_end   = p_mat[:, -1]  # DF(t, T_end),             shape (n_paths,)
    float_leg = notional * (p_first - p_end)

    # Fixed leg: Σ_k α_k × K × DF(t, T_k)
    # All periods use dt_fixed (uniform coupon structure)
    fixed_leg = notional * fixed_rate * dt_fixed * np.sum(p_mat, axis=1)

    # Net value from the perspective of the fixed payer:
    #   V = float_leg - fixed_leg
    # For a fixed receiver, flip sign.
    if pay_fixed:
        mtm = float_leg - fixed_leg
    else:
        mtm = fixed_leg - float_leg

    return mtm


# ── Core XVA functions ─────────────────────────────────────────────────────────

def compute_epe_profile(
    curve:       DiscountCurve,
    g2pp_params: G2ppParams,
    maturity:    float,
    fixed_rate:  float,
    notional:    float,
    pay_fixed:   bool,
    xva_params:  XVAParams,
    freq:        int = 1,
) -> EPEProfile:
    """
    Simulate G2++ paths and compute the expected exposure profile.

    The swap mark-to-market at each simulation step is valued analytically
    using the G2++ ZCB formula, conditioning on the simulated state (x, y).

    Parameters
    ----------
    curve       : nominal SOFR discount curve
    g2pp_params : G2++ model parameters
    maturity    : swap maturity in years
    fixed_rate  : fixed coupon rate (decimal)
    notional    : swap notional
    pay_fixed   : True = fixed payer (standard), False = fixed receiver
    xva_params  : simulation and funding parameters
    freq        : fixed leg payment frequency (1=annual, 2=semi-annual)

    Returns
    -------
    EPEProfile with times, epe, ene, mean_exposure arrays of length n_steps+1
    """
    if maturity <= 0:
        raise ValueError("maturity must be positive")
    if notional <= 0:
        raise ValueError("notional must be positive")
    if freq <= 0:
        raise ValueError("freq must be positive")

    pay_times = _payment_times(maturity, freq)
    dt_fixed  = 1.0 / freq

    # Simulate G2++ up to the swap maturity
    sim = simulate_g2pp(
        curve      = curve,
        params     = g2pp_params,
        horizon    = maturity,
        n_steps    = xva_params.n_steps,
        n_paths    = xva_params.n_paths,
        seed       = xva_params.seed,
        antithetic = True,
    )

    n_grid = xva_params.n_steps + 1
    epe           = np.zeros(n_grid)
    ene           = np.zeros(n_grid)
    mean_exposure = np.zeros(n_grid)

    for step in range(n_grid):
        mtm = _swap_mtm_at_step(
            sim       = sim,
            step      = step,
            pay_times = pay_times,
            dt_fixed  = dt_fixed,
            fixed_rate = fixed_rate,
            notional  = notional,
            pay_fixed = pay_fixed,
        )
        epe[step]           = float(np.mean(np.maximum(mtm,  0.0)))
        ene[step]           = float(np.mean(np.maximum(-mtm, 0.0)))
        mean_exposure[step] = float(np.mean(mtm))

    return EPEProfile(
        times         = sim.t_grid.copy(),
        epe           = epe,
        ene           = ene,
        mean_exposure = mean_exposure,
    )


def cva(
    curve:             DiscountCurve,
    counterparty_hazard: HazardRateCurve,
    epe_profile:       EPEProfile,
) -> float:
    """
    Compute CVA using the standard credit-loss integral:

        CVA = (1-R) × Σᵢ DF(0,tᵢ) × EPE(tᵢ) × PD(tᵢ₋₁, tᵢ)

    Parameters
    ----------
    curve                : nominal discount curve for DF(0,t)
    counterparty_hazard  : counterparty's default intensity curve
    epe_profile          : pre-computed EPE profile

    Returns
    -------
    CVA as a positive float (cost to us of counterparty default)
    """
    R        = counterparty_hazard.recovery
    times    = epe_profile.times
    epe      = epe_profile.epe

    total = 0.0
    t_prev = 0.0
    for i in range(1, len(times)):
        t_i   = float(times[i])
        df_i  = float(curve.df(t_i))
        epe_i = float(epe[i])
        pd_i  = counterparty_hazard.default_prob(t_prev, t_i)
        total += df_i * epe_i * pd_i
        t_prev = t_i

    return (1.0 - R) * total


def dva(
    curve:        DiscountCurve,
    own_hazard:   HazardRateCurve,
    epe_profile:  EPEProfile,
) -> float:
    """
    Compute DVA using the standard formula with Expected Negative Exposure:

        DVA = (1-R_own) × Σᵢ DF(0,tᵢ) × ENE(tᵢ) × PD_own(tᵢ₋₁, tᵢ)

    Parameters
    ----------
    curve        : nominal discount curve
    own_hazard   : our own default intensity curve
    epe_profile  : pre-computed EPE profile (ENE field used)

    Returns
    -------
    DVA as a positive float (benefit from our own potential default)
    """
    R        = own_hazard.recovery
    times    = epe_profile.times
    ene      = epe_profile.ene

    total = 0.0
    t_prev = 0.0
    for i in range(1, len(times)):
        t_i   = float(times[i])
        df_i  = float(curve.df(t_i))
        ene_i = float(ene[i])
        pd_i  = own_hazard.default_prob(t_prev, t_i)
        total += df_i * ene_i * pd_i
        t_prev = t_i

    return (1.0 - R) * total


def fva(
    curve:       DiscountCurve,
    epe_profile: EPEProfile,
    xva_params:  XVAParams,
) -> float:
    """
    Compute FVA as the discounted funding cost/benefit on the net exposure:

        FVA = -s_f × Σᵢ DF(0,tᵢ) × (EPE(tᵢ) − ENE(tᵢ)) × Δtᵢ

    A positive net exposure (EPE > ENE) means we fund the position,
    making FVA negative (a cost). A negative net exposure yields positive FVA
    (a funding benefit from the liability position).

    Parameters
    ----------
    curve       : nominal discount curve
    epe_profile : pre-computed EPE/ENE profile
    xva_params  : contains funding_spread

    Returns
    -------
    FVA (negative for net borrowing, positive for net lending)
    """
    s_f   = xva_params.funding_spread
    times = epe_profile.times
    epe   = epe_profile.epe
    ene   = epe_profile.ene

    total = 0.0
    for i in range(1, len(times)):
        t_i   = float(times[i])
        t_prev = float(times[i - 1])
        dt_i  = t_i - t_prev
        df_i  = float(curve.df(t_i))
        net   = float(epe[i]) - float(ene[i])
        total += df_i * net * dt_i

    return -s_f * total


def full_xva(
    curve:               DiscountCurve,
    g2pp_params:         G2ppParams,
    maturity:            float,
    fixed_rate:          float,
    notional:            float,
    pay_fixed:           bool,
    counterparty_hazard: HazardRateCurve,
    own_hazard:          HazardRateCurve,
    xva_params:          XVAParams,
    freq:                int = 1,
) -> XVAResult:
    """
    Compute full bilateral XVA (CVA + DVA + FVA) for a SOFR OIS swap.

    Builds the EPE/ENE profile via G2++ Monte Carlo, then applies the
    standard CVA, DVA, and simplified FVA formulas.

    Parameters
    ----------
    curve                : nominal SOFR discount curve
    g2pp_params          : G2++ model parameters for rate simulation
    maturity             : swap maturity in years
    fixed_rate           : swap fixed rate (decimal)
    notional             : swap notional
    pay_fixed            : True = pay fixed (payer swap), False = receive fixed
    counterparty_hazard  : counterparty default intensity curve
    own_hazard           : our own default intensity curve
    xva_params           : simulation and funding configuration
    freq                 : fixed leg payment frequency

    Returns
    -------
    XVAResult with cva, dva, fva, total_xva, epe_profile, cva_by_period
    """
    # Build exposure profile
    profile = compute_epe_profile(
        curve       = curve,
        g2pp_params = g2pp_params,
        maturity    = maturity,
        fixed_rate  = fixed_rate,
        notional    = notional,
        pay_fixed   = pay_fixed,
        xva_params  = xva_params,
        freq        = freq,
    )

    # Compute top-level adjustments
    cva_val = cva(curve, counterparty_hazard, profile)
    dva_val = dva(curve, own_hazard, profile)
    fva_val = fva(curve, profile, xva_params)
    total   = cva_val - dva_val + fva_val

    # CVA decomposition by time bucket
    R        = counterparty_hazard.recovery
    times    = profile.times
    epe_arr  = profile.epe
    n_buckets = xva_params.n_steps
    cva_by_period = np.zeros(n_buckets)

    t_prev = 0.0
    for i in range(1, len(times)):
        t_i   = float(times[i])
        df_i  = float(curve.df(t_i))
        epe_i = float(epe_arr[i])
        pd_i  = counterparty_hazard.default_prob(t_prev, t_i)
        cva_by_period[i - 1] = (1.0 - R) * df_i * epe_i * pd_i
        t_prev = t_i

    return XVAResult(
        cva           = cva_val,
        dva           = dva_val,
        fva           = fva_val,
        total_xva     = total,
        epe_profile   = profile,
        cva_by_period = cva_by_period,
    )


def cva_sensitivity(
    curve:        DiscountCurve,
    g2pp_params:  G2ppParams,
    maturity:     float,
    fixed_rate:   float,
    notional:     float,
    pay_fixed:    bool,
    hazard_curve: HazardRateCurve,
    xva_params:   XVAParams,
    freq:         int   = 1,
    bump_bps:     float = 1.0,
) -> float:
    """
    CS01: sensitivity of CVA to a parallel +1bp shift in counterparty hazard rates.

    Uses bump-and-revalue with central finite differences.  The EPE profile is
    computed once (it is interest-rate driven, not credit driven) and reused for
    both bumped curves, making this efficient.

    Parameters
    ----------
    curve        : nominal discount curve
    g2pp_params  : G2++ model parameters
    maturity     : swap maturity in years
    fixed_rate   : swap fixed rate
    notional     : swap notional
    pay_fixed    : True = pay fixed
    hazard_curve : counterparty hazard rate curve (flat or bootstrapped)
    xva_params   : simulation parameters
    freq         : fixed leg payment frequency
    bump_bps     : hazard rate bump size in basis points (default 1bp)

    Returns
    -------
    CS01 in same units as CVA (e.g. currency). Positive = CVA rises as
    counterparty spread widens.
    """
    if bump_bps <= 0:
        raise ValueError("bump_bps must be positive")

    bump = bump_bps * 1e-4  # convert bps → decimal

    # Build bumped hazard curves
    h_up = HazardRateCurve(
        times    = hazard_curve.times.copy(),
        hazards  = hazard_curve.hazards + bump,
        recovery = hazard_curve.recovery,
    )
    h_dn = HazardRateCurve(
        times    = hazard_curve.times.copy(),
        hazards  = np.maximum(hazard_curve.hazards - bump, 0.0),
        recovery = hazard_curve.recovery,
    )

    # Compute EPE profile once — it does not depend on credit curve
    profile = compute_epe_profile(
        curve       = curve,
        g2pp_params = g2pp_params,
        maturity    = maturity,
        fixed_rate  = fixed_rate,
        notional    = notional,
        pay_fixed   = pay_fixed,
        xva_params  = xva_params,
        freq        = freq,
    )

    cva_up = cva(curve, h_up, profile)
    cva_dn = cva(curve, h_dn, profile)

    return (cva_up - cva_dn) / 2.0
