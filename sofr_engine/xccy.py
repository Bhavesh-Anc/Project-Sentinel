"""
Cross-Currency Basis Swap framework for USD SOFR vs EUR €STR/EURIBOR.

Implements:
  - FX forward pricing via Covered Interest Parity (CIP)
  - Par basis computation for floating-floating XCCY swaps
  - Mark-to-market (PV) of XCCY basis swaps
  - DV01 (USD curve sensitivity) and CS01 (basis sensitivity)
  - EUR discount curve implied from market FX forwards
  - Basis term structure across a range of maturities

Financial conventions
---------------------
  - Spot FX: S = USD per EUR (e.g. 1.08 means 1 EUR = 1.08 USD)
  - CIP:     F(0,T) = S × P_EUR(0,T) / P_USD(0,T)
  - Par basis b satisfies: swap NPV = 0 at initiation
    b_par = [P_EUR(T) - P_USD(T)] / Σ_{k=1}^{N} alpha_k × P_EUR(T_k)
  - Floating-floating XCCY: USD pays SOFR, EUR receives EURIBOR + b
  - MTM to USD payer: PV = N_USD × b × Σ alpha_k × P_EUR(T_k)
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from datetime import date
from typing import Sequence

import numpy as np

from .curve import DiscountCurve


# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------

@dataclass
class FXForwardCurve:
    """
    FX forward curve built from spot rate + Covered Interest Parity (CIP).

    Convention: spot_fx = USD per EUR (e.g. 1.08 means 1 EUR = 1.08 USD).
    Forward rate F(0,T) = spot_fx × P_EUR(0,T) / P_USD(0,T).
    """

    spot_fx: float          # S(0) in USD/EUR
    usd_curve: DiscountCurve
    eur_curve: DiscountCurve

    def __post_init__(self) -> None:
        if self.spot_fx <= 0.0:
            raise ValueError(f"spot_fx must be positive, got {self.spot_fx}")

    def forward_fx(self, T: float) -> float:
        """
        CIP forward FX rate at tenor T (years).

        F(0,T) = S × P_EUR(0,T) / P_USD(0,T)
        """
        if T < 0.0:
            raise ValueError(f"T must be non-negative, got {T}")
        if T < 1e-10:
            return self.spot_fx
        p_eur = float(self.eur_curve.df(T))
        p_usd = float(self.usd_curve.df(T))
        return self.spot_fx * p_eur / p_usd

    def implied_basis(self, T: float, mkt_forward: float) -> float:
        """
        Annualised continuously-compounded basis implied by a market forward.

        basis = log(F_mkt / F_cip) / T

        A positive value means the market forward is higher than CIP predicts
        (EUR is 'expensive' in the forward market relative to rate parity).
        """
        if T <= 0.0:
            raise ValueError(f"T must be positive for implied_basis, got {T}")
        if mkt_forward <= 0.0:
            raise ValueError(f"mkt_forward must be positive, got {mkt_forward}")
        f_cip = self.forward_fx(T)
        return math.log(mkt_forward / f_cip) / T

    def usd_from_eur(self, eur_amount: float, T: float) -> float:
        """
        Convert a EUR cash flow at time T to USD using the CIP forward rate.

        USD amount = eur_amount × F(0,T)
        """
        return eur_amount * self.forward_fx(T)


@dataclass
class CrossCurrencySwap:
    """
    Floating-floating XCCY basis swap (USD SOFR vs EUR €STR + basis spread).

    The USD leg pays SOFR flat (floating at par).
    The EUR leg pays EURIBOR + basis_spread (floating at par + basis annuity).

    Notional exchange is assumed at maturity (standard XCCY convention).
    """

    maturity_years: float
    notional_usd: float = 10_000_000.0
    basis_spread: float = 0.0        # b in decimal (e.g. -0.0010 = -10 bps)
    freq: int = 4                    # payment frequency (4 = quarterly)
    pay_usd: bool = True             # True → we pay SOFR leg, receive EUR + basis

    def __post_init__(self) -> None:
        if self.maturity_years <= 0.0:
            raise ValueError(
                f"maturity_years must be positive, got {self.maturity_years}"
            )
        if self.notional_usd == 0.0:
            raise ValueError("notional_usd must not be zero")
        if self.freq <= 0:
            raise ValueError(f"freq must be a positive integer, got {self.freq}")


@dataclass
class XCCYResult:
    """
    Mark-to-market result for a cross-currency basis swap.

    All monetary values in USD.
    """

    pv_usd: float                # total MTM of the swap in USD
    usd_leg_pv: float            # PV of the USD SOFR floating leg
    eur_leg_pv_in_usd: float     # PV of the EUR (EURIBOR + basis) leg, converted to USD
    par_basis_bps: float         # par basis spread in basis points
    notional_eur: float          # EUR notional = N_USD / spot_fx
    eur_annuity_in_usd: float    # Σ alpha_k × P_EUR(T_k) × spot_fx  (annuity in USD)
    spot_fx: float               # spot FX rate (USD per EUR) used in pricing


# ---------------------------------------------------------------------------
# Core functions
# ---------------------------------------------------------------------------

def fx_forward(
    usd_curve: DiscountCurve,
    eur_curve: DiscountCurve,
    spot_fx: float,
    T: float,
) -> float:
    """
    Compute the CIP forward FX rate at tenor T.

    F(0,T) = S × P_EUR(0,T) / P_USD(0,T)

    Parameters
    ----------
    usd_curve : USD SOFR discount curve
    eur_curve : EUR €STR/EURIBOR discount curve
    spot_fx   : spot rate S (USD per EUR, e.g. 1.08)
    T         : tenor in years

    Returns
    -------
    Forward FX rate in USD per EUR
    """
    if spot_fx <= 0.0:
        raise ValueError(f"spot_fx must be positive, got {spot_fx}")
    if T < 0.0:
        raise ValueError(f"T must be non-negative, got {T}")
    if T < 1e-10:
        return spot_fx
    p_usd = float(usd_curve.df(T))
    p_eur = float(eur_curve.df(T))
    return spot_fx * p_eur / p_usd


def xccy_par_basis(
    usd_curve: DiscountCurve,
    eur_curve: DiscountCurve,
    spot_fx: float,
    maturity: float,
    freq: int = 4,
) -> float:
    """
    Compute the par basis spread for a floating-floating XCCY swap.

    The par basis b is the spread added to the EUR floating leg such that the
    swap has zero NPV at initiation:

        b_par = (P_EUR(T) - P_USD(T)) / Σ_{k=1}^{N} alpha_k × P_EUR(T_k)

    where T_k = k / freq, k = 1 … N, N = maturity × freq, alpha_k = 1/freq.

    Parameters
    ----------
    usd_curve : USD discount curve
    eur_curve : EUR discount curve
    spot_fx   : spot FX rate (not used in the formula but kept for consistency)
    maturity  : swap maturity in years
    freq      : payment frequency per year (default 4 = quarterly)

    Returns
    -------
    Par basis spread (decimal, e.g. -0.0015 = -15 bps)
    """
    if maturity <= 0.0:
        raise ValueError(f"maturity must be positive, got {maturity}")
    if freq <= 0:
        raise ValueError(f"freq must be a positive integer, got {freq}")
    if spot_fx <= 0.0:
        raise ValueError(f"spot_fx must be positive, got {spot_fx}")

    dt = 1.0 / freq
    payment_times = np.arange(dt, maturity + 1e-10, dt)

    # EUR annuity: Σ alpha_k × P_EUR(T_k)
    eur_annuity = float(np.sum(dt * eur_curve.df(payment_times)))

    if eur_annuity < 1e-14:
        raise ValueError("EUR annuity is effectively zero — check curve and maturity")

    p_eur_T = float(eur_curve.df(maturity))
    p_usd_T = float(usd_curve.df(maturity))

    return (p_eur_T - p_usd_T) / eur_annuity


def xccy_swap_pv(
    usd_curve: DiscountCurve,
    eur_curve: DiscountCurve,
    spot_fx: float,
    swap: CrossCurrencySwap,
) -> XCCYResult:
    """
    Mark-to-market of a floating-floating XCCY basis swap.

    Pricing logic (both legs float at par in their respective currencies):
      - USD floating leg PV (with notional exchange) = N_USD
      - EUR floating leg PV in EUR (with notional exchange) = N_EUR = N_USD / S
      - EUR leg PV in USD via CIP: N_EUR × S = N_USD  (CIP-neutral component)
      - Extra value from basis spread:
          basis_pv = N_USD × b × Σ alpha_k × P_EUR(T_k)

    For pay_usd=True (we pay SOFR, receive EUR + basis):
      PV = EUR_leg_in_USD - USD_leg = N_USD × b × eur_annuity

    Parameters
    ----------
    usd_curve : USD SOFR discount curve
    eur_curve : EUR €STR/EURIBOR discount curve
    spot_fx   : spot FX rate (USD per EUR)
    swap      : CrossCurrencySwap instance

    Returns
    -------
    XCCYResult with full PV breakdown
    """
    if spot_fx <= 0.0:
        raise ValueError(f"spot_fx must be positive, got {spot_fx}")

    T = swap.maturity_years
    N = swap.notional_usd
    b = swap.basis_spread
    dt = 1.0 / swap.freq

    payment_times = np.arange(dt, T + 1e-10, dt)

    # EUR annuity in EUR units
    eur_annuity_eur = float(np.sum(dt * eur_curve.df(payment_times)))

    # EUR annuity converted to USD at spot (reflects USD value of EUR coupon stream)
    eur_annuity_usd = eur_annuity_eur * spot_fx

    # Both floating legs price at par (plus notional) in their own currency.
    # Converted to a common USD basis they are equal under CIP.
    usd_leg_pv = N                                            # USD SOFR leg at par
    notional_eur = N / spot_fx
    eur_leg_pv_usd = N + N * b * eur_annuity_eur              # par + basis annuity

    # Directional sign: pay_usd means we pay USD leg, receive EUR leg
    if swap.pay_usd:
        pv_usd = eur_leg_pv_usd - usd_leg_pv   # = N × b × eur_annuity_eur
    else:
        pv_usd = usd_leg_pv - eur_leg_pv_usd   # = -N × b × eur_annuity_eur

    par_b = xccy_par_basis(usd_curve, eur_curve, spot_fx, T, swap.freq)

    return XCCYResult(
        pv_usd=pv_usd,
        usd_leg_pv=usd_leg_pv,
        eur_leg_pv_in_usd=eur_leg_pv_usd,
        par_basis_bps=par_b * 10_000.0,
        notional_eur=notional_eur,
        eur_annuity_in_usd=eur_annuity_usd,
        spot_fx=spot_fx,
    )


def xccy_dv01(
    usd_curve: DiscountCurve,
    eur_curve: DiscountCurve,
    spot_fx: float,
    swap: CrossCurrencySwap,
) -> float:
    """
    DV01 of the XCCY swap PV with respect to a parallel shift in the USD curve.

    Computed via bump-and-revalue: shift all USD discount factors by -1 bp
    (i.e. P_USD_bumped(T) = P_USD(T) × exp(-0.0001 × T)) and reprice.

    DV01 = PV_bumped - PV_base

    A positive DV01 means the position gains value when USD rates fall (rates
    fall → DFs rise → swap value increases for the USD payer who is implicitly
    long the EUR annuity / short USD duration).

    Parameters
    ----------
    usd_curve : USD discount curve
    eur_curve : EUR discount curve
    spot_fx   : spot FX rate
    swap      : CrossCurrencySwap

    Returns
    -------
    DV01 in USD (value change per 1 bp parallel fall in USD rates)
    """
    if spot_fx <= 0.0:
        raise ValueError(f"spot_fx must be positive, got {spot_fx}")

    # Base PV
    base_result = xccy_swap_pv(usd_curve, eur_curve, spot_fx, swap)
    pv_base = base_result.pv_usd

    # Build bumped USD curve: bump all pillars by -1bp
    bump = 0.0001
    times = usd_curve._times.copy()
    log_dfs_bumped = usd_curve._log_df - bump * times   # log(DF) - bump × T
    dfs_bumped = np.exp(log_dfs_bumped)

    # Reconstruct — skip T=0 (DF=1 always) for the DiscountCurve constructor
    # The constructor prepends 0 if the first time is > 0; we must handle T=0 ourselves.
    mask = times > 1e-10
    t_pillars = times[mask]
    df_pillars = dfs_bumped[mask]

    if len(t_pillars) == 0:
        # degenerate: only T=0 pillar
        return 0.0

    bumped_usd = DiscountCurve(
        ref_date=usd_curve.ref_date,
        times=t_pillars,
        dfs=df_pillars,
        label=usd_curve.label + "_bumped",
    )

    bumped_result = xccy_swap_pv(bumped_usd, eur_curve, spot_fx, swap)
    return bumped_result.pv_usd - pv_base


def xccy_cs01(
    usd_curve: DiscountCurve,
    eur_curve: DiscountCurve,
    spot_fx: float,
    swap: CrossCurrencySwap,
) -> float:
    """
    CS01 ("Basis-01"): sensitivity of swap PV to a 1 bp change in basis spread.

    CS01 = ∂PV/∂b × 0.0001
         = N_USD × 0.0001 × Σ alpha_k × P_EUR(T_k)   (for pay_usd=True)

    This is the dollar value of a 1 bp move in the basis spread.
    Positive for pay_usd=True (receiving the basis, so wider basis → higher PV).
    Negative for pay_usd=False.

    Parameters
    ----------
    usd_curve : USD discount curve
    eur_curve : EUR discount curve
    spot_fx   : spot FX rate
    swap      : CrossCurrencySwap

    Returns
    -------
    CS01 in USD
    """
    if spot_fx <= 0.0:
        raise ValueError(f"spot_fx must be positive, got {spot_fx}")

    T = swap.maturity_years
    dt = 1.0 / swap.freq
    payment_times = np.arange(dt, T + 1e-10, dt)
    eur_annuity = float(np.sum(dt * eur_curve.df(payment_times)))

    cs01 = swap.notional_usd * 0.0001 * eur_annuity
    return cs01 if swap.pay_usd else -cs01


def build_eur_curve_from_xccy(
    usd_curve: DiscountCurve,
    spot_fx: float,
    fx_forwards: Sequence[float],
    maturities: Sequence[float],
    ref_date: date,
) -> DiscountCurve:
    """
    Imply a EUR discount curve from market FX forward quotes using CIP.

    Rearranging CIP:  F(0,T) = S × P_EUR(T) / P_USD(T)
    →  P_EUR(T) = F(0,T) / S × P_USD(T)

    Parameters
    ----------
    usd_curve   : USD SOFR discount curve
    spot_fx     : spot FX rate S (USD per EUR)
    fx_forwards : market FX forward rates F(0,T_k) [USD per EUR] at each maturity
    maturities  : tenor points T_k (years) corresponding to each forward
    ref_date    : curve reference date

    Returns
    -------
    DiscountCurve labelled "EUR (CIP-implied)"
    """
    if spot_fx <= 0.0:
        raise ValueError(f"spot_fx must be positive, got {spot_fx}")

    mats = np.asarray(maturities, dtype=float)
    fwds = np.asarray(fx_forwards, dtype=float)

    if len(mats) != len(fwds):
        raise ValueError(
            f"maturities and fx_forwards must have the same length, "
            f"got {len(mats)} and {len(fwds)}"
        )
    if np.any(mats <= 0.0):
        raise ValueError("All maturities must be positive")
    if np.any(fwds <= 0.0):
        raise ValueError("All FX forwards must be positive")

    # P_EUR(T_k) = F(0,T_k) / S × P_USD(T_k)
    p_usd = np.array([float(usd_curve.df(t)) for t in mats])
    dfs_eur = fwds / spot_fx * p_usd

    if np.any(dfs_eur <= 0.0):
        raise ValueError(
            "Implied EUR discount factors are non-positive — check FX forward inputs"
        )

    return DiscountCurve(
        ref_date=ref_date,
        times=mats,
        dfs=dfs_eur,
        label="EUR (CIP-implied)",
    )


def xccy_basis_term_structure(
    usd_curve: DiscountCurve,
    eur_curve: DiscountCurve,
    spot_fx: float,
    maturities: Sequence[float],
    freq: int = 4,
) -> list[dict]:
    """
    Compute the par basis spread at each maturity and return a term structure.

    Parameters
    ----------
    usd_curve  : USD discount curve
    eur_curve  : EUR discount curve
    spot_fx    : spot FX rate (USD per EUR)
    maturities : list/array of tenor points (years)
    freq       : payment frequency (default 4 = quarterly)

    Returns
    -------
    List of dicts, each with keys:
      "maturity"      : float  (years)
      "par_basis_bps" : float  (basis points)
    """
    if spot_fx <= 0.0:
        raise ValueError(f"spot_fx must be positive, got {spot_fx}")
    if freq <= 0:
        raise ValueError(f"freq must be a positive integer, got {freq}")

    results = []
    for T in maturities:
        T = float(T)
        if T <= 0.0:
            raise ValueError(f"All maturities must be positive, got {T}")
        b = xccy_par_basis(usd_curve, eur_curve, spot_fx, T, freq)
        results.append({"maturity": T, "par_basis_bps": b * 10_000.0})
    return results
