"""
Convexity adjustment for SOFR futures vs. OIS forward rates.

The Problem
-----------
A SOFR futures contract is marked-to-market daily with immediate cash settlement
of gains/losses (variation margin). A forward rate agreement (FRA) settles only
at maturity. Because of this daily settlement, futures rates are systematically
HIGHER than OIS forward rates for the same period — this bias is the convexity
adjustment.

Hull-White 1-Factor Model (Analytical Solution)
------------------------------------------------
Under the Hull-White model for the short rate:
    dr = [θ(t) - a·r] dt + σ dW

The convexity adjustment from futures to forward rate is:

    CA(T₁, T₂) = ½ · σ² · B(T₁, T₂) · [B(0, T₁) - B(T₁, T₂)]   (simplified)

For the common approximation used by practitioners (mean-reversion a → 0):

    CA(T₁, T₂) ≈ ½ · σ² · T₁ · T₂

where:
    T₁ = futures contract expiry (years from today)
    T₂ = end of futures accrual period (years from today)
    σ  = short-rate volatility (annualized, basis points / √year expressed as decimal)

Typical values (2024 market):
    σ ≈ 0.008–0.015 (80–150 bps annual vol on short rates)
    Adjustment for 1Y contract: ½ × 0.010² × 1 × 1.25 ≈ 0.6 bps
    Adjustment for 2Y contract: ½ × 0.010² × 2 × 2.25 ≈ 2.3 bps

For near-term (< 6 months) contracts the adjustment is <1bp and often ignored.
For contracts beyond 18 months it becomes material.
"""
from __future__ import annotations
import numpy as np
from datetime import date


def hull_white_convexity_adjustment(
    t1: float,
    t2: float,
    sigma: float = 0.010,
    mean_reversion: float = 0.0,
) -> float:
    """
    Compute the Hull-White convexity adjustment (futures → forward).

    forward_rate = futures_rate - CA

    Parameters
    ----------
    t1            : futures expiry in years from pricing date
    t2            : end of futures accrual period in years from pricing date
    sigma         : short-rate volatility (decimal annualized, e.g. 0.010 = 1%)
    mean_reversion: Hull-White mean-reversion speed a (set to 0 for simpler formula)

    Returns
    -------
    Convexity adjustment in decimal (e.g. 0.00006 = 0.6 bps)
    """
    if t1 < 0 or t2 <= t1:
        return 0.0

    if abs(mean_reversion) < 1e-8:
        # Zero mean-reversion (Vasicek/HJM limit): CA = ½σ²T₁T₂
        return 0.5 * sigma**2 * t1 * t2
    else:
        # Full Hull-White: B(0,T) = (1 - e^{-aT}) / a
        # CA = ½σ²B(0,T1)B(0,T2)  [exact for HW; converges to ½σ²T1T2 as a→0]
        a = mean_reversion
        B_0_T1 = (1 - np.exp(-a * t1)) / a
        B_0_T2 = (1 - np.exp(-a * t2)) / a
        return 0.5 * sigma**2 * B_0_T1 * B_0_T2


def calibrate_sigma_from_caps(
    cap_vols: dict[float, float],
    mean_reversion: float = 0.0,
) -> float:
    """
    Rough calibration of σ from at-the-money cap/floor implied vols.
    cap_vols: {maturity_years: implied_normal_vol_decimal}
    Normal vol for cap ≈ σ (Hull-White, short maturities).
    Returns average σ estimate.
    """
    if not cap_vols:
        return 0.010  # default 1% if no data
    return float(np.mean(list(cap_vols.values())))


def adjustment_schedule(
    expiry_dates: list[date],
    accrual_end_dates: list[date],
    ref_date: date,
    sigma: float = 0.010,
    mean_reversion: float = 0.0,
) -> list[float]:
    """
    Compute convexity adjustments for a list of futures contracts.
    Returns a list of adjustments in decimal (same length as expiry_dates).
    """
    adjustments = []
    for exp, acc_end in zip(expiry_dates, accrual_end_dates):
        t1 = (exp - ref_date).days / 365.25
        t2 = (acc_end - ref_date).days / 365.25
        ca = hull_white_convexity_adjustment(t1, t2, sigma, mean_reversion)
        adjustments.append(ca)
    return adjustments


def futures_to_forward(
    futures_rate: float,
    t1: float,
    t2: float,
    sigma: float = 0.010,
    mean_reversion: float = 0.0,
) -> float:
    """
    Convert a futures-implied rate to an OIS-consistent forward rate.
    forward_rate = futures_rate - CA(T₁, T₂)
    """
    ca = hull_white_convexity_adjustment(t1, t2, sigma, mean_reversion)
    return futures_rate - ca


if __name__ == "__main__":
    # Illustrative table of convexity adjustments for SR3 contracts
    print("Hull-White Convexity Adjustment Schedule (σ=1.0%, a=0)")
    print(f"{'Contract':<12} {'T1 (yr)':>8} {'T2 (yr)':>8} {'CA (bps)':>10}")
    print("-" * 45)
    contracts = [
        ("SR3 Jun-25", 0.50, 0.75),
        ("SR3 Sep-25", 0.75, 1.00),
        ("SR3 Dec-25", 1.00, 1.25),
        ("SR3 Mar-26", 1.25, 1.50),
        ("SR3 Jun-26", 1.50, 1.75),
        ("SR3 Sep-26", 1.75, 2.00),
        ("SR3 Dec-26", 2.00, 2.25),
        ("SR3 Mar-27", 2.25, 2.50),
    ]
    for name, t1, t2 in contracts:
        ca_bps = hull_white_convexity_adjustment(t1, t2, sigma=0.010) * 10_000
        print(f"{name:<12} {t1:>8.2f} {t2:>8.2f} {ca_bps:>10.2f}")
