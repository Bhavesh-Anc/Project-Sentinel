"""
FX Options Pricing — Garman-Kohlhagen with Vanna-Volga Vol Surface.

Provides:
  - FXOptionParams   : GK model inputs
  - FXOptionResult   : price + full Greeks
  - gk_price         : Garman-Kohlhagen call/put price
  - gk_greeks        : full Greeks (delta, gamma, vega, theta, rho, vanna, volga)
  - gk_implied_vol   : Newton-Raphson implied vol inversion
  - FXVolSurface     : ATM + 25Δ RR/BF → smile construction
  - vol_for_strike   : Vanna-Volga interpolated vol
  - fx_smile         : full smile array
  - FXVolCone        : realized vol cone from return series

References
----------
Garman, M. & Kohlhagen, S. (1983). Foreign currency option values.
  Journal of International Money and Finance, 2(3), 231–237.
Castagna, A. & Mercurio, F. (2007). The Vanna-Volga method for implied
  volatilities. Risk, 20(1), 106–111.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import List, Tuple

import numpy as np
from scipy.optimize import brentq
from scipy.stats import norm


# ── Core model ────────────────────────────────────────────────────────────────

@dataclass
class FXOptionParams:
    """
    Garman-Kohlhagen model inputs for a single FX option.

    Convention: spot = S in domestic/foreign (e.g. USD/EUR means 1 EUR = spot USD).

    Parameters
    ----------
    spot            : S₀ (e.g. 1.09 USD per EUR)
    strike          : K
    vol             : implied vol σ (annualised, decimal)
    domestic_rate   : r_d (continuous, e.g. USD SOFR)
    foreign_rate    : r_f (continuous, e.g. EUR €STR)
    maturity        : T in years
    is_call         : True = call on foreign currency (long EUR), False = put
    """
    spot:          float
    strike:        float
    vol:           float
    domestic_rate: float
    foreign_rate:  float
    maturity:      float
    is_call:       bool = True

    def __post_init__(self) -> None:
        if self.spot <= 0:
            raise ValueError(f"spot must be positive, got {self.spot}")
        if self.strike <= 0:
            raise ValueError(f"strike must be positive, got {self.strike}")
        if self.vol < 0:
            raise ValueError(f"vol must be non-negative, got {self.vol}")
        if self.maturity < 0:
            raise ValueError(f"maturity must be non-negative, got {self.maturity}")


@dataclass
class FXOptionResult:
    """Full Garman-Kohlhagen option result with Greeks."""
    pv:     float
    delta:  float   # ∂V/∂S
    gamma:  float   # ∂²V/∂S²
    vega:   float   # ∂V/∂σ (per unit vol, not per 1%)
    theta:  float   # ∂V/∂t (daily)
    rho_d:  float   # ∂V/∂r_d
    rho_f:  float   # ∂V/∂r_f
    vanna:  float   # ∂²V/∂S∂σ
    volga:  float   # ∂²V/∂σ²


def _d1d2(p: FXOptionParams) -> Tuple[float, float]:
    """Compute d1 and d2 for GK formula."""
    if p.maturity < 1e-10 or p.vol < 1e-10:
        return math.inf, math.inf
    sqrtT = math.sqrt(p.maturity)
    d1 = (math.log(p.spot / p.strike)
          + (p.domestic_rate - p.foreign_rate + 0.5 * p.vol ** 2) * p.maturity
          ) / (p.vol * sqrtT)
    d2 = d1 - p.vol * sqrtT
    return d1, d2


def gk_price(p: FXOptionParams) -> float:
    """
    Garman-Kohlhagen call or put price.

    Call:  S·exp(-r_f·T)·N(d1) - K·exp(-r_d·T)·N(d2)
    Put:   K·exp(-r_d·T)·N(-d2) - S·exp(-r_f·T)·N(-d1)
    """
    S  = p.spot
    K  = p.strike
    rd = p.domestic_rate
    rf = p.foreign_rate
    T  = p.maturity

    # Intrinsic at expiry
    if T < 1e-10:
        if p.is_call:
            return max(S - K, 0.0)
        return max(K - S, 0.0)

    if p.vol < 1e-10:
        df_d = math.exp(-rd * T)
        df_f = math.exp(-rf * T)
        fwd  = S * df_f / df_d
        if p.is_call:
            return max(fwd - K, 0.0) * df_d
        return max(K - fwd, 0.0) * df_d

    d1, d2 = _d1d2(p)
    df_d = math.exp(-rd * T)
    df_f = math.exp(-rf * T)

    if p.is_call:
        return S * df_f * norm.cdf(d1) - K * df_d * norm.cdf(d2)
    return K * df_d * norm.cdf(-d2) - S * df_f * norm.cdf(-d1)


def gk_greeks(p: FXOptionParams) -> FXOptionResult:
    """
    Full Garman-Kohlhagen price and Greeks.
    """
    pv = gk_price(p)

    if p.maturity < 1e-10 or p.vol < 1e-10:
        return FXOptionResult(pv=pv, delta=float(p.is_call) - (0.0 if p.is_call else 0.0),
                              gamma=0.0, vega=0.0, theta=0.0,
                              rho_d=0.0, rho_f=0.0, vanna=0.0, volga=0.0)

    S  = p.spot
    K  = p.strike
    rd = p.domestic_rate
    rf = p.foreign_rate
    T  = p.maturity
    sig = p.vol
    sqrtT = math.sqrt(T)

    d1, d2 = _d1d2(p)
    df_d = math.exp(-rd * T)
    df_f = math.exp(-rf * T)
    phi1 = norm.pdf(d1)  # φ(d1)

    # Greeks
    if p.is_call:
        delta = df_f * norm.cdf(d1)
        theta = (- S * df_f * phi1 * sig / (2.0 * sqrtT)
                 - rd * K * df_d * norm.cdf(d2)
                 + rf * S * df_f * norm.cdf(d1)) / 365.0
        rho_d = K * T * df_d * norm.cdf(d2)
        rho_f = -S * T * df_f * norm.cdf(d1)
    else:
        delta = -df_f * norm.cdf(-d1)
        theta = (- S * df_f * phi1 * sig / (2.0 * sqrtT)
                 + rd * K * df_d * norm.cdf(-d2)
                 - rf * S * df_f * norm.cdf(-d1)) / 365.0
        rho_d = -K * T * df_d * norm.cdf(-d2)
        rho_f = S * T * df_f * norm.cdf(-d1)

    gamma = df_f * phi1 / (S * sig * sqrtT)
    vega  = S * df_f * phi1 * sqrtT        # per unit vol
    vanna = -df_f * phi1 * d2 / sig
    volga = S * df_f * phi1 * sqrtT * d1 * d2 / sig

    return FXOptionResult(
        pv=pv, delta=delta, gamma=gamma, vega=vega, theta=theta,
        rho_d=rho_d, rho_f=rho_f, vanna=vanna, volga=volga,
    )


def gk_implied_vol(
    market_price: float,
    spot:         float,
    strike:       float,
    domestic_rate: float,
    foreign_rate:  float,
    maturity:     float,
    is_call:      bool = True,
    tol:          float = 1e-8,
) -> float:
    """
    Implied vol via Newton-Raphson (Brenner-Subrahmanyam initial guess).

    Raises ValueError if no solution found in [1e-6, 10.0].
    """
    if maturity < 1e-10:
        raise ValueError("maturity must be positive for IV calculation")

    def price_at_vol(v: float) -> float:
        p = FXOptionParams(
            spot=spot, strike=strike, vol=v,
            domestic_rate=domestic_rate, foreign_rate=foreign_rate,
            maturity=maturity, is_call=is_call,
        )
        return gk_price(p)

    # Intrinsic lower bound
    df_d = math.exp(-domestic_rate * maturity)
    df_f = math.exp(-foreign_rate  * maturity)
    fwd  = spot * df_f / df_d
    intrinsic = max(fwd - strike, 0.0) * df_d if is_call else max(strike - fwd, 0.0) * df_d

    if market_price <= intrinsic + 1e-12:
        # Price at intrinsic → vol approaches zero
        if market_price < intrinsic - 1e-6:
            raise ValueError(f"market_price {market_price} below intrinsic {intrinsic}")
        return 1e-6

    # Brenner-Subrahmanyam initial guess: σ₀ ≈ √(2π/T) * C/S
    sig0 = math.sqrt(2.0 * math.pi / maturity) * market_price / spot
    sig0 = max(0.001, min(5.0, sig0))

    # Newton-Raphson
    sig = sig0
    for _ in range(100):
        p = FXOptionParams(
            spot=spot, strike=strike, vol=sig,
            domestic_rate=domestic_rate, foreign_rate=foreign_rate,
            maturity=maturity, is_call=is_call,
        )
        price = gk_price(p)
        sqrtT = math.sqrt(maturity)
        d1, _ = _d1d2(p)
        vega  = spot * math.exp(-foreign_rate * maturity) * norm.pdf(d1) * sqrtT
        diff  = price - market_price
        if abs(diff) < tol:
            return sig
        if abs(vega) < 1e-12:
            break
        sig -= diff / vega
        sig = max(1e-6, min(10.0, sig))

    # Fallback: brentq
    try:
        return brentq(lambda v: price_at_vol(v) - market_price, 1e-6, 10.0, xtol=tol)
    except ValueError:
        raise ValueError(f"Cannot find implied vol for market_price={market_price}")


# ── Vol surface ───────────────────────────────────────────────────────────────

@dataclass
class FXVolSurface:
    """
    FX Vol Surface from standard market quotes.

    Market convention uses 3 quotes per expiry:
      atm_vol : ATM straddle vol
      rr25    : 25Δ risk reversal = σ_25c - σ_25p
      bf25    : 25Δ butterfly = (σ_25c + σ_25p)/2 - σ_atm

    Parameters
    ----------
    maturities : list of tenors in years [T1, T2, …]
    atm_vols   : ATM vol at each tenor
    rr25       : 25Δ risk reversal at each tenor
    bf25       : 25Δ butterfly at each tenor
    spot       : current spot FX rate
    domestic_rate : domestic (USD) rate
    foreign_rate  : foreign (EUR) rate
    """
    maturities:    List[float]
    atm_vols:      List[float]
    rr25:          List[float]
    bf25:          List[float]
    spot:          float
    domestic_rate: float = 0.04
    foreign_rate:  float = 0.03

    def __post_init__(self) -> None:
        n = len(self.maturities)
        if not (len(self.atm_vols) == len(self.rr25) == len(self.bf25) == n):
            raise ValueError("maturities, atm_vols, rr25, bf25 must have the same length")
        if n == 0:
            raise ValueError("FXVolSurface must have at least one maturity")

    def _smile_vols(self, i: int) -> Tuple[float, float, float]:
        """Return (σ_25p, σ_atm, σ_25c) for maturity index i."""
        atm = self.atm_vols[i]
        rr  = self.rr25[i]
        bf  = self.bf25[i]
        sig_25c = atm + bf + rr / 2.0
        sig_25p = atm + bf - rr / 2.0
        return sig_25p, atm, sig_25c

    def _strike_from_delta(self, delta: float, vol: float, T: float, is_call: bool) -> float:
        """Find strike K such that GK delta(K) = delta."""
        S  = self.spot
        rd = self.domestic_rate
        rf = self.foreign_rate
        df_f = math.exp(-rf * T)
        sqrtT = math.sqrt(T)

        # For a call with spot-delta Δ_c = exp(-rf*T)*N(d1):
        #   N(d1) = Δ_c / exp(-rf*T)  → d1 = norm.ppf(Δ_c / df_f)
        #
        # For a put with |spot-delta| = delta (positive convention):
        #   Δ_p = -exp(-rf*T)*N(-d1) = -delta
        #   N(-d1) = delta/df_f  → -d1 = norm.ppf(delta/df_f) → d1 = -norm.ppf(delta/df_f)
        n_ppf = norm.ppf(delta / df_f)  # norm.ppf(N(d1)) for a call
        if is_call:
            d1_target = n_ppf
        else:
            d1_target = -n_ppf  # put: flip sign

        # From d1 = (ln(S/K) + (rd-rf+σ²/2)T)/(σ√T):
        # ln(S/K) = d1*σ√T - (rd-rf+σ²/2)*T
        # K = S * exp(-(d1*σ√T - (rd-rf+σ²/2)*T))
        log_ratio = d1_target * vol * sqrtT - (rd - rf + 0.5 * vol ** 2) * T
        return S * math.exp(-log_ratio)

    def pillar_strikes(self, i: int) -> Tuple[float, float, float]:
        """
        Return (K_25p, K_atm, K_25c) for maturity index i.

        K_atm: delta-neutral straddle K = S * exp((r_d - r_f + σ²/2) * T)  [approximately]
        """
        T   = self.maturities[i]
        sig_25p, sig_atm, sig_25c = self._smile_vols(i)
        rd  = self.domestic_rate
        rf  = self.foreign_rate
        S   = self.spot

        # ATM: forward (ATMF convention)
        k_atm = S * math.exp((rd - rf) * T)

        k_25c = self._strike_from_delta(0.25, sig_25c, T, is_call=True)
        k_25p = self._strike_from_delta(0.25, sig_25p, T, is_call=False)

        return k_25p, k_atm, k_25c


def vol_for_strike(surface: FXVolSurface, maturity: float, strike: float) -> float:
    """
    Vanna-Volga interpolated vol at (maturity, strike).

    Uses quadratic interpolation across (K_25p, K_atm, K_25c) anchor vols,
    then linearly interpolates between the two nearest maturity pillars.
    """
    mats = surface.maturities
    n = len(mats)

    # Find bracketing maturities
    if maturity <= mats[0]:
        idx_lo, idx_hi, w = 0, 0, 0.0
    elif maturity >= mats[-1]:
        idx_lo, idx_hi, w = n - 1, n - 1, 0.0
    else:
        for k in range(n - 1):
            if mats[k] <= maturity <= mats[k + 1]:
                idx_lo, idx_hi = k, k + 1
                w = (maturity - mats[k]) / (mats[k + 1] - mats[k])
                break
        else:
            idx_lo, idx_hi, w = n - 2, n - 1, 1.0

    def interp_vol_at_idx(i: int) -> float:
        k25p, katm, k25c = surface.pillar_strikes(i)
        sig25p, sigatm, sig25c = surface._smile_vols(i)

        # Quadratic interpolation: σ(K) = a + b*K + c*K²
        # passing through (K_25p, sig_25p), (K_atm, sig_atm), (K_25c, sig_25c)
        pts = np.array([k25p, katm, k25c])
        vls = np.array([sig25p, sigatm, sig25c])
        coeffs = np.polyfit(pts, vls, 2)
        vol_k = np.polyval(coeffs, strike)
        # Clamp to reasonable range
        return float(max(0.001, min(2.0, vol_k)))

    v_lo = interp_vol_at_idx(idx_lo)
    if idx_lo == idx_hi:
        return v_lo
    v_hi = interp_vol_at_idx(idx_hi)
    return (1.0 - w) * v_lo + w * v_hi


def fx_smile(
    surface:  FXVolSurface,
    maturity: float,
    n_strikes: int = 21,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Return (strikes, vols) arrays for the smile at a given maturity.

    Strikes span from 0.7 × ATM-fwd to 1.3 × ATM-fwd.
    """
    rd = surface.domestic_rate
    rf = surface.foreign_rate
    T  = maturity
    fwd = surface.spot * math.exp((rd - rf) * T)

    strikes = np.linspace(0.7 * fwd, 1.3 * fwd, n_strikes)
    vols    = np.array([vol_for_strike(surface, maturity, k) for k in strikes])
    return strikes, vols


# ── Vol cone ─────────────────────────────────────────────────────────────────

@dataclass
class FXVolCone:
    """
    Historical realized vol cone.

    Attributes
    ----------
    tenors      : window lengths in trading days
    percentiles : array shape (5, n_tenors) — rows = [10, 25, 50, 75, 90] pctile
    current     : most recent realized vol at each tenor
    """
    tenors:      List[int]       # window lengths in trading days
    percentiles: np.ndarray      # (5, n_tenors)
    current:     np.ndarray      # (n_tenors,)


def vol_cone(
    log_returns: np.ndarray,
    tenors:      List[int] | None = None,
) -> FXVolCone:
    """
    Compute the historical realized vol cone from a daily log-return series.

    Parameters
    ----------
    log_returns : 1-D array of daily log-returns (e.g. np.log(S[1:]/S[:-1]))
    tenors      : rolling window sizes in trading days (default [5,10,21,42,63,126,252])

    Returns
    -------
    FXVolCone with annualised vol percentiles per tenor
    """
    if tenors is None:
        tenors = [5, 10, 21, 42, 63, 126, 252]

    pctile_rows = []
    current_row = []

    for w in tenors:
        # Rolling std × sqrt(252) = annualised vol
        rolling_vols = np.array([
            np.std(log_returns[max(0, i - w):i], ddof=1) * math.sqrt(252)
            for i in range(w, len(log_returns) + 1)
        ])
        rolling_vols = rolling_vols[np.isfinite(rolling_vols)]
        if len(rolling_vols) == 0:
            rolling_vols = np.array([0.0])
        pctile_rows.append(np.percentile(rolling_vols, [10, 25, 50, 75, 90]))
        current_row.append(rolling_vols[-1] if len(rolling_vols) > 0 else 0.0)

    return FXVolCone(
        tenors=tenors,
        percentiles=np.array(pctile_rows).T,  # (5, n_tenors)
        current=np.array(current_row),
    )


__all__ = [
    "FXOptionParams",
    "FXOptionResult",
    "gk_price",
    "gk_greeks",
    "gk_implied_vol",
    "FXVolSurface",
    "vol_for_strike",
    "fx_smile",
    "FXVolCone",
    "vol_cone",
]
