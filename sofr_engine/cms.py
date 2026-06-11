"""
CMS (Constant Maturity Swap) Pricing Module
============================================
Convexity-adjusted CMS rates, CMS caplets/floorlets, CMS spread options,
and CMS swap valuation.

Models implemented
------------------
Linear TSR  : closed-form annuity-derivative convexity adjustment (Hagan 2003)
Replication : numerical integration over swaption smile (static replication)
Kirk        : bivariate-normal approximation for spread options

References
----------
Hagan, P. (2003) "Convexity Conundrums: Pricing CMS Swaps, Caps, and Floors"
Wilmott, Mar 2003.

Kirk, E. (1995) "Correlation in Energy Markets", Managing Energy Price Risk,
Risk Publications, 71-78.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Literal

import numpy as np
from scipy import integrate
from scipy.stats import norm

from .curve import DiscountCurve

Φ = norm.cdf
φ = norm.pdf

__all__ = [
    "CMSConvexityResult",
    "CMSCaplet",
    "CMSSpreadOption",
    "CMSSwap",
    "cms_convexity_adj",
    "cms_caplet_pv",
    "cms_floorlet_pv",
    "cms_spread_option_pv",
]


# ── Annuity math ──────────────────────────────────────────────────────────────

def _annuity_par(S: float, n_periods: int, freq: int) -> float:
    """
    Level annuity of a fixed-rate bond paying S/freq each period, par redeemed.

    G(S) = (1 - (1 + S/m)^{-n}) / (S/m)   where m = freq, n = n_periods

    Returns G(S) in *per-unit* terms (multiply by notional and period length
    to get dollar annuity).
    """
    if S < 1e-10:
        return float(n_periods) / freq
    u = 1.0 + S / freq
    return (1.0 - u ** (-n_periods)) * freq / S


def _annuity_par_deriv(S: float, n_periods: int, freq: int) -> float:
    """
    dG/dS — derivative of the level annuity G(S) with respect to S.

    G = (1 - u^{-n}) / (S/m),  u = 1 + S/m
    dG/dS = [n × u^{-n-1}/m × (S/m) - (1 - u^{-n})] / (S/m)²
    """
    if S < 1e-10:
        n = n_periods
        # Taylor expansion: G'(0) = -n(n+1)/(2*freq)
        return -float(n * (n + 1)) / (2.0 * freq)
    m = float(freq)
    u = 1.0 + S / m
    n = n_periods
    G = (1.0 - u ** (-n)) * m / S
    # dG/dS = [n*u^{-n-1}/m - G/m] / (S/m) via quotient rule
    dG = (n * u ** (-n - 1) / m - G / m) / (S / m)
    return dG


def _h_factor(S: float, n_periods: int, freq: int) -> float:
    """
    H(S) = S × G'(S) / G(S) — dimensionless factor used in convexity adj.

    The linear-TSR convexity adjustment is proportional to H.
    """
    G = _annuity_par(S, n_periods, freq)
    Gp = _annuity_par_deriv(S, n_periods, freq)
    if abs(G) < 1e-14:
        return 0.0
    return S * Gp / G


# ── Dataclasses ───────────────────────────────────────────────────────────────

@dataclass
class CMSConvexityResult:
    forward_swap_rate: float
    convexity_adj:     float
    cms_rate:          float
    expiry:            float
    swap_tenor:        float
    vol:               float
    model:             str

    @property
    def convexity_adj_bps(self) -> float:
        return self.convexity_adj * 10_000


@dataclass
class CMSCaplet:
    """A single CMS caplet or floorlet.

    Parameters
    ----------
    t_fix       : option expiry / rate-fixing date (years from today)
    t_pay       : payment date (years from today); typically t_fix + δ
    swap_tenor  : tenor of the underlying CMS rate (years, e.g. 10.0 for 10Y CMS)
    strike      : caplet strike (decimal)
    notional    : face value
    cap_floor   : "cap" (caplet) or "floor" (floorlet)
    freq        : CMS swap payment frequency (2 = semi-annual)
    """
    t_fix:      float
    t_pay:      float
    swap_tenor: float
    strike:     float
    notional:   float = 1_000_000.0
    cap_floor:  Literal["cap", "floor"] = "cap"
    freq:       int   = 2

    @property
    def tau(self) -> float:
        return self.t_pay - self.t_fix


@dataclass
class CMSSpreadOption:
    """CMS spread option: pays max(±(S_long - S_short - K), 0).

    E.g. 10Y-2Y steepener call: long_tenor=10, short_tenor=2.

    Parameters
    ----------
    t_fix        : fixing / expiry date (years)
    t_pay        : payment date (years)
    long_tenor   : tenor of the "long" CMS leg (years, e.g. 10)
    short_tenor  : tenor of the "short" CMS leg (years, e.g. 2)
    spread_strike: strike on the spread (decimal, e.g. 0.005 = 50bps)
    notional     : face value
    call_put     : "call" = pay if spread > K (steepener); "put" = pay if spread < K
    freq         : underlying swap frequency (2 = semi-annual)
    """
    t_fix:         float
    t_pay:         float
    long_tenor:    float
    short_tenor:   float
    spread_strike: float
    notional:      float = 1_000_000.0
    call_put:      Literal["call", "put"] = "call"
    freq:          int   = 2

    @property
    def tau(self) -> float:
        return self.t_pay - self.t_fix


# ── CMS convexity adjustment ──────────────────────────────────────────────────

def cms_convexity_adj(
    curve:        DiscountCurve,
    expiry:       float,
    swap_tenor:   float,
    vol:          float,
    freq:         int = 2,
    model:        Literal["linear_tsr", "replication"] = "linear_tsr",
    n_strikes:    int = 100,
) -> CMSConvexityResult:
    """
    Compute the CMS convexity adjustment under the Linear TSR or static
    replication model.

    Parameters
    ----------
    curve       : DiscountCurve with today's discount factors
    expiry      : option/payment expiry T (years from today)
    swap_tenor  : tenor of the underlying swap (years, e.g. 10.0)
    vol         : Black-76 lognormal vol of the underlying swap rate
    freq        : underlying swap reset frequency (2 = semi-annual)
    model       : "linear_tsr" (fast) or "replication" (more accurate)
    n_strikes   : number of integration points for replication model

    Returns
    -------
    CMSConvexityResult with forward_swap_rate, convexity_adj, cms_rate, etc.
    """
    if expiry <= 0:
        raise ValueError("expiry must be positive")
    if swap_tenor <= 0:
        raise ValueError("swap_tenor must be positive")
    if vol <= 0:
        raise ValueError("vol must be positive")

    n_periods = int(round(swap_tenor * freq))
    S_fwd = _forward_swap_rate(curve, expiry, swap_tenor, freq)

    if model == "linear_tsr":
        adj = _convexity_linear_tsr(S_fwd, expiry, vol, n_periods, freq)
    elif model == "replication":
        adj = _convexity_replication(curve, expiry, swap_tenor, vol, freq,
                                     n_strikes, S_fwd)
    else:
        raise ValueError(f"Unknown model '{model}'")

    cms_rate = S_fwd + adj
    return CMSConvexityResult(
        forward_swap_rate=S_fwd,
        convexity_adj=adj,
        cms_rate=cms_rate,
        expiry=expiry,
        swap_tenor=swap_tenor,
        vol=vol,
        model=model,
    )


def _forward_swap_rate(
    curve:      DiscountCurve,
    t_start:    float,
    swap_tenor: float,
    freq:       int,
) -> float:
    """Forward swap rate S(t_start, t_start+swap_tenor) from discount curve."""
    t_end     = t_start + swap_tenor
    df_start  = float(curve.df(t_start))
    df_end    = float(curve.df(t_end))
    annuity   = _annuity_from_curve(curve, t_start, swap_tenor, freq)
    return (df_start - df_end) / annuity if annuity > 1e-14 else np.nan


def _annuity_from_curve(
    curve:      DiscountCurve,
    t_start:    float,
    swap_tenor: float,
    freq:       int,
) -> float:
    """Dollar annuity A = Σᵢ (1/freq) × DF(tᵢ) for fixed-leg payment times."""
    dt        = 1.0 / freq
    n_periods = int(round(swap_tenor * freq))
    annuity   = sum(
        dt * float(curve.df(t_start + (k + 1) * dt))
        for k in range(n_periods)
    )
    return annuity


def _convexity_linear_tsr(
    S_fwd:     float,
    T:         float,
    vol:       float,
    n_periods: int,
    freq:      int,
) -> float:
    """
    Linear TSR convexity adjustment (Hagan 2003, Equation 2.8):

        adj = σ² × T × S_fwd × H(S_fwd)

    where H = S × G'(S) / G(S) captures the convexity of the annuity mapping.
    The sign of H is typically negative (annuity decreasing in rates), making
    adj positive (CMS > forward swap rate) as expected economically.
    """
    H = _h_factor(S_fwd, n_periods, freq)
    return vol**2 * T * S_fwd * H * (-1.0)


def _convexity_replication(
    curve:      DiscountCurve,
    expiry:     float,
    swap_tenor: float,
    vol:        float,
    freq:       int,
    n_strikes:  int,
    S_fwd:      float,
) -> float:
    """
    Static replication convexity adjustment via numerical integration of
    payer and receiver swaption prices over the strike space.

        E[S(T)] = S_fwd + 2 × ∫_{S_fwd}^{∞} Payer(K)/A(0) dK
                        - 2 × ∫_{0}^{S_fwd} Receiver(K)/A(0) dK

    where Payer(K) and Receiver(K) are Black-76 swaption prices normalised
    by the annuity. This integral converges rapidly for finite vol.
    """
    from scipy.stats import norm as _norm
    annuity_0 = _annuity_from_curve(curve, expiry, swap_tenor, freq)

    def payer_unnorm(K: float) -> float:
        if K <= 0 or vol <= 0 or expiry <= 0:
            return 0.0
        sqT = math.sqrt(expiry)
        d1 = (math.log(S_fwd / K) + 0.5 * vol**2 * expiry) / (vol * sqT)
        d2 = d1 - vol * sqT
        return S_fwd * _norm.cdf(d1) - K * _norm.cdf(d2)

    def receiver_unnorm(K: float) -> float:
        if K <= 0 or vol <= 0 or expiry <= 0:
            return 0.0
        sqT = math.sqrt(expiry)
        d1 = (math.log(S_fwd / K) + 0.5 * vol**2 * expiry) / (vol * sqT)
        d2 = d1 - vol * sqT
        return K * _norm.cdf(-d2) - S_fwd * _norm.cdf(-d1)

    K_max = S_fwd * math.exp(3.5 * vol * math.sqrt(expiry))
    K_min = max(S_fwd * math.exp(-3.5 * vol * math.sqrt(expiry)), 1e-6)

    # Payer integral: ∫_{S_fwd}^{K_max} g''(K) × payer(K) dK
    # where g(S) = S (identity payoff) → g''(K) = 0, so use weight = 1/annuity
    # Proper Hagan formula: adj = (1/A) × ∫ d²f/dS² × swaption(S) dS
    # For CMS payoff f(S) = S: d²f/dS² = 0 for linear payoffs → use 2nd deriv
    # Use the direct approach: compute E^{T_pay}[S] numerically:
    # E[S] ≈ S_fwd + ∫_{S_fwd}^∞ Φ(d1(K)) dK - ∫_0^{S_fwd} Φ(-d1(K)) dK
    # (call/put delta decomposition)

    def integrand_call(K: float) -> float:
        if K <= 0 or vol <= 0:
            return 0.0
        sqT = math.sqrt(expiry)
        d1 = (math.log(S_fwd / K) + 0.5 * vol**2 * expiry) / (vol * sqT)
        return _norm.cdf(d1)

    def integrand_put(K: float) -> float:
        if K <= 0 or vol <= 0:
            return 0.0
        sqT = math.sqrt(expiry)
        d1 = (math.log(S_fwd / K) + 0.5 * vol**2 * expiry) / (vol * sqT)
        return _norm.cdf(-d1)

    call_int, _ = integrate.quad(integrand_call, S_fwd, K_max, limit=n_strikes)
    put_int,  _ = integrate.quad(integrand_put,  K_min, S_fwd, limit=n_strikes)

    # E[S(T)] under T-forward measure ≈ S_fwd + call_int - put_int
    # (difference from S_fwd gives the convexity correction)
    e_cms    = S_fwd + call_int - put_int
    adj      = e_cms - S_fwd
    return adj


# ── CMS Caplet / Floorlet ────────────────────────────────────────────────────

def cms_caplet_pv(
    caplet:  CMSCaplet,
    curve:   DiscountCurve,
    vol:     float,
    model:   Literal["linear_tsr", "replication"] = "linear_tsr",
) -> float:
    """
    Price a CMS caplet under Black-76 with TSR-adjusted forward.

    The CMS caplet pays N × τ × max(S_CMS(T) - K, 0) at T_pay.
    Under the T_pay-forward measure with convexity-adjusted CMS rate R_cms:

        CMS_Caplet = DF(0, T_pay) × N × τ × [R_cms × Φ(d1) - K × Φ(d2)]
        d1 = [ln(R_cms/K) + ½σ²T] / (σ√T)
        d2 = d1 - σ√T

    Parameters
    ----------
    caplet : CMSCaplet descriptor
    curve  : discount curve
    vol    : Black-76 vol of the underlying CMS swap rate
    model  : convexity adjustment model

    Returns
    -------
    PV in same currency units as caplet.notional
    """
    if caplet.t_fix <= 0:
        raise ValueError("t_fix must be positive")
    if caplet.swap_tenor <= 0:
        raise ValueError("swap_tenor must be positive")
    if caplet.strike < 0:
        raise ValueError("strike must be non-negative")
    if vol <= 0:
        raise ValueError("vol must be positive")

    res = cms_convexity_adj(curve, caplet.t_fix, caplet.swap_tenor, vol,
                            freq=caplet.freq, model=model)
    R_cms = res.cms_rate
    K     = caplet.strike
    T     = caplet.t_fix
    tau   = caplet.tau
    df    = float(curve.df(caplet.t_pay))

    if R_cms <= 0 or K <= 0:
        # Use Bachelier (normal) formula instead
        return _cms_caplet_bachelier(R_cms, K, T, tau, df,
                                     vol * R_cms, caplet.notional,
                                     caplet.cap_floor)

    sqT = math.sqrt(T)
    d1  = (math.log(R_cms / K) + 0.5 * vol**2 * T) / (vol * sqT)
    d2  = d1 - vol * sqT

    if caplet.cap_floor == "cap":
        raw = R_cms * Φ(d1) - K * Φ(d2)
    else:
        raw = K * Φ(-d2) - R_cms * Φ(-d1)

    return caplet.notional * tau * df * raw


def cms_floorlet_pv(
    caplet:  CMSCaplet,
    curve:   DiscountCurve,
    vol:     float,
    model:   Literal["linear_tsr", "replication"] = "linear_tsr",
) -> float:
    """Price a CMS floorlet (convenience wrapper with cap_floor='floor')."""
    fl = CMSCaplet(
        t_fix=caplet.t_fix, t_pay=caplet.t_pay,
        swap_tenor=caplet.swap_tenor, strike=caplet.strike,
        notional=caplet.notional, cap_floor="floor", freq=caplet.freq,
    )
    return cms_caplet_pv(fl, curve, vol, model)


def _cms_caplet_bachelier(
    F:     float,
    K:     float,
    T:     float,
    tau:   float,
    df:    float,
    sigma: float,
    N:     float,
    cap_floor: str,
) -> float:
    """Bachelier (normal) CMS caplet/floorlet when rates are near zero."""
    if sigma * math.sqrt(T) < 1e-12:
        payoff = max(F - K, 0.0) if cap_floor == "cap" else max(K - F, 0.0)
        return N * tau * df * payoff
    sig_sqt = sigma * math.sqrt(T)
    d = (F - K) / sig_sqt
    if cap_floor == "cap":
        raw = sig_sqt * (d * Φ(d) + φ(d))
    else:
        raw = sig_sqt * (-d * Φ(-d) + φ(d))
    return N * tau * df * raw


# ── CMS cap/floor parity ─────────────────────────────────────────────────────

def cms_caplet_floorlet_parity(
    caplet:  CMSCaplet,
    curve:   DiscountCurve,
    vol:     float,
    model:   Literal["linear_tsr", "replication"] = "linear_tsr",
) -> float:
    """
    CMS caplet - CMS floorlet = DF(T_pay) × N × τ × (R_cms - K).
    Returns the difference (model-independent, analytical).
    """
    res  = cms_convexity_adj(curve, caplet.t_fix, caplet.swap_tenor, vol,
                             freq=caplet.freq, model=model)
    df   = float(curve.df(caplet.t_pay))
    return caplet.notional * caplet.tau * df * (res.cms_rate - caplet.strike)


# ── CMS Spread Option ─────────────────────────────────────────────────────────

def cms_spread_option_pv(
    option:    CMSSpreadOption,
    curve:     DiscountCurve,
    vol_long:  float,
    vol_short: float,
    rho:       float,
    model:     Literal["linear_tsr", "replication"] = "linear_tsr",
) -> float:
    """
    Price a CMS spread option using Kirk's (1995) approximation.

    Pays N × τ × max(S_long - S_short - K, 0) at T_pay  [call]
    or  N × τ × max(K - (S_long - S_short), 0) at T_pay [put]

    Kirk's formula treats the spread as a single lognormal:
      - F_spread = R_long - R_short (adjusted forward rates)
      - vol_spread = sqrt(σ_long² + σ_short²×(F2/F_spread)² - 2ρσ_long σ_short(F2/F_spread))
      - d1, d2 standard Black-76

    Parameters
    ----------
    option    : CMSSpreadOption descriptor
    curve     : discount curve
    vol_long  : Black vol of the long-tenor CMS rate
    vol_short : Black vol of the short-tenor CMS rate
    rho       : correlation between the two CMS rates (-1 ≤ ρ ≤ 1)
    model     : convexity adjustment model

    Returns
    -------
    PV in same currency units as option.notional
    """
    if not -1.0 <= rho <= 1.0:
        raise ValueError("rho must be in [-1, 1]")
    if vol_long <= 0 or vol_short <= 0:
        raise ValueError("vols must be positive")

    T   = option.t_fix
    tau = option.tau
    df  = float(curve.df(option.t_pay))
    K   = option.spread_strike

    res_l = cms_convexity_adj(curve, T, option.long_tenor,  vol_long,
                              freq=option.freq, model=model)
    res_s = cms_convexity_adj(curve, T, option.short_tenor, vol_short,
                              freq=option.freq, model=model)

    F1 = res_l.cms_rate    # long CMS rate
    F2 = res_s.cms_rate    # short CMS rate

    # Kirk's approximation: treat F1 and F2 + K as lognormals
    Fp  = F2 + K          # shifted denominator in Kirk's formulation
    # Effective vol of the spread under Kirk (equation 4 from Kirk 1995):
    if Fp <= 0 or F1 <= 0:
        # Use Bachelier approximation for near-zero / negative rates
        return _spread_option_bachelier(F1, F2, K, T, tau, df,
                                        vol_long, vol_short, rho,
                                        option.notional, option.call_put)

    sqT     = math.sqrt(T)
    vol_eff = math.sqrt(
        vol_long**2
        + vol_short**2 * (Fp / F1)**2
        - 2.0 * rho * vol_long * vol_short * (Fp / F1)
    )
    if vol_eff < 1e-12:
        payoff = max(F1 - F2 - K, 0.0) if option.call_put == "call" else max(K - (F1 - F2), 0.0)
        return option.notional * tau * df * payoff

    d1 = (math.log(F1 / Fp) + 0.5 * vol_eff**2 * T) / (vol_eff * sqT)
    d2 = d1 - vol_eff * sqT

    if option.call_put == "call":
        raw = F1 * Φ(d1) - Fp * Φ(d2)
    else:
        raw = Fp * Φ(-d2) - F1 * Φ(-d1)

    return option.notional * tau * df * raw


def _spread_option_bachelier(
    F1: float, F2: float, K: float,
    T: float, tau: float, df: float,
    v1: float, v2: float, rho: float,
    N: float, call_put: str,
) -> float:
    """Bachelier spread option when rates are near zero."""
    sig = math.sqrt(v1**2 + v2**2 - 2.0 * rho * v1 * v2) * math.sqrt(T)
    Fspr = F1 - F2
    if sig < 1e-12:
        payoff = max(Fspr - K, 0.0) if call_put == "call" else max(K - Fspr, 0.0)
        return N * tau * df * payoff
    d = (Fspr - K) / sig
    if call_put == "call":
        raw = sig * (d * Φ(d) + φ(d))
    else:
        raw = sig * (-d * Φ(-d) + φ(d))
    return N * tau * df * raw


# ── CMS Swap ──────────────────────────────────────────────────────────────────

@dataclass
class CMSSwap:
    """
    CMS swap: one leg pays CMS rate (with convexity adjustment), other pays fixed.

    Parameters
    ----------
    cms_tenor       : tenor of the floating CMS reference rate (years, e.g. 10.0)
    first_payment   : first payment date (years from today)
    last_payment    : last payment date (years from today)
    fixed_rate      : fixed leg rate (decimal)
    notional        : face value
    pay_cms         : True = pay CMS / receive fixed; False = pay fixed / receive CMS
    payment_freq    : number of payments per year on both legs
    cms_swap_freq   : reset frequency of underlying CMS swap (2 = semi-annual)
    """
    cms_tenor:     float
    first_payment: float
    last_payment:  float
    fixed_rate:    float
    notional:      float = 10_000_000.0
    pay_cms:       bool  = True
    payment_freq:  int   = 1
    cms_swap_freq: int   = 2

    def _payment_dates(self) -> list[float]:
        dt  = 1.0 / self.payment_freq
        eps = 1e-9
        t   = self.first_payment
        dates: list[float] = []
        while t <= self.last_payment + eps:
            dates.append(round(t, 8))
            t += dt
        return dates

    def fixed_leg_pv(self, curve: DiscountCurve) -> float:
        """PV of the fixed coupon leg."""
        dt  = 1.0 / self.payment_freq
        pv  = 0.0
        for t_pay in self._payment_dates():
            pv += dt * float(curve.df(t_pay))
        return self.notional * self.fixed_rate * pv

    def cms_leg_pv(self, curve: DiscountCurve, vol: float,
                   model: Literal["linear_tsr", "replication"] = "linear_tsr",
                   ) -> float:
        """
        PV of the CMS floating leg.

        Each payment: N × (1/freq) × E[S_cms(t_fix)] × DF(t_pay)
                    = N × (1/freq) × (S_fwd + convexity_adj) × DF(t_pay)
        """
        dt  = 1.0 / self.payment_freq
        pv  = 0.0
        for t_pay in self._payment_dates():
            t_fix = t_pay  # typical in-arrears for CMS
            if t_fix <= 0:
                continue
            res  = cms_convexity_adj(curve, t_fix, self.cms_tenor, vol,
                                     freq=self.cms_swap_freq, model=model)
            pv  += dt * res.cms_rate * float(curve.df(t_pay))
        return self.notional * pv

    def pv(self, curve: DiscountCurve, vol: float,
           model: Literal["linear_tsr", "replication"] = "linear_tsr",
           ) -> float:
        """Net PV from the perspective of the CMS payer."""
        cms_pv   = self.cms_leg_pv(curve, vol, model)
        fixed_pv = self.fixed_leg_pv(curve)
        if self.pay_cms:
            return cms_pv - fixed_pv
        else:
            return fixed_pv - cms_pv

    def par_fixed_rate(self, curve: DiscountCurve, vol: float,
                       model: Literal["linear_tsr", "replication"] = "linear_tsr",
                       ) -> float:
        """Fixed rate that sets swap PV = 0 (CMS par rate)."""
        dt     = 1.0 / self.payment_freq
        annuity = sum(
            dt * float(curve.df(t_pay))
            for t_pay in self._payment_dates()
        )
        if annuity < 1e-14:
            return float("nan")
        cms_pv_per_notional = self.cms_leg_pv(curve, vol, model) / self.notional
        return cms_pv_per_notional / annuity

    def dv01(self, curve: DiscountCurve, vol: float,
             shift_bps: float = 1.0,
             model: Literal["linear_tsr", "replication"] = "linear_tsr",
             ) -> float:
        """Sensitivity to 1bp parallel shift in the discount curve."""
        bump  = shift_bps / 10_000
        times = curve._times
        log_dfs_up   = curve._log_df - bump * times
        log_dfs_down = curve._log_df + bump * times
        curve_up   = DiscountCurve(curve.ref_date, times, np.exp(log_dfs_up))
        curve_down = DiscountCurve(curve.ref_date, times, np.exp(log_dfs_down))
        return (self.pv(curve_up, vol, model) - self.pv(curve_down, vol, model)) / 2.0
