"""
SOFR Cap / Floor Pricing Engine
================================
Prices interest-rate caps and floors as strips of caplets / floorlets.

Pricing models
--------------
Black-76        : log-normal forward model (industry standard for USD caps)
Bachelier       : normal-vol model (better for low / near-zero rate environments)

Both models are implemented; functions for converting between them are provided.

Instruments
-----------
Caplet          : call option on a single SOFR compounding period  [reset, pay]
Floorlet        : put option on a single SOFR compounding period
Cap             : strip of caplets sharing a common flat (term) vol
Floor           : strip of floorlets sharing a common flat (term) vol
CapFloorVolSurface : 2-D (tenor × strike) vol grid with bilinear interpolation

Conventions (USD)
-----------------
- Quarterly reset periods (0.25 y accrual)
- First caplet excluded by default (immediate-set convention)
- ACT/360 day-count for accrual factors
- Discount factors from SOFR OIS curve

Greeks
------
DV01  : PV sensitivity to +1bp parallel yield shift
Vega  : PV sensitivity to +1 vol point (absolute for Bachelier, relative for Black)
Theta : PV sensitivity to passage of 1 business day

Bootstrap
---------
`strip_caplet_vols` : convert term (flat cap) vols to per-caplet forward vols
                      by sequentially pricing the marginal caplet in each strip.

References
----------
Black, F. (1976). "The pricing of commodity contracts." JFE 3(1-2), 167-179.
Hull, J. (2022). Options, Futures, and Other Derivatives (11th ed.), Ch. 29.
Brigo, D. & Mercurio, F. (2006). Interest Rate Models — Theory and Practice.
"""
from __future__ import annotations

import numpy as np
from dataclasses import dataclass, field
from datetime import date, timedelta
from typing import Literal

from scipy.optimize import brentq
from scipy.stats import norm as _norm

from .curve import DiscountCurve

_Φ = _norm.cdf
_φ = _norm.pdf


# ── Low-level option formulas ────────────────────────────────────────────────

def caplet_black_pv(
    F: float,
    K: float,
    T: float,
    tau: float,
    df_pay: float,
    vol: float,
    notional: float = 1.0,
    cap_floor: Literal["cap", "floor"] = "cap",
) -> float:
    """
    Black-76 price for a single caplet or floorlet.

    Parameters
    ----------
    F        : forward SOFR compound rate for [T, T+tau]
    K        : strike (decimal)
    T        : expiry / fixing date in years from today
    tau      : accrual period in years
    df_pay   : discount factor to payment date T + tau
    vol      : Black-76 lognormal vol (decimal, e.g. 0.50 = 50%)
    notional : face value
    cap_floor: 'cap' (call) or 'floor' (put)

    Returns
    -------
    Present value in the same currency as notional.
    """
    if T <= 1e-8 or vol <= 1e-10:
        intrinsic = (max(F - K, 0.0) if cap_floor == "cap" else max(K - F, 0.0))
        return float(notional * tau * df_pay * intrinsic)
    if K <= 1e-10:
        return float(notional * tau * df_pay * F) if cap_floor == "cap" else 0.0

    sqrt_T = np.sqrt(T)
    d1 = (np.log(F / K) + 0.5 * vol ** 2 * T) / (vol * sqrt_T)
    d2 = d1 - vol * sqrt_T

    if cap_floor == "cap":
        pv = notional * tau * df_pay * (F * _Φ(d1) - K * _Φ(d2))
    else:
        pv = notional * tau * df_pay * (K * _Φ(-d2) - F * _Φ(-d1))
    return float(pv)


def caplet_bachelier_pv(
    F: float,
    K: float,
    T: float,
    tau: float,
    df_pay: float,
    normal_vol: float,
    notional: float = 1.0,
    cap_floor: Literal["cap", "floor"] = "cap",
) -> float:
    """
    Bachelier (normal-vol) price for a single caplet or floorlet.

    PV_cap  = N·τ·DF·σ_N·√T·[d·Φ(d) + φ(d)]
    PV_floor = N·τ·DF·σ_N·√T·[−d·Φ(−d) + φ(d)]
    where d = (F − K) / (σ_N·√T)

    Parameters
    ----------
    normal_vol : absolute normal vol in rate units (e.g. 0.01 = 100 bps/√yr)
    """
    if T <= 1e-8 or normal_vol <= 1e-10:
        intrinsic = (max(F - K, 0.0) if cap_floor == "cap" else max(K - F, 0.0))
        return float(notional * tau * df_pay * intrinsic)

    sig_sqt = normal_vol * np.sqrt(T)
    d = (F - K) / sig_sqt

    if cap_floor == "cap":
        pv = notional * tau * df_pay * sig_sqt * (d * _Φ(d) + _φ(d))
    else:
        pv = notional * tau * df_pay * sig_sqt * (-d * _Φ(-d) + _φ(d))
    return float(pv)


def black_to_normal_vol(F: float, K: float, T: float, black_vol: float) -> float:
    """
    Approximate Black-76 vol → Bachelier (normal) vol conversion.

    For ATM (F ≈ K): σ_N ≈ σ_B · F
    General: use the Hagan (2003) approximation.
    """
    if T <= 0 or black_vol <= 0:
        return 0.0
    if abs(F - K) < 1e-8:
        return black_vol * F
    lnFK = np.log(F / K)
    z = black_vol * lnFK / (black_vol * np.sqrt(T) * np.sqrt(F * K))
    return black_vol * np.sqrt(F * K) * z / (np.exp(z) - 1) if abs(z) > 1e-6 else black_vol * np.sqrt(F * K)


# ── Caplet class ─────────────────────────────────────────────────────────────

@dataclass
class Caplet:
    """
    Single option on the SOFR compound rate over [t_reset, t_pay].

    t_reset and t_pay are in years from the reference date (DiscountCurve.ref_date).
    cap_floor = 'cap' for a call (profits when rates rise above K),
                'floor' for a put (profits when rates fall below K).
    """
    t_reset:   float                          # fixing date (years)
    t_pay:     float                          # payment date (years)
    strike:    float                          # K (decimal, e.g. 0.05 = 5%)
    notional:  float = 1_000_000.0
    cap_floor: Literal["cap", "floor"] = "cap"

    @property
    def tau(self) -> float:
        return self.t_pay - self.t_reset

    def forward_rate(self, curve: DiscountCurve) -> float:
        """Simple (linear) forward SOFR for the accrual period."""
        tau = self.tau
        if tau <= 1e-10:
            return 0.0
        return (curve.df(self.t_reset) / curve.df(self.t_pay) - 1.0) / tau

    def black_pv(self, curve: DiscountCurve, vol: float) -> float:
        """Black-76 PV."""
        return caplet_black_pv(
            self.forward_rate(curve), self.strike,
            self.t_reset, self.tau, curve.df(self.t_pay),
            vol, self.notional, self.cap_floor,
        )

    def bachelier_pv(self, curve: DiscountCurve, normal_vol: float) -> float:
        """Bachelier (normal-vol) PV."""
        return caplet_bachelier_pv(
            self.forward_rate(curve), self.strike,
            self.t_reset, self.tau, curve.df(self.t_pay),
            normal_vol, self.notional, self.cap_floor,
        )

    def implied_black_vol(self, curve: DiscountCurve, price: float) -> float:
        """Implied Black-76 vol from market price (Brent's method)."""
        def obj(v):
            return self.black_pv(curve, v) - price
        try:
            return brentq(obj, 1e-6, 5.0, xtol=1e-10)
        except ValueError:
            return float("nan")

    def implied_normal_vol(self, curve: DiscountCurve, price: float) -> float:
        """Implied Bachelier normal vol from market price."""
        def obj(v):
            return self.bachelier_pv(curve, v) - price
        try:
            return brentq(obj, 1e-8, 1.0, xtol=1e-12)
        except ValueError:
            return float("nan")


# ── Cap / Floor strip ────────────────────────────────────────────────────────

class Cap:
    """
    Interest-rate cap = strip of caplets on quarterly SOFR reset periods.

    By convention (USD market) the first caplet (immediate reset at t=0) is
    excluded.  The cap starts accruing from the first future reset period.

    Parameters
    ----------
    maturity_years     : total cap maturity
    strike             : cap strike (decimal)
    notional           : face value
    freq               : reset frequency per year (4 = quarterly)
    first_caplet_idx   : number of caplets to skip from the front (default 1)
    """

    def __init__(
        self,
        maturity_years: float,
        strike: float,
        notional: float = 1_000_000.0,
        freq: int = 4,
        first_caplet_idx: int = 1,
    ):
        self.maturity_years   = maturity_years
        self.strike           = strike
        self.notional         = notional
        self.freq             = freq
        self.first_caplet_idx = first_caplet_idx
        self._period          = 1.0 / freq
        n                     = max(1, int(round(maturity_years * freq)))
        self._all_periods     = [(i * self._period, (i + 1) * self._period)
                                 for i in range(n)]

    def caplets(self) -> list[Caplet]:
        return [
            Caplet(ts, te, self.strike, self.notional, "cap")
            for ts, te in self._all_periods[self.first_caplet_idx:]
        ]

    def n_caplets(self) -> int:
        return max(0, len(self._all_periods) - self.first_caplet_idx)

    def annuity(self, curve: DiscountCurve) -> float:
        """Sum of tau_i × DF(T_i) over active caplets."""
        return sum(
            (te - ts) * curve.df(te)
            for ts, te in self._all_periods[self.first_caplet_idx:]
        )

    def atm_forward(self, curve: DiscountCurve) -> float:
        """
        Annuity-weighted average of period forward rates.
        This is the fair strike for a zero-value cap (= floor strike for same vol).
        """
        caplets = self.caplets()
        numer   = sum(cl.forward_rate(curve) * cl.tau * curve.df(cl.t_pay)
                      for cl in caplets)
        denom   = self.annuity(curve)
        return numer / denom if denom > 1e-12 else 0.0

    def pv(self, curve: DiscountCurve, vol: float) -> float:
        """Total cap PV under Black-76 flat (term) vol."""
        return sum(cl.black_pv(curve, vol) for cl in self.caplets())

    def pv_bachelier(self, curve: DiscountCurve, normal_vol: float) -> float:
        """Total cap PV under Bachelier flat vol."""
        return sum(cl.bachelier_pv(curve, normal_vol) for cl in self.caplets())

    def implied_vol(
        self,
        curve: DiscountCurve,
        price: float,
        initial_guess: float = 0.20,
    ) -> float:
        """Black-76 flat (term) implied vol for the full cap strip."""
        def obj(v): return self.pv(curve, v) - price
        try:
            return brentq(obj, 1e-6, 5.0, xtol=1e-10)
        except ValueError:
            return float("nan")

    def dv01(self, curve: DiscountCurve, vol: float, bump_bps: float = 1.0) -> float:
        """Dollar DV01: PV change for +1bp parallel shift."""
        bump    = bump_bps / 10_000.0
        times   = curve._times
        log_dfs = curve._log_df
        dfs_up  = np.exp(log_dfs - bump * times)
        c_up    = DiscountCurve(curve.ref_date, times, dfs_up)
        return self.pv(c_up, vol) - self.pv(curve, vol)

    def vega(self, curve: DiscountCurve, vol: float, dvol: float = 0.0001) -> float:
        """Vega: PV change per 1bp increase in Black vol."""
        return (self.pv(curve, vol + dvol) - self.pv(curve, vol - dvol)) / (2 * dvol)

    def theta(self, curve: DiscountCurve, vol: float, dt: float = 1.0 / 252) -> float:
        """Theta: PV change per 1 business day of time decay (approximate)."""
        pv_now  = self.pv(curve, vol)
        pv_next = sum(
            Caplet(
                max(cl.t_reset - dt, 0.0),
                max(cl.t_pay - dt, max(cl.t_reset - dt, 0.0) + 1e-6),
                cl.strike, cl.notional, cl.cap_floor,
            ).black_pv(curve, vol)
            for cl in self.caplets()
        )
        return pv_next - pv_now

    def summary(self, curve: DiscountCurve, vol: float) -> dict:
        pv = self.pv(curve, vol)
        F  = self.atm_forward(curve)
        return {
            "type":            "cap",
            "maturity_years":  self.maturity_years,
            "strike_pct":      round(self.strike * 100, 5),
            "atm_forward_pct": round(F * 100, 5),
            "moneyness_bps":   round((F - self.strike) * 10_000, 2),
            "n_caplets":       self.n_caplets(),
            "annuity":         round(self.annuity(curve), 8),
            "pv":              round(pv, 2),
            "vol_pct":         round(vol * 100, 4),
            "dv01":            round(self.dv01(curve, vol), 2),
            "vega_per_bp":     round(self.vega(curve, vol), 2),
        }


class Floor:
    """
    Interest-rate floor = strip of floorlets on quarterly SOFR reset periods.

    Put-call parity:  Cap(K) − Floor(K) = PV(float leg) − PV(fixed leg at K)
    So a floor receiver profits when rates fall below K.
    """

    def __init__(
        self,
        maturity_years: float,
        strike: float,
        notional: float = 1_000_000.0,
        freq: int = 4,
        first_caplet_idx: int = 1,
    ):
        self.maturity_years   = maturity_years
        self.strike           = strike
        self.notional         = notional
        self.freq             = freq
        self.first_caplet_idx = first_caplet_idx
        self._period          = 1.0 / freq
        n                     = max(1, int(round(maturity_years * freq)))
        self._all_periods     = [(i * self._period, (i + 1) * self._period)
                                 for i in range(n)]

    def floorlets(self) -> list[Caplet]:
        return [
            Caplet(ts, te, self.strike, self.notional, "floor")
            for ts, te in self._all_periods[self.first_caplet_idx:]
        ]

    def n_floorlets(self) -> int:
        return max(0, len(self._all_periods) - self.first_caplet_idx)

    def annuity(self, curve: DiscountCurve) -> float:
        return sum(
            (te - ts) * curve.df(te)
            for ts, te in self._all_periods[self.first_caplet_idx:]
        )

    def atm_forward(self, curve: DiscountCurve) -> float:
        cap = Cap(self.maturity_years, self.strike, self.notional,
                  self.freq, self.first_caplet_idx)
        return cap.atm_forward(curve)

    def pv(self, curve: DiscountCurve, vol: float) -> float:
        """Total floor PV under Black-76 flat vol."""
        return sum(fl.black_pv(curve, vol) for fl in self.floorlets())

    def pv_bachelier(self, curve: DiscountCurve, normal_vol: float) -> float:
        return sum(fl.bachelier_pv(curve, normal_vol) for fl in self.floorlets())

    def implied_vol(self, curve: DiscountCurve, price: float) -> float:
        def obj(v): return self.pv(curve, v) - price
        try:
            return brentq(obj, 1e-6, 5.0, xtol=1e-10)
        except ValueError:
            return float("nan")

    def dv01(self, curve: DiscountCurve, vol: float, bump_bps: float = 1.0) -> float:
        bump    = bump_bps / 10_000.0
        times   = curve._times
        log_dfs = curve._log_df
        dfs_up  = np.exp(log_dfs - bump * times)
        c_up    = DiscountCurve(curve.ref_date, times, dfs_up)
        return self.pv(c_up, vol) - self.pv(curve, vol)

    def vega(self, curve: DiscountCurve, vol: float, dvol: float = 0.0001) -> float:
        return (self.pv(curve, vol + dvol) - self.pv(curve, vol - dvol)) / (2 * dvol)

    def summary(self, curve: DiscountCurve, vol: float) -> dict:
        pv = self.pv(curve, vol)
        F  = self.atm_forward(curve)
        return {
            "type":            "floor",
            "maturity_years":  self.maturity_years,
            "strike_pct":      round(self.strike * 100, 5),
            "atm_forward_pct": round(F * 100, 5),
            "moneyness_bps":   round((F - self.strike) * 10_000, 2),
            "n_floorlets":     self.n_floorlets(),
            "annuity":         round(self.annuity(curve), 8),
            "pv":              round(pv, 2),
            "vol_pct":         round(vol * 100, 4),
            "dv01":            round(self.dv01(curve, vol), 2),
            "vega_per_bp":     round(self.vega(curve, vol), 2),
        }


# ── Put-call parity ──────────────────────────────────────────────────────────

def cap_floor_parity_pv(
    curve: DiscountCurve,
    maturity_years: float,
    strike: float,
    notional: float = 1_000_000.0,
    freq: int = 4,
    first_caplet_idx: int = 1,
) -> float:
    """
    Theoretical Cap − Floor difference from put-call parity (no vol required).

    Cap(K) - Floor(K) = N × Σ_i [τ_i × DF_i × (F_i − K)]

    Equals the PV of a receive-floating pay-fixed swap at rate K.
    """
    period = 1.0 / freq
    n      = max(1, int(round(maturity_years * freq)))
    total  = 0.0
    for i in range(first_caplet_idx, n):
        ts  = i * period
        te  = (i + 1) * period
        tau = te - ts
        df  = curve.df(te)
        F   = (curve.df(ts) / curve.df(te) - 1.0) / tau
        total += tau * df * (F - strike)
    return float(notional * total)


# ── Vol surface ─────────────────────────────────────────────────────────────

class CapFloorVolSurface:
    """
    2-D cap volatility surface: term tenors × strikes.

    Stores Black-76 implied flat (term) vols for each (tenor, strike) node.
    Bilinear interpolation in tenor and strike space.

    The surface is built from USD market convention:
      - Tenors: 1Y, 2Y, 3Y, 5Y, 7Y, 10Y
      - Strikes: ATM-100bps, ATM-50bps, ATM, ATM+50bps, ATM+100bps
      (ATM ≈ 4.33% for a SOFR ON of 4.33%)
    """

    def __init__(
        self,
        tenors: list[float],
        strikes: list[float],
        vols: list[list[float]],   # vols[i][j] = vol at tenors[i], strikes[j]
    ):
        self._tenors  = tenors
        self._strikes = strikes
        self._vols    = np.array(vols, dtype=float)

    def vol(self, tenor: float, strike: float) -> float:
        """Bilinear interpolation of Black-76 flat vol."""
        t_arr = np.array(self._tenors)
        k_arr = np.array(self._strikes)

        i = max(0, min(len(t_arr) - 2, np.searchsorted(t_arr, tenor) - 1))
        j = max(0, min(len(k_arr) - 2, np.searchsorted(k_arr, strike) - 1))

        wt = (tenor  - t_arr[i]) / (t_arr[i + 1] - t_arr[i])   if t_arr[i+1] > t_arr[i] else 0.0
        wk = (strike - k_arr[j]) / (k_arr[j + 1] - k_arr[j])   if k_arr[j+1] > k_arr[j] else 0.0
        wt = float(np.clip(wt, 0.0, 1.0))
        wk = float(np.clip(wk, 0.0, 1.0))

        v00 = self._vols[i,     j    ]
        v10 = self._vols[i + 1, j    ]
        v01 = self._vols[i,     j + 1]
        v11 = self._vols[i + 1, j + 1]
        return float((1 - wt) * (1 - wk) * v00
                     + wt      * (1 - wk) * v10
                     + (1 - wt) * wk      * v01
                     + wt       * wk      * v11)

    @classmethod
    def typical_market(cls, sofr_on: float = 0.0433) -> "CapFloorVolSurface":
        """
        Approximate 2024-vintage USD SOFR cap vol surface (Black-76).

        The ATM vol term structure is calibrated to roughly match
        post-LIBOR transition market levels for a 4.33% SOFR curve.
        Smile/skew modelled as a parabolic adjustment around ATM.
        """
        tenors  = [1.0, 2.0, 3.0, 5.0, 7.0, 10.0]
        atm_est = sofr_on  # very rough ATM proxy
        # Strikes: ATM - 150bps to ATM + 150bps in 50bp increments
        n_strikes = 7
        strikes = [atm_est - 0.015 + 0.005 * k for k in range(n_strikes)]

        # ATM vols: roughly 30-40% for short tenors, steeper inversion
        # Market-calibrated approximate levels:
        atm_vols_by_tenor = {1.0: 0.38, 2.0: 0.33, 3.0: 0.29,
                             5.0: 0.26, 7.0: 0.24, 10.0: 0.22}

        vols = []
        for T in tenors:
            atm_v = atm_vols_by_tenor[T]
            row   = []
            for K in strikes:
                moneyness = (K - atm_est) / atm_est
                # Parabolic skew: OTM receiver (low K) has lower vol; OTM payer higher
                smile_adj = 0.08 * moneyness + 0.15 * moneyness ** 2
                row.append(max(0.05, atm_v + smile_adj))
            vols.append(row)

        return cls(tenors, strikes, vols)


# ── Caplet vol bootstrap ─────────────────────────────────────────────────────

def strip_caplet_vols(
    term_vols: dict[float, float],
    curve: DiscountCurve,
    strike: float,
    notional: float = 1_000_000.0,
    freq: int = 4,
) -> list[tuple[float, float]]:
    """
    Bootstrap per-caplet (forward) vols from a curve of term (flat cap) vols.

    Strips individual caplet volatilities using sequential difference:
        CapletPV(T) = CapPV(T, flat_vol_T) − CapPV(T − dt, flat_vol_{T-dt})

    Parameters
    ----------
    term_vols : {tenor: flat_black_vol} mapping; tenors sorted ascending
    curve     : DiscountCurve for discounting and forward rates
    strike    : common cap strike used for all tenors
    notional  : cap notional
    freq      : reset frequency per year (4 = quarterly)

    Returns
    -------
    List of (expiry_years, caplet_black_vol) tuples in expiry order.
    """
    tenors  = sorted(term_vols.keys())
    period  = 1.0 / freq
    results: list[tuple[float, float]] = []
    prev_pv = 0.0

    for T in tenors:
        flat_vol = term_vols[T]
        cap      = Cap(T, strike, notional, freq)

        if cap.n_caplets() == 0:
            continue

        total_pv    = cap.pv(curve, flat_vol)
        marginal_pv = total_pv - prev_pv

        # Identify the marginal caplet
        n_all     = max(1, int(round(T * freq)))
        t_reset_i = (n_all - 1) * period   # second-to-last endpoint is the reset
        t_pay_i   = n_all * period

        cl = Caplet(t_reset_i, t_pay_i, strike, notional, "cap")

        try:
            fwd_vol = cl.implied_black_vol(curve, marginal_pv)
            if np.isfinite(fwd_vol):
                results.append((t_reset_i, fwd_vol))
        except Exception:
            results.append((t_reset_i, flat_vol))  # fall back to flat vol

        prev_pv = total_pv

    return results


# ── Convenience function ─────────────────────────────────────────────────────

def price_cap_floor(
    curve: DiscountCurve,
    maturity_years: float,
    strike: float | None = None,
    notional: float = 1_000_000.0,
    vol: float = 0.30,
    instrument: Literal["cap", "floor"] = "cap",
    freq: int = 4,
) -> dict:
    """
    Price a cap or floor and return a full analytics dictionary.

    If strike is None, uses ATM (annuity-weighted forward rate).
    """
    if instrument == "cap":
        obj = Cap(maturity_years, strike or 0.0, notional, freq)
        if strike is None:
            strike = obj.atm_forward(curve)
            obj    = Cap(maturity_years, strike, notional, freq)
    else:
        obj = Floor(maturity_years, strike or 0.0, notional, freq)
        if strike is None:
            strike = obj.atm_forward(curve)
            obj    = Floor(maturity_years, strike, notional, freq)

    return obj.summary(curve, vol)
