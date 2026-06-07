"""
Swaption pricing for the SOFR engine.

Classes
-------
Swaption           : European swaption (payer or receiver), Black-76 model
SwaptionVolSurface : ATM implied vol surface with bilinear interpolation

Module-level
------------
price_swaption     : Convenience wrapper returning a full analytics dict

Pricing model
-------------
Black-76 (log-normal forward swap rate) — the standard quoting convention
for USD swaptions.  Normal (Bachelier) vol can be converted externally via
  σ_normal ≈ σ_black × S  (at-the-money approximation).

Annuity (present value of a basis point, PVBP)
  A(0,T,τ) = Σ_{i=1}^{n} DF(T + i·Δ) · Δ   Δ = 1/freq (annual by default)

Forward swap rate
  S(T,τ)   = [DF(T) − DF(T+τ)] / A(0,T,τ)

Black-76 PV
  payer:   N·A·[S·Φ(d₁) − K·Φ(d₂)]
  receiver: N·A·[K·Φ(−d₂) − S·Φ(−d₁)]
  d₁ = [ln(S/K) + ½σ²T] / (σ√T)
  d₂ = d₁ − σ√T
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from dataclasses import dataclass
from typing import Literal

from scipy.stats import norm
from scipy.optimize import brentq

from .curve import DiscountCurve


# ── European Swaption ─────────────────────────────────────────────────────────

@dataclass
class Swaption:
    """
    European swaption: the right (not obligation) to enter a fixed/floating
    SOFR OIS swap at expiry.

    Parameters
    ----------
    expiry_years      : Option expiry T in years from the curve reference date.
    swap_tenor_years  : Tenor τ of the underlying swap (years).
    strike            : Fixed rate K of the underlying swap (decimal).
    notional          : Swap notional.
    swaption_type     : 'payer'  → right to pay fixed / receive floating
                        'receiver' → right to receive fixed / pay floating
    vol               : Black-76 annualized lognormal implied vol (decimal).
    freq              : Fixed-leg payment frequency (default 1 = annual).
    """
    expiry_years:     float
    swap_tenor_years: float
    strike:           float
    notional:         float                          = 1_000_000.0
    swaption_type:    Literal["payer", "receiver"]   = "payer"
    vol:              float                          = 0.20
    freq:             int                            = 1

    # ── Annuity ───────────────────────────────────────────────────────────────

    def annuity(self, curve: DiscountCurve) -> float:
        """
        Present value of the fixed-leg annuity discounted to today.

        A(0,T,τ) = Σ_{i=1}^{n} DF(T + i·Δ) · Δ
        where Δ = 1/freq and n = swap_tenor_years × freq.
        """
        dt = 1.0 / self.freq
        n  = round(self.swap_tenor_years * self.freq)
        T  = self.expiry_years
        return sum(curve.df(T + i * dt) * dt for i in range(1, n + 1))

    # ── Forward swap rate ─────────────────────────────────────────────────────

    def forward_swap_rate(self, curve: DiscountCurve) -> float:
        """
        Forward par swap rate at option expiry.

        S(T,τ) = [DF(T) − DF(T+τ)] / A(0,T,τ)
        """
        T   = self.expiry_years
        tau = self.swap_tenor_years
        a   = self.annuity(curve)
        if a < 1e-14:
            return np.nan
        return (curve.df(T) - curve.df(T + tau)) / a

    # ── Black-76 PV ───────────────────────────────────────────────────────────

    def black_pv(self, curve: DiscountCurve) -> float:
        """
        Black-76 present value of the swaption.

        Returns intrinsic value (max(±(S−K), 0) × N × A) when T ≤ 0 or vol ≤ 0,
        so the function is always well-defined.
        """
        T   = self.expiry_years
        S   = self.forward_swap_rate(curve)
        K   = self.strike
        A   = self.annuity(curve)
        N   = self.notional
        vol = self.vol

        if np.isnan(S) or A < 1e-14:
            return 0.0

        # Intrinsic value branch (expired or zero vol)
        if T <= 0.0 or vol <= 0.0:
            if self.swaption_type == "payer":
                return N * A * max(S - K, 0.0)
            else:
                return N * A * max(K - S, 0.0)

        # Black-76 requires S > 0 and K > 0 for log; guard against extreme inputs
        if S <= 0.0 or K <= 0.0:
            if self.swaption_type == "payer":
                return N * A * max(S - K, 0.0)
            else:
                return N * A * max(K - S, 0.0)

        sqrt_T = np.sqrt(T)
        d1 = (np.log(S / K) + 0.5 * vol ** 2 * T) / (vol * sqrt_T)
        d2 = d1 - vol * sqrt_T

        if self.swaption_type == "payer":
            option_pv = S * norm.cdf(d1) - K * norm.cdf(d2)
        else:
            option_pv = K * norm.cdf(-d2) - S * norm.cdf(-d1)

        return N * A * option_pv

    # ── Implied vol ───────────────────────────────────────────────────────────

    def implied_vol(
        self,
        curve: DiscountCurve,
        market_pv: float,
        vol_lo: float = 0.001,
        vol_hi: float = 2.0,
    ) -> float:
        """
        Invert Black-76 to find the implied vol that reproduces ``market_pv``.

        Uses Brent's method on the interval [vol_lo, vol_hi].

        Raises
        ------
        ValueError
            If ``market_pv`` is not bracketed by the PV at vol_lo and vol_hi,
            or if the swaption is expired / has zero annuity.
        """
        T = self.expiry_years
        A = self.annuity(curve)
        if T <= 0.0 or A < 1e-14:
            raise ValueError(
                "Cannot compute implied vol: swaption is expired or annuity is zero."
            )

        def _objective(v: float) -> float:
            tmp = Swaption(
                expiry_years=self.expiry_years,
                swap_tenor_years=self.swap_tenor_years,
                strike=self.strike,
                notional=self.notional,
                swaption_type=self.swaption_type,
                vol=v,
                freq=self.freq,
            )
            return tmp.black_pv(curve) - market_pv

        f_lo = _objective(vol_lo)
        f_hi = _objective(vol_hi)
        if f_lo * f_hi > 0.0:
            raise ValueError(
                f"market_pv={market_pv:.6f} is not bracketed by PV at "
                f"vol_lo={vol_lo} (PV={market_pv + f_lo:.6f}) and "
                f"vol_hi={vol_hi} (PV={market_pv + f_hi:.6f})."
            )
        return brentq(_objective, vol_lo, vol_hi, xtol=1e-10, rtol=1e-10)

    # ── Greeks ────────────────────────────────────────────────────────────────

    def vega(self, curve: DiscountCurve) -> float:
        """
        Vega: ∂PV/∂σ via central finite difference with a 1 bp vol bump.

        Returns the change in PV per 1 bp (0.0001) move in implied vol.
        """
        dv = 1e-4  # 1 bp
        pv_up = Swaption(
            expiry_years=self.expiry_years,
            swap_tenor_years=self.swap_tenor_years,
            strike=self.strike,
            notional=self.notional,
            swaption_type=self.swaption_type,
            vol=self.vol + dv,
            freq=self.freq,
        ).black_pv(curve)
        pv_dn = Swaption(
            expiry_years=self.expiry_years,
            swap_tenor_years=self.swap_tenor_years,
            strike=self.strike,
            notional=self.notional,
            swaption_type=self.swaption_type,
            vol=self.vol - dv,
            freq=self.freq,
        ).black_pv(curve)
        return (pv_up - pv_dn) / 2.0

    def delta(self, curve: DiscountCurve, shift_bps: float = 1.0) -> float:
        """
        Delta: ∂PV/∂S where S is the forward swap rate.

        Computed via central finite difference: bump the forward swap rate by
        ±shift_bps basis points and reprice, holding the strike K fixed.

        Returns PV change per 1 bp move in the forward swap rate.
        """
        ds = shift_bps / 10_000.0
        S_base = self.forward_swap_rate(curve)
        A      = self.annuity(curve)
        T      = self.expiry_years
        vol    = self.vol
        K      = self.strike
        N      = self.notional

        def _pv_shifted(bump: float) -> float:
            S = S_base + bump
            if S <= 0.0 or A < 1e-14:
                if self.swaption_type == "payer":
                    return N * A * max(S - K, 0.0)
                else:
                    return N * A * max(K - S, 0.0)
            if T <= 0.0 or vol <= 0.0:
                if self.swaption_type == "payer":
                    return N * A * max(S - K, 0.0)
                else:
                    return N * A * max(K - S, 0.0)
            sqrt_T = np.sqrt(T)
            d1 = (np.log(S / K) + 0.5 * vol ** 2 * T) / (vol * sqrt_T)
            d2 = d1 - vol * sqrt_T
            if self.swaption_type == "payer":
                option_pv = S * norm.cdf(d1) - K * norm.cdf(d2)
            else:
                option_pv = K * norm.cdf(-d2) - S * norm.cdf(-d1)
            return N * A * option_pv

        return (_pv_shifted(ds) - _pv_shifted(-ds)) / 2.0

    # ── Summary ───────────────────────────────────────────────────────────────

    def summary(self, curve: DiscountCurve) -> dict:
        """
        Full analytics dictionary.

        Keys
        ----
        expiry_years, swap_tenor_years, strike_pct, forward_rate_pct,
        annuity, black_pv, intrinsic_value, time_value, vega, delta,
        moneyness_bps
        """
        S   = self.forward_swap_rate(curve)
        A   = self.annuity(curve)
        pv  = self.black_pv(curve)

        if self.swaption_type == "payer":
            intrinsic = self.notional * A * max(S - self.strike, 0.0)
        else:
            intrinsic = self.notional * A * max(self.strike - S, 0.0)

        # moneyness: positive = in the money for the option holder
        if self.swaption_type == "payer":
            moneyness_bps = (S - self.strike) * 10_000.0
        else:
            moneyness_bps = (self.strike - S) * 10_000.0

        return {
            "expiry_years":     self.expiry_years,
            "swap_tenor_years": self.swap_tenor_years,
            "strike_pct":       self.strike * 100.0,
            "forward_rate_pct": S * 100.0,
            "annuity":          A,
            "black_pv":         pv,
            "intrinsic_value":  intrinsic,
            "time_value":       pv - intrinsic,
            "vega":             self.vega(curve),
            "delta":            self.delta(curve),
            "moneyness_bps":    moneyness_bps,
        }

    def __repr__(self) -> str:
        return (
            f"Swaption({self.swaption_type.capitalize()} "
            f"T={self.expiry_years}y×{self.swap_tenor_years}y, "
            f"K={self.strike*100:.4f}%, vol={self.vol*100:.2f}%, "
            f"N={self.notional:,.0f})"
        )


# ── ATM Swaption Vol Surface ──────────────────────────────────────────────────

class SwaptionVolSurface:
    """
    ATM implied vol surface for swaptions, indexed by (expiry, tenor).

    Vols are stored as lognormal (Black-76) implied vols in decimal form.
    Interpolation is bilinear on the expiry × tenor grid; values outside
    the grid are clamped to the nearest edge (flat extrapolation).

    Parameters
    ----------
    expiries : list of option expiries in years, e.g. [0.5, 1, 2, 5, 10]
    tenors   : list of swap tenors in years,    e.g. [1, 2, 5, 10, 30]
    vols     : 2-D array of shape (n_expiries, n_tenors), decimal vols
    """

    def __init__(
        self,
        expiries: list[float],
        tenors:   list[float],
        vols:     np.ndarray,
    ) -> None:
        expiries_arr = np.asarray(expiries, dtype=float)
        tenors_arr   = np.asarray(tenors,   dtype=float)
        vols_arr     = np.asarray(vols,     dtype=float)

        if vols_arr.shape != (len(expiries_arr), len(tenors_arr)):
            raise ValueError(
                f"vols shape {vols_arr.shape} does not match "
                f"({len(expiries_arr)}, {len(tenors_arr)})"
            )
        if np.any(vols_arr <= 0):
            raise ValueError("All vols must be strictly positive.")
        if np.any(np.diff(expiries_arr) <= 0):
            raise ValueError("expiries must be strictly increasing.")
        if np.any(np.diff(tenors_arr) <= 0):
            raise ValueError("tenors must be strictly increasing.")

        self._expiries = expiries_arr
        self._tenors   = tenors_arr
        self._vols     = vols_arr

    # ── Interpolation ─────────────────────────────────────────────────────────

    def vol(self, expiry: float, tenor: float) -> float:
        """
        Bilinear interpolation of the vol surface at (expiry, tenor).

        Points outside the grid are clamped to the nearest boundary
        (flat extrapolation).
        """
        # Clamp to grid boundaries
        e = float(np.clip(expiry, self._expiries[0], self._expiries[-1]))
        t = float(np.clip(tenor,  self._tenors[0],   self._tenors[-1]))

        # Locate bracketing indices for expiry
        i1 = int(np.searchsorted(self._expiries, e, side="right")) - 1
        i1 = int(np.clip(i1, 0, len(self._expiries) - 2))
        i2 = i1 + 1

        # Locate bracketing indices for tenor
        j1 = int(np.searchsorted(self._tenors, t, side="right")) - 1
        j1 = int(np.clip(j1, 0, len(self._tenors) - 2))
        j2 = j1 + 1

        e1, e2 = self._expiries[i1], self._expiries[i2]
        t1, t2 = self._tenors[j1],   self._tenors[j2]

        # Bilinear weights
        de = (e - e1) / (e2 - e1) if e2 > e1 else 0.0
        dt = (t - t1) / (t2 - t1) if t2 > t1 else 0.0

        v11 = self._vols[i1, j1]
        v12 = self._vols[i1, j2]
        v21 = self._vols[i2, j1]
        v22 = self._vols[i2, j2]

        return float(
            v11 * (1 - de) * (1 - dt)
            + v12 * (1 - de) * dt
            + v21 * de * (1 - dt)
            + v22 * de * dt
        )

    # ── DataFrame view ────────────────────────────────────────────────────────

    @staticmethod
    def _label_years(t: float) -> str:
        """Format a year-fraction as a human-readable label ('6M', '1Y', etc.)."""
        months = round(t * 12)
        if months < 12:
            return f"{months}M"
        years = months // 12
        rem   = months % 12
        if rem == 0:
            return f"{years}Y"
        return f"{years}Y{rem}M"

    def to_dataframe(self) -> pd.DataFrame:
        """
        Return the vol surface as a DataFrame.

        Rows are labeled by option expiry ('6M', '1Y', …);
        columns are labeled by swap tenor ('1Y', '2Y', …).
        Values are vols in percent (decimal × 100).
        """
        row_labels = [self._label_years(e) for e in self._expiries]
        col_labels = [self._label_years(t) for t in self._tenors]
        return pd.DataFrame(
            self._vols * 100.0,
            index=pd.Index(row_labels, name="expiry"),
            columns=pd.Index(col_labels, name="tenor"),
        )

    # ── Factory ───────────────────────────────────────────────────────────────

    @classmethod
    def typical_market(cls) -> "SwaptionVolSurface":
        """
        Construct a representative 2024-vintage USD ATM swaption vol surface.

        Vols are lognormal (Black-76) implied vols in decimal.  They are
        calibrated so that the corresponding normal vols
        (σ_normal ≈ σ_black × S) at a ~5% forward rate lie roughly in the
        40–100 bp range, consistent with observed SOFR swaption markets
        in 2023–2024.

        Surface shape: short-expiry/short-tenor vols are highest; the surface
        is humped in the expiry direction and slopes downward with tenor.
        """
        # Expiries:  6M    1Y    2Y    5Y    10Y
        # Tenors:    1Y    2Y    5Y    10Y   30Y
        expiries = [0.5, 1.0, 2.0,  5.0, 10.0]
        tenors   = [1.0, 2.0, 5.0, 10.0, 30.0]

        # Lognormal vols (decimal).  At a 5% forward rate:
        #   σ_normal (bps) ≈ σ_black × 500 bps
        # so 0.16 lognormal → ~80 bp normal, 0.08 → ~40 bp normal.
        vols = np.array([
            #  1Y      2Y      5Y     10Y     30Y
            [0.165,  0.150,  0.130,  0.110,  0.090],   # 6M expiry
            [0.155,  0.140,  0.125,  0.105,  0.085],   # 1Y expiry
            [0.140,  0.130,  0.115,  0.098,  0.080],   # 2Y expiry
            [0.115,  0.110,  0.100,  0.088,  0.074],   # 5Y expiry
            [0.095,  0.092,  0.086,  0.078,  0.068],   # 10Y expiry
        ])
        return cls(expiries=expiries, tenors=tenors, vols=vols)

    def __repr__(self) -> str:
        return (
            f"SwaptionVolSurface("
            f"expiries={list(self._expiries)}, "
            f"tenors={list(self._tenors)}, "
            f"shape={self._vols.shape})"
        )


# ── Module-level convenience ──────────────────────────────────────────────────

def price_swaption(
    curve: DiscountCurve,
    expiry_years: float,
    swap_tenor_years: float,
    strike: float | None = None,
    notional: float = 1_000_000.0,
    swaption_type: str = "payer",
    vol: float = 0.20,
) -> dict:
    """
    Price a swaption and return a full analytics dictionary.

    Parameters
    ----------
    curve            : Calibrated SOFR discount curve.
    expiry_years     : Option expiry in years from the curve reference date.
    swap_tenor_years : Underlying swap tenor in years.
    strike           : Fixed rate (decimal).  Pass ``None`` to price ATM
                       (strike = forward swap rate at expiry).
    notional         : Swap notional.
    swaption_type    : 'payer' or 'receiver'.
    vol              : Black-76 implied vol (decimal).

    Returns
    -------
    dict
        Output of :meth:`Swaption.summary`.
    """
    if swaption_type not in ("payer", "receiver"):
        raise ValueError(
            f"swaption_type must be 'payer' or 'receiver', got {swaption_type!r}"
        )

    if strike is None:
        # Resolve ATM strike by computing the forward swap rate
        _tmp = Swaption(
            expiry_years=expiry_years,
            swap_tenor_years=swap_tenor_years,
            strike=0.0,          # placeholder
            notional=notional,
            swaption_type=swaption_type,  # type: ignore[arg-type]
            vol=vol,
        )
        strike = _tmp.forward_swap_rate(curve)
        if np.isnan(strike):
            raise ValueError(
                "Cannot determine ATM strike: forward swap rate is NaN. "
                "Check that the curve covers the full expiry + tenor range."
            )

    sw = Swaption(
        expiry_years=expiry_years,
        swap_tenor_years=swap_tenor_years,
        strike=strike,
        notional=notional,
        swaption_type=swaption_type,  # type: ignore[arg-type]
        vol=vol,
    )
    return sw.summary(curve)


# ── Self-test ─────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    from .bootstrap import flat_sofr_curve
    from datetime import date

    ref   = date(2024, 6, 5)
    curve = flat_sofr_curve(ref, rate=0.0530)

    # --- ATM payer swaption ---
    atm_strike = Swaption(
        expiry_years=1.0, swap_tenor_years=5.0, strike=0.0
    ).forward_swap_rate.__func__  # resolved below
    sw_atm = Swaption(
        expiry_years=1.0,
        swap_tenor_years=5.0,
        strike=curve.par_ois_rate(6.0),   # ATM ≈ expiry + tenor forward rate
        notional=10_000_000,
        swaption_type="payer",
        vol=0.20,
    )
    print("1Y x 5Y ATM Payer Swaption:")
    for k, v in sw_atm.summary(curve).items():
        print(f"  {k:<22}: {v:.6f}" if isinstance(v, float) else f"  {k:<22}: {v}")

    # --- Implied vol round-trip ---
    mkt_pv = sw_atm.black_pv(curve)
    iv     = sw_atm.implied_vol(curve, mkt_pv)
    assert abs(iv - sw_atm.vol) < 1e-8, f"Implied vol round-trip failed: {iv} != {sw_atm.vol}"
    print(f"\nImplied vol round-trip: {iv*100:.4f}%  OK")

    # --- Put-call parity: payer PV - receiver PV = N x A x (S - K) ---
    sw_rec = Swaption(
        expiry_years=1.0,
        swap_tenor_years=5.0,
        strike=sw_atm.strike,
        notional=10_000_000,
        swaption_type="receiver",
        vol=0.20,
    )
    S        = sw_atm.forward_swap_rate(curve)
    A        = sw_atm.annuity(curve)
    parity   = sw_atm.black_pv(curve) - sw_rec.black_pv(curve)
    expected = sw_atm.notional * A * (S - sw_atm.strike)
    assert abs(parity - expected) < 1.0, (
        f"Put-call parity failed: {parity:.2f} != {expected:.2f}"
    )
    print(f"Put-call parity check: {parity:.2f} approx {expected:.2f}  OK")

    # --- Vol surface ---
    surf = SwaptionVolSurface.typical_market()
    print("\nTypical market vol surface (%):")
    print(surf.to_dataframe().to_string())

    # --- price_swaption convenience function ---
    result = price_swaption(curve, expiry_years=2.0, swap_tenor_years=10.0)
    print(f"\nprice_swaption 2Y x 10Y ATM payer: PV = {result['black_pv']:,.2f}")

    print("\nAll swaption tests passed.")
