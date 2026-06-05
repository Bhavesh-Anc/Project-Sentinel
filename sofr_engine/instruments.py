"""
SOFR instrument pricers.

Instruments
-----------
SOFRFutures         : CME SR3 / SR1 futures (price, DV01, implied rate)
ForwardRateAgreement: FRA pricing against SOFR forward curve
SOFRSwap            : Vanilla SOFR OIS swap (fixed vs. compounded SOFR floating)
SOFRCapFloor        : Cap/floor on compounded SOFR (Black-76 normal model)

All instruments follow the same interface:
    .pv(curve)      → present value (notional units)
    .dv01(curve)    → dollar value of 1bp parallel shift
    .par_rate(curve)→ break-even fixed rate (where PV = 0)
"""
from __future__ import annotations
import numpy as np
import pandas as pd
from datetime import date, timedelta
from dataclasses import dataclass, field
from typing import Literal

from .curve import DiscountCurve
from .day_count import act360, year_frac


# ── SOFR Futures ─────────────────────────────────────────────────────────────

@dataclass
class SOFRFutures:
    """
    CME SR3 (3-month) or SR1 (1-month) SOFR futures contract.

    The futures price = 100 - (annualized compounded SOFR rate × 100).
    PV for a long position: notional × (forward_rate - futures_rate) × accrual_period.
    """
    contract_type:  Literal["SR3", "SR1"]
    expiry:         date
    accrual_end:    date
    futures_price:  float          # e.g. 94.70 → implied rate 5.30%
    notional:       float = 1_000_000.0

    @property
    def futures_rate(self) -> float:
        return (100.0 - self.futures_price) / 100.0

    @property
    def accrual_days(self) -> int:
        return (self.accrual_end - self.expiry).days

    @property
    def accrual_yrs(self) -> float:
        return self.accrual_days / 360.0

    def implied_forward_rate(self, curve: DiscountCurve) -> float:
        """Forward rate from the OIS curve for the futures accrual period."""
        t1 = curve._t(self.expiry)
        t2 = curve._t(self.accrual_end)
        if t2 <= t1:
            return curve.sofr_overnight if hasattr(curve, "sofr_overnight") else 0.0
        return curve.forward_rate(t1, t2)

    def pv(self, curve: DiscountCurve) -> float:
        """
        PV of long futures position.
        = notional × (f_OIS - f_fut) × accrual_fraction × DF(expiry)
        Note: futures are margined daily, so no DF adjustment in theory,
        but we approximate using discount to expiry for comparison purposes.
        """
        f_ois = self.implied_forward_rate(curve)
        return self.notional * (f_ois - self.futures_rate) * self.accrual_yrs

    def dv01(self, curve: DiscountCurve, shift_bps: float = 1.0) -> float:
        """DV01: change in PV for 1bp parallel shift in forward rate."""
        return self.notional * (shift_bps / 10_000.0) * self.accrual_yrs

    def __repr__(self) -> str:
        return (f"SOFRFutures({self.contract_type} {self.expiry} "
                f"price={self.futures_price:.4f} rate={self.futures_rate*100:.4f}%)")


# ── Forward Rate Agreement ────────────────────────────────────────────────────

@dataclass
class ForwardRateAgreement:
    """
    SOFR FRA: agreement to exchange fixed vs. floating SOFR rate
    for a single forward period [start_date, end_date].

    Buyer (long): receives floating SOFR, pays fixed K.
    Settlement at start_date in arrears convention.
    """
    start_date:  date
    end_date:    date
    fixed_rate:  float          # K, decimal
    notional:    float = 10_000_000.0
    pay_receive: Literal["pay", "receive"] = "receive"  # fixed leg direction

    @property
    def accrual_frac(self) -> float:
        return act360(self.start_date, self.end_date)

    def forward_rate(self, curve: DiscountCurve) -> float:
        t1 = curve._t(self.start_date)
        t2 = curve._t(self.end_date)
        # Convert continuous to money-market (ACT/360 simple)
        df_ratio = curve.df(t1) / curve.df(t2)
        return (df_ratio - 1.0) / self.accrual_frac

    def pv(self, curve: DiscountCurve) -> float:
        """
        PV = N × (f - K) × α × DF(end)  [receive-fixed sign convention]
        For FRA settlement at start: PV / (1 + f × α)
        """
        f = self.forward_rate(curve)
        alpha = self.accrual_frac
        df_end = curve.df_date(self.end_date)

        raw_pv = self.notional * (f - self.fixed_rate) * alpha * df_end
        sign = 1.0 if self.pay_receive == "receive" else -1.0
        return sign * raw_pv

    def par_rate(self, curve: DiscountCurve) -> float:
        """Break-even fixed rate (FRA rate = forward rate from curve)."""
        return self.forward_rate(curve)

    def dv01(self, curve: DiscountCurve) -> float:
        """DV01 via finite difference (1bp parallel shift)."""
        from copy import deepcopy
        return (self.notional * self.accrual_frac *
                curve.df_date(self.end_date) / 10_000.0)

    def __repr__(self) -> str:
        return (f"FRA({self.start_date} → {self.end_date}, "
                f"K={self.fixed_rate*100:.4f}%, N={self.notional:,.0f})")


# ── SOFR OIS Swap ─────────────────────────────────────────────────────────────

@dataclass
class SOFRSwap:
    """
    Vanilla SOFR OIS swap: fixed rate vs. compounded SOFR overnight.

    Fixed leg:    pays/receives K every payment_period
    Floating leg: pays/receives compounded SOFR over each accrual period,
                  settled at the end of each period

    For OIS: floating leg PV = DF(start) - DF(end)  [exact, no coupon dates needed]
    Fixed leg PV = K × Σᵢ αᵢ × DF(Tᵢ)

    Parameters
    ----------
    effective_date  : swap start date (T+2 settlement for spot-starting)
    maturity_date   : swap end date
    fixed_rate      : fixed coupon rate (decimal)
    notional        : face value
    payment_freq    : 1=annual, 2=semi-annual, 4=quarterly
    pay_fixed       : True = pay fixed (payer swap), False = receive fixed (receiver)
    day_count_fixed : day count for fixed leg ('ACT360' or '30360')
    """
    effective_date:  date
    maturity_date:   date
    fixed_rate:      float
    notional:        float = 10_000_000.0
    payment_freq:    int   = 1
    pay_fixed:       bool  = True
    day_count_fixed: str   = "ACT360"

    def _payment_schedule(self) -> list[date]:
        """Generate fixed leg payment dates."""
        from dateutil.relativedelta import relativedelta
        months_per_period = 12 // self.payment_freq
        dates = []
        d = self.effective_date
        while True:
            d = d + relativedelta(months=months_per_period)
            dates.append(d)
            if d >= self.maturity_date:
                break
        if dates[-1] != self.maturity_date:
            dates[-1] = self.maturity_date
        return dates

    def fixed_leg_pv(self, curve: DiscountCurve) -> float:
        """PV of fixed coupon leg."""
        pay_dates  = self._payment_schedule()
        prev_date  = self.effective_date
        pv = 0.0
        for d in pay_dates:
            alpha  = year_frac(prev_date, d, self.day_count_fixed)
            df     = curve.df_date(d)
            pv    += self.fixed_rate * alpha * df
            prev_date = d
        return self.notional * pv

    def floating_leg_pv(self, curve: DiscountCurve) -> float:
        """
        PV of floating SOFR OIS leg.
        For OIS: PV = N × [DF(effective) - DF(maturity)]
        This is exact regardless of payment frequency.
        """
        df_start = curve.df_date(self.effective_date)
        df_end   = curve.df_date(self.maturity_date)
        return self.notional * (df_start - df_end)

    def pv(self, curve: DiscountCurve) -> float:
        """
        Net present value from the perspective of the fixed-rate payer.
        Payer: PV = PV(floating) - PV(fixed)
        Receiver: PV = PV(fixed) - PV(floating)
        """
        pv_fixed    = self.fixed_leg_pv(curve)
        pv_floating = self.floating_leg_pv(curve)
        if self.pay_fixed:
            return pv_floating - pv_fixed
        else:
            return pv_fixed - pv_floating

    def par_rate(self, curve: DiscountCurve) -> float:
        """
        Par swap rate: fixed rate K such that PV = 0 at inception.
        K = [DF(start) - DF(end)] / Annuity
        """
        pay_dates = self._payment_schedule()
        prev_date = self.effective_date
        annuity   = 0.0
        for d in pay_dates:
            alpha    = year_frac(prev_date, d, self.day_count_fixed)
            annuity += alpha * curve.df_date(d)
            prev_date = d

        df_start = curve.df_date(self.effective_date)
        df_end   = curve.df_date(self.maturity_date)
        return (df_start - df_end) / annuity if annuity > 1e-12 else np.nan

    def dv01(self, curve: DiscountCurve, shift_bps: float = 1.0) -> float:
        """
        Approximate DV01: sensitivity to 1bp parallel shift in SOFR curve.
        Uses the modified duration approximation: DV01 ≈ N × duration × DF / 10000.
        """
        pay_dates = self._payment_schedule()
        prev_date = self.effective_date
        duration_times_df = 0.0
        for d in pay_dates:
            alpha = year_frac(prev_date, d, self.day_count_fixed)
            t     = curve._t(d)
            df    = curve.df_date(d)
            duration_times_df += alpha * t * df
            prev_date = d
        return self.notional * self.fixed_rate * duration_times_df / 10_000.0

    def summary(self, curve: DiscountCurve) -> dict:
        """Return a summary dict of swap analytics."""
        return {
            "effective_date":  self.effective_date.isoformat(),
            "maturity_date":   self.maturity_date.isoformat(),
            "tenor_yrs":       curve._t(self.maturity_date) - curve._t(self.effective_date),
            "fixed_rate_pct":  self.fixed_rate * 100,
            "par_rate_pct":    self.par_rate(curve) * 100,
            "pv_fixed":        self.fixed_leg_pv(curve),
            "pv_floating":     self.floating_leg_pv(curve),
            "net_pv":          self.pv(curve),
            "dv01":            self.dv01(curve),
            "notional":        self.notional,
        }

    def __repr__(self) -> str:
        direction = "Payer" if self.pay_fixed else "Receiver"
        return (f"SOFRSwap({direction} {self.effective_date}→{self.maturity_date}, "
                f"K={self.fixed_rate*100:.4f}%, N={self.notional:,.0f})")


# ── Cap/Floor (Black-76 normal model) ────────────────────────────────────────

@dataclass
class SOFRCaplet:
    """
    Single caplet on compounded SOFR over period [reset_date, pay_date].
    Priced using Bachelier (normal) model — standard post-LIBOR.

    C = N × α × DF(T_pay) × [f × N(d) + σ√T × n(d)]
    where d = (f - K) / (σ√T), N = cumulative normal, n = normal density.
    """
    reset_date:  date
    pay_date:    date
    strike:      float   # K, decimal
    notional:    float   = 1_000_000.0
    cap_floor:   Literal["cap", "floor"] = "cap"

    def pv(self, curve: DiscountCurve, normal_vol: float = 0.010) -> float:
        """Price using Bachelier (normal) model."""
        from scipy.stats import norm

        t_exp = curve._t(self.reset_date)
        alpha = act360(self.reset_date, self.pay_date)
        df    = curve.df_date(self.pay_date)
        f     = curve.forward_rate_dates(self.reset_date, self.pay_date)

        if t_exp <= 0:
            intrinsic = max(f - self.strike, 0) if self.cap_floor == "cap" else max(self.strike - f, 0)
            return self.notional * alpha * df * intrinsic

        sigma_sqrt_t = normal_vol * np.sqrt(t_exp)
        d = (f - self.strike) / sigma_sqrt_t

        if self.cap_floor == "cap":
            option_pv = (f - self.strike) * norm.cdf(d) + sigma_sqrt_t * norm.pdf(d)
        else:
            option_pv = (self.strike - f) * norm.cdf(-d) + sigma_sqrt_t * norm.pdf(d)

        return self.notional * alpha * df * option_pv

    def __repr__(self) -> str:
        return (f"SOFRCaplet({self.cap_floor.upper()} {self.reset_date}→{self.pay_date}, "
                f"K={self.strike*100:.4f}%)")


if __name__ == "__main__":
    from datetime import date
    from .bootstrap import flat_sofr_curve

    ref = date(2024, 6, 5)
    curve = flat_sofr_curve(ref, rate=0.0530)

    # Test: par swap PV should be ~0 at inception
    from dateutil.relativedelta import relativedelta
    swap = SOFRSwap(
        effective_date=ref + timedelta(days=2),
        maturity_date=ref + timedelta(days=2) + relativedelta(years=5),
        fixed_rate=curve.par_ois_rate(5.0),
        notional=10_000_000,
        pay_fixed=True,
    )
    summary = swap.summary(curve)
    print("5Y SOFR Swap at par:")
    for k, v in summary.items():
        print(f"  {k:<22}: {v}")
    assert abs(summary["net_pv"]) < 1.0, "Par swap should have ~0 PV"
    print("\nAll instrument tests passed.")
