"""
DiscountCurve — core data structure for the SOFR pricing engine.

Stores a set of (time, discount_factor) pillar points and interpolates
using either:
  - log-linear (default): piecewise-flat forward rates, guaranteed positive,
    market-standard for simple curve stripping
  - pchip: monotone cubic Hermite spline on log-DFs, producing smooth C1
    continuous forward rates with no kinks at pillar boundaries

Time units: years (as floats), measured from the curve's reference date.
"""
from __future__ import annotations
import numpy as np
import pandas as pd
from datetime import date, datetime
from typing import Sequence

from scipy.interpolate import PchipInterpolator


class DiscountCurve:
    """
    Discount curve with log-linear or PCHIP spline interpolation.

    Parameters
    ----------
    ref_date      : curve anchor (today's date)
    times         : sorted array of pillar times in years from ref_date
    dfs           : discount factors at each pillar, DF(0) = 1.0
    label         : optional name (e.g. 'SOFR OIS', 'Treasury')
    interp_method : 'log-linear' (default, piecewise-flat forwards) or
                    'pchip' (monotone cubic spline, smooth C1 forward curve)
    """

    def __init__(
        self,
        ref_date: date,
        times: Sequence[float],
        dfs: Sequence[float],
        label: str = "SOFR",
        interp_method: str = "log-linear",
    ):
        self.ref_date = ref_date
        self.label = label
        self._interp_method = interp_method

        if interp_method not in ("log-linear", "pchip"):
            raise ValueError(f"interp_method must be 'log-linear' or 'pchip', got {interp_method!r}")

        times_arr = np.asarray(times, dtype=float)
        dfs_arr   = np.asarray(dfs,   dtype=float)

        # Prepend time=0, DF=1 if not already present
        if times_arr[0] > 1e-10:
            times_arr = np.concatenate([[0.0], times_arr])
            dfs_arr   = np.concatenate([[1.0], dfs_arr])

        if np.any(dfs_arr <= 0):
            raise ValueError("All discount factors must be strictly positive")
        if np.any(np.diff(times_arr) <= 0):
            raise ValueError("Pillar times must be strictly increasing")

        self._times  = times_arr
        self._log_df = np.log(dfs_arr)

        # Build PCHIP interpolant on log-DFs for smooth forward curves
        if interp_method == "pchip":
            self._pchip = PchipInterpolator(self._times, self._log_df, extrapolate=True)
        else:
            self._pchip = None

    # ── Core interpolation ────────────────────────────────────────────────────

    def df(self, t: float | np.ndarray) -> float | np.ndarray:
        """
        Discount factor at time t (years).

        Uses PCHIP spline (smooth C1 forwards) when interp_method='pchip',
        or log-linear (piecewise-flat forwards) otherwise.
        """
        if self._pchip is not None:
            log_df = self._pchip(t)
            return float(np.exp(log_df)) if np.ndim(t) == 0 else np.exp(log_df)
        log_df = np.interp(t, self._times, self._log_df)
        return np.exp(log_df)

    def df_date(self, d: date) -> float:
        """Discount factor for a calendar date."""
        return self.df(self._t(d))

    def zero_rate(self, t: float, compounding: str = "continuous") -> float:
        """
        Zero (spot) rate at time t.

        compounding : 'continuous' → z = -ln(DF)/t
                      'annual'     → z = DF^(-1/t) - 1
                      'act360'     → z = (1/DF - 1) * 360/t_days (money market)
        """
        d = self.df(t)
        if t < 1e-10:
            return 0.0
        if compounding == "continuous":
            return -np.log(d) / t
        elif compounding == "annual":
            return d ** (-1.0 / t) - 1.0
        elif compounding == "act360":
            return (1.0 / d - 1.0) / t  # t already in years (ACT/360)
        else:
            raise ValueError(f"Unknown compounding: {compounding}")

    def forward_rate(self, t1: float, t2: float) -> float:
        """
        Continuously compounded forward rate for the period [t1, t2].
        f(t1,t2) = -ln(DF(t2)/DF(t1)) / (t2 - t1)
        """
        if t2 <= t1:
            raise ValueError("t2 must be > t1")
        return -np.log(self.df(t2) / self.df(t1)) / (t2 - t1)

    def forward_rate_dates(self, start: date, end: date) -> float:
        """Forward rate between two calendar dates."""
        return self.forward_rate(self._t(start), self._t(end))

    def par_ois_rate(self, maturity_years: float, payment_freq: int = 1) -> float:
        """
        Par OIS swap rate for a given tenor.
        Fixed leg: annual (default) or semi-annual payments.
        Floating leg (OIS/SOFR): DF(0) - DF(T) (present value of floating = notional difference).

        K = [DF(0) - DF(T)] / Σᵢ αᵢ DF(Tᵢ)
        """
        dt = 1.0 / payment_freq
        payment_times = np.arange(dt, maturity_years + 1e-10, dt)
        annuity = sum(dt * self.df(t) for t in payment_times)
        if annuity < 1e-12:
            return np.nan
        return (1.0 - self.df(maturity_years)) / annuity

    # ── Curve analytics ───────────────────────────────────────────────────────

    def zero_curve(self, tenors: Sequence[float] | None = None) -> pd.DataFrame:
        """Return a DataFrame of tenors, zero rates, and discount factors."""
        if tenors is None:
            tenors = [0.25, 0.5, 1, 2, 3, 5, 7, 10, 15, 20, 30]
        rows = []
        for t in tenors:
            rows.append({
                "tenor_yrs":   t,
                "discount_factor": self.df(t),
                "zero_rate_pct":   self.zero_rate(t) * 100,
                "fwd_1y_pct":      self.forward_rate(t, t + 1.0) * 100 if t + 1 <= self._times[-1] else np.nan,
            })
        return pd.DataFrame(rows)

    def dv01(self, t: float, notional: float = 1_000_000) -> float:
        """
        Approximate DV01 (dollar value of 1bp) for a zero-coupon instrument.
        DV01 = notional × t × DF(t) / 10_000
        """
        return notional * t * self.df(t) / 10_000.0

    # ── Pillar access ─────────────────────────────────────────────────────────

    @property
    def pillars(self) -> pd.DataFrame:
        """Return pillar times and discount factors as a DataFrame."""
        return pd.DataFrame({
            "time_yrs": self._times,
            "df":        np.exp(self._log_df),
            "zero_pct":  [-np.log(np.exp(ldf)) / t * 100 if t > 1e-10 else 0.0
                          for t, ldf in zip(self._times, self._log_df)],
        })

    # ── Helpers ───────────────────────────────────────────────────────────────

    def _t(self, d: date) -> float:
        """Convert a calendar date to time in years from ref_date (ACT/365.25)."""
        return (d - self.ref_date).days / 365.25

    def __repr__(self) -> str:
        return (
            f"DiscountCurve(label='{self.label}', ref={self.ref_date}, "
            f"pillars={len(self._times)}, max_tenor={self._times[-1]:.1f}y, "
            f"interp={self._interp_method})"
        )
