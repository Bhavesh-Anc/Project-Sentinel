"""
SOFR Forward Curve Bootstrapper.

Constructs a SOFR OIS discount curve from:
  1. Overnight SOFR rate  → anchor at T=1 business day
  2. CME SR3 futures      → short-to-medium end (0–2 years), convexity-adjusted
  3. SOFR OIS swap quotes → long end (2–30 years)

Algorithm
---------
Step 1: Anchor
    DF(0) = 1.0  (by definition)
    DF(T_ON) = 1 / (1 + r_SOFR × d_ON/360)   where d_ON = 1 or 2 (T+1 settlement)

Step 2: Futures strip (SR3 contracts, quarterly IMM dates)
    Each contract gives an implied forward rate for period [T_start, T_end].
    Convexity-adjusted forward rate: f_fwd = f_fut - CA(T_start, T_end)
    DF(T_end) = DF(T_start) × exp(-f_fwd × (T_end - T_start))

Step 3: OIS swaps (for tenors beyond futures curve)
    For each swap tenor T_n with par rate K_n:
    Annuity(T_n) = Σᵢ αᵢ × DF(Tᵢ)   [payment times already bootstrapped]
    DF(T_n) = [DF(0) - K_n × Annuity(T_n_prev)] / [1 + K_n × α_n]
    → Sequential bootstrapping, one swap at a time

Output: DiscountCurve object valid from T=0 to T=max_tenor.
"""
from __future__ import annotations
import numpy as np
import pandas as pd
from datetime import date, timedelta
from typing import Sequence

from .curve import DiscountCurve
from .convexity import futures_to_forward
from .day_count import act360


class SOFRCurveBootstrapper:
    """
    Bootstrap a SOFR OIS discount curve from market quotes.

    Parameters
    ----------
    ref_date        : pricing date (today)
    sofr_overnight  : overnight SOFR fixing (decimal, e.g. 0.0530)
    sigma           : Hull-White short-rate vol for convexity adj (default 1%)
    mean_reversion  : Hull-White mean-reversion a (default 0 = simpler formula)
    """

    def __init__(
        self,
        ref_date: date,
        sofr_overnight: float,
        sigma: float = 0.010,
        mean_reversion: float = 0.0,
    ):
        self.ref_date = ref_date
        self.sofr_overnight = sofr_overnight
        self.sigma = sigma
        self.mean_reversion = mean_reversion

        # Pillar lists — populated by add_futures / add_ois_swaps / bootstrap
        self._times: list[float] = [0.0]
        self._dfs:   list[float] = [1.0]

        # Seed overnight
        self._add_overnight()

    def _add_overnight(self) -> None:
        """Add overnight deposit pillar: DF(T+1) = 1/(1 + r × 1/360)."""
        t_on = 1 / 365.25  # ~1 business day
        df_on = 1.0 / (1.0 + self.sofr_overnight * 1 / 360.0)
        self._times.append(t_on)
        self._dfs.append(df_on)

    def add_futures(
        self,
        futures_data: pd.DataFrame,
        contract_type: str = "SR3",
    ) -> "SOFRCurveBootstrapper":
        """
        Add SR3 (3-month) or SR1 (1-month) SOFR futures to the curve.

        Parameters
        ----------
        futures_data : DataFrame with columns:
            expiry        (date) — futures contract expiry (IMM date)
            accrual_end   (date) — end of accrual period
            implied_rate  (float) — (100 - price) / 100
        contract_type : 'SR3' or 'SR1'
        """
        df_sorted = futures_data.sort_values("expiry").reset_index(drop=True)

        for _, row in df_sorted.iterrows():
            exp      = row["expiry"]
            acc_end  = row["accrual_end"]
            f_fut    = float(row["implied_rate"])

            t1 = (exp     - self.ref_date).days / 365.25
            t2 = (acc_end - self.ref_date).days / 365.25

            # Skip contracts that have already expired
            if t2 <= self._times[-1] + 1e-6:
                continue

            # Convexity adjustment: futures → forward
            f_fwd = futures_to_forward(f_fut, t1, t2, self.sigma, self.mean_reversion)

            # Discount factor at accrual_end using existing curve up to expiry
            t1_clamped = max(t1, self._times[-1])
            df_t1 = self._interp_df(t1_clamped)
            tenor = t2 - t1_clamped
            if tenor < 1e-6:
                continue

            df_t2 = df_t1 * np.exp(-f_fwd * tenor)

            # Only extend curve forward (never go back)
            if t2 > self._times[-1] + 1e-6:
                self._times.append(t2)
                self._dfs.append(df_t2)

        return self

    def add_ois_swaps(
        self,
        swap_quotes: list[tuple[float, float]],
        payment_freq: int = 1,
    ) -> "SOFRCurveBootstrapper":
        """
        Bootstrap the long end from SOFR OIS swap par rates.

        Parameters
        ----------
        swap_quotes  : list of (tenor_years, par_rate_decimal)
                       e.g. [(2.0, 0.0480), (5.0, 0.0440), (10.0, 0.0420)]
        payment_freq : fixed leg payment frequency (1=annual, 2=semi-annual)
        """
        dt = 1.0 / payment_freq

        for tenor, par_rate in sorted(swap_quotes, key=lambda x: x[0]):
            payment_times = np.arange(dt, tenor + 1e-10, dt)

            # Compute annuity for all payment times except the last
            annuity_known = 0.0
            for t_pay in payment_times[:-1]:
                annuity_known += dt * self._interp_df(t_pay)

            # Bootstrap: solve for DF at swap maturity
            # K × Σᵢ αᵢ DF(Tᵢ) + DF(T_n) = 1  (OIS: floating = DF start - DF end)
            # K × annuity_known + K × dt × DF(T_n) + DF(T_n) = 1
            # DF(T_n) × (1 + K × dt) = 1 - K × annuity_known
            K = par_rate
            denom = 1.0 + K * dt
            df_swap = (1.0 - K * annuity_known) / denom

            if df_swap <= 0:
                raise ValueError(f"Bootstrapped negative DF at tenor {tenor}y — check swap quotes")

            t_n = tenor
            if t_n > self._times[-1] + 1e-6:
                self._times.append(t_n)
                self._dfs.append(df_swap)

        return self

    def build(self) -> DiscountCurve:
        """Construct and return the bootstrapped DiscountCurve."""
        return DiscountCurve(
            ref_date=self.ref_date,
            times=self._times,
            dfs=self._dfs,
            label="SOFR OIS",
        )

    def _interp_df(self, t: float) -> float:
        """Log-linear interpolation on current pillar set."""
        log_dfs = np.log(self._dfs)
        return float(np.exp(np.interp(t, self._times, log_dfs)))

    # ── Convenience class method ──────────────────────────────────────────────

    @classmethod
    def from_market_data(
        cls,
        ref_date: date,
        sofr_overnight: float,
        futures_df: pd.DataFrame | None = None,
        ois_quotes: list[tuple[float, float]] | None = None,
        sigma: float = 0.010,
    ) -> DiscountCurve:
        """
        One-shot builder.

        Parameters
        ----------
        ref_date       : pricing date
        sofr_overnight : overnight SOFR (decimal)
        futures_df     : DataFrame from SOFRFuturesCurve.contracts (or None)
        ois_quotes     : [(tenor_yrs, par_rate_decimal), ...] (or None)
        sigma          : HW short-rate vol for convexity adjustment

        Returns
        -------
        DiscountCurve ready to use for pricing
        """
        builder = cls(ref_date, sofr_overnight, sigma=sigma)

        if futures_df is not None and not futures_df.empty:
            builder.add_futures(futures_df)

        if ois_quotes:
            builder.add_ois_swaps(ois_quotes)

        return builder.build()


# ── Flat-curve constructor (for testing / benchmarking) ──────────────────────

def flat_sofr_curve(ref_date: date, rate: float, max_tenor: float = 30.0) -> DiscountCurve:
    """
    Construct a flat SOFR OIS curve at a single rate.
    Useful for unit-testing pricers (PV of par swap should be 0).

    DF(T) = exp(-rate × T)
    """
    tenors = np.array([0.0, 0.25, 0.5, 1, 2, 3, 5, 7, 10, 15, 20, 30])
    tenors = tenors[tenors <= max_tenor + 1e-6]
    dfs = np.exp(-rate * tenors)
    return DiscountCurve(ref_date, tenors, dfs, label=f"Flat {rate*100:.2f}%")


if __name__ == "__main__":
    from datetime import date
    import pandas as pd

    ref = date(2024, 6, 5)
    sofr_on = 0.0530  # 5.30% overnight

    # Synthetic futures strip (SR3 quarterly contracts)
    futures = pd.DataFrame([
        {"expiry": date(2024, 9, 18), "accrual_end": date(2024, 12, 18), "implied_rate": 0.0520},
        {"expiry": date(2024, 12, 18), "accrual_end": date(2025, 3, 19), "implied_rate": 0.0490},
        {"expiry": date(2025, 3, 19),  "accrual_end": date(2025, 6, 18), "implied_rate": 0.0460},
        {"expiry": date(2025, 6, 18),  "accrual_end": date(2025, 9, 17), "implied_rate": 0.0435},
        {"expiry": date(2025, 9, 17),  "accrual_end": date(2025, 12, 17), "implied_rate": 0.0415},
        {"expiry": date(2025, 12, 17), "accrual_end": date(2026, 3, 18), "implied_rate": 0.0400},
    ])

    # Synthetic OIS quotes for longer tenors
    ois = [(2.0, 0.0448), (3.0, 0.0430), (5.0, 0.0415), (10.0, 0.0400), (30.0, 0.0395)]

    curve = SOFRCurveBootstrapper.from_market_data(ref, sofr_on, futures, ois)

    print(f"\n{curve}")
    print("\nZero Curve:")
    print(curve.zero_curve().to_string(index=False))
