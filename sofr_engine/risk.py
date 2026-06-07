"""
Risk analytics for SOFR curves and instrument portfolios.

Classes
-------
ScenarioEngine  : build shifted curves (parallel, twist, key-rate, butterfly)
RiskReport      : aggregate DV01 ladder, scenario grid, VaR for a portfolio
"""
from __future__ import annotations
import numpy as np
import pandas as pd
from dataclasses import dataclass, field
from typing import Sequence, Protocol

from .curve import DiscountCurve


# ── Shifted curve factory ─────────────────────────────────────────────────────

class ScenarioEngine:
    """Create hypothetical shifted curves for scenario analysis."""

    @staticmethod
    def parallel_shift(curve: DiscountCurve, shift_bps: float) -> DiscountCurve:
        """
        Shift all zero rates by shift_bps in parallel.
        Achieves this by multiplying every discount factor by exp(-shift * t).
        """
        shift = shift_bps / 10_000.0
        new_log_df = curve._log_df - shift * curve._times
        new_dfs    = np.exp(new_log_df)
        return DiscountCurve(
            curve.ref_date,
            curve._times[1:],   # drop t=0 anchor
            new_dfs[1:],
            label=f"{curve.label} +{shift_bps:+.0f}bp",
        )

    @staticmethod
    def key_rate_shift(
        curve: DiscountCurve,
        key_tenor: float,
        shift_bps: float,
        width: float = 2.0,
    ) -> DiscountCurve:
        """
        Key-rate duration shift: triangular shift centred on key_tenor,
        tapering linearly to zero at key_tenor ± width years.
        """
        shift        = shift_bps / 10_000.0
        times        = curve._times
        bump_profile = np.zeros_like(times)
        for i, t in enumerate(times):
            dist = abs(t - key_tenor)
            if dist < width:
                bump_profile[i] = (1.0 - dist / width) * shift
        new_log_df = curve._log_df - bump_profile * times
        new_dfs    = np.exp(new_log_df)
        return DiscountCurve(
            curve.ref_date,
            times[1:],
            new_dfs[1:],
            label=f"{curve.label} KR{key_tenor}Y {shift_bps:+.0f}bp",
        )

    @staticmethod
    def twist_shift(
        curve: DiscountCurve,
        short_shift_bps: float,
        long_shift_bps: float,
        pivot: float = 5.0,
    ) -> DiscountCurve:
        """
        Linear twist: shift at t=0 equals short_shift_bps, shifts linearly
        to long_shift_bps at the long end; interpolated through pivot.
        """
        max_t = curve._times[-1]
        times = curve._times
        # Interpolate linearly: at t<=pivot from short, at t>=pivot to long
        interp_shift = np.interp(times, [0.0, pivot, max_t],
                                 [short_shift_bps, (short_shift_bps + long_shift_bps) / 2,
                                  long_shift_bps])
        shift_arr  = interp_shift / 10_000.0
        new_log_df = curve._log_df - shift_arr * times
        new_dfs    = np.exp(new_log_df)
        return DiscountCurve(
            curve.ref_date,
            times[1:],
            new_dfs[1:],
            label=f"{curve.label} twist {short_shift_bps:+.0f}/{long_shift_bps:+.0f}bp",
        )

    @staticmethod
    def butterfly_shift(
        curve: DiscountCurve,
        wing_shift_bps: float,
        belly_shift_bps: float,
        wing1: float = 2.0,
        belly: float = 5.0,
        wing2: float = 10.0,
    ) -> DiscountCurve:
        """
        Butterfly shift: wings shift by wing_shift_bps, belly by belly_shift_bps.
        Interpolated between the three key points.
        """
        times = curve._times
        knots = [0.0, wing1, belly, wing2, times[-1]]
        vals  = [wing_shift_bps, wing_shift_bps, belly_shift_bps, wing_shift_bps, wing_shift_bps]
        shift_arr  = np.interp(times, knots, vals) / 10_000.0
        new_log_df = curve._log_df - shift_arr * times
        new_dfs    = np.exp(new_log_df)
        return DiscountCurve(
            curve.ref_date,
            times[1:],
            new_dfs[1:],
            label=f"{curve.label} fly belly{belly_shift_bps:+.0f}bp",
        )


# ── Instrument protocol ───────────────────────────────────────────────────────

class Priceable(Protocol):
    """Any object with a .pv(curve) method."""
    def pv(self, curve: DiscountCurve) -> float: ...


# ── Portfolio risk report ─────────────────────────────────────────────────────

KEY_RATE_TENORS = [0.25, 0.5, 1.0, 2.0, 3.0, 5.0, 7.0, 10.0, 15.0, 20.0, 30.0]


class RiskReport:
    """
    Aggregate risk metrics for a named portfolio of SOFR instruments.

    Parameters
    ----------
    instruments : dict[str, Priceable]  — name → instrument
    curve       : pricing curve
    """

    def __init__(
        self,
        instruments: dict[str, "Priceable"],
        curve: DiscountCurve,
    ):
        self.instruments = instruments
        self.curve       = curve
        self._base_pvs   = {n: i.pv(curve) for n, i in instruments.items()}

    @property
    def total_pv(self) -> float:
        return sum(self._base_pvs.values())

    # ── DV01 ladder ──────────────────────────────────────────────────────────

    def dv01_ladder(
        self,
        key_tenors: Sequence[float] = KEY_RATE_TENORS,
        shift_bps: float = 1.0,
        width: float = 1.5,
    ) -> pd.DataFrame:
        """
        Key-rate DV01 ladder: P&L sensitivity to 1bp shift at each key tenor.

        Returns DataFrame with columns:
          tenor_yrs, total_dv01_usd, [one column per instrument]
        """
        rows = []
        for t in key_tenors:
            shifted = ScenarioEngine.key_rate_shift(self.curve, t, shift_bps, width)
            row     = {"tenor_yrs": t}
            total   = 0.0
            for name, instr in self.instruments.items():
                delta = instr.pv(shifted) - self._base_pvs[name]
                row[name] = delta
                total    += delta
            row["total_dv01_usd"] = total
            rows.append(row)
        return pd.DataFrame(rows)

    # ── Parallel shift P&L ───────────────────────────────────────────────────

    def parallel_pnl(
        self, shifts_bps: Sequence[float] = range(-100, 101, 10)
    ) -> pd.DataFrame:
        """P&L for a range of parallel curve shifts."""
        rows = []
        for s in shifts_bps:
            shifted = ScenarioEngine.parallel_shift(self.curve, float(s))
            total   = sum(i.pv(shifted) for i in self.instruments.values())
            rows.append({"shift_bps": s, "pnl_usd": total - self.total_pv})
        return pd.DataFrame(rows)

    # ── 2D scenario grid ─────────────────────────────────────────────────────

    def scenario_grid(
        self,
        short_shifts: Sequence[float] = [-50, -25, 0, 25, 50, 75, 100],
        long_shifts:  Sequence[float] = [-50, -25, 0, 25, 50, 75, 100],
        short_tenor:  float = 2.0,
        long_tenor:   float = 10.0,
    ) -> pd.DataFrame:
        """
        2D scenario grid of portfolio P&L vs independent 2Y and 10Y shifts.
        Returns DataFrame with short_shift_bps as index, long_shift_bps as columns.
        """
        grid = {}
        for s_long in long_shifts:
            col = {}
            for s_short in short_shifts:
                c1 = ScenarioEngine.key_rate_shift(self.curve, short_tenor, s_short, width=1.5)
                c2 = ScenarioEngine.key_rate_shift(c1, long_tenor, s_long, width=3.0)
                total = sum(i.pv(c2) for i in self.instruments.values())
                col[s_short] = total - self.total_pv
            grid[s_long] = col
        df = pd.DataFrame(grid)
        df.index.name   = "2Y_shift_bps"
        df.columns.name = "10Y_shift_bps"
        return df.round(0)

    # ── Historical VaR / ES ───────────────────────────────────────────────────

    @staticmethod
    def historical_var(
        daily_pnl: pd.Series,
        confidence: float = 0.99,
        horizon: int = 1,
    ) -> dict[str, float]:
        """
        Historical VaR and Expected Shortfall.

        Parameters
        ----------
        daily_pnl  : Series of daily P&L (bps or USD)
        confidence : VaR confidence level (default 99%)
        horizon    : holding period in days (scales by √horizon)

        Returns dict with:
          var_1d, var_Nd, es_1d, es_Nd, worst_day, best_day
        """
        clean = daily_pnl.dropna()
        scale = np.sqrt(horizon)
        var_1d = np.percentile(clean, (1 - confidence) * 100)
        es_1d  = clean[clean <= var_1d].mean()
        return {
            f"var_{horizon}d_{int(confidence*100)}pct": var_1d * scale,
            f"es_{horizon}d_{int(confidence*100)}pct":  es_1d  * scale,
            "var_1d_99pct": np.percentile(clean, 1),
            "worst_day":    clean.min(),
            "best_day":     clean.max(),
            "mean_daily":   clean.mean(),
            "vol_daily":    clean.std(),
        }

    # ── Stress tests ─────────────────────────────────────────────────────────

    def stress_test(self) -> pd.DataFrame:
        """
        Named stress scenarios representative of historical extremes.

        Scenarios
        ---------
        taper_tantrum_2013  : +100bp parallel
        fed_hike_2022       : +300bp parallel  (March→July 2022)
        covid_rally_2020    : -100bp parallel
        inversion_2023      : 2Y +50bp, 10Y -25bp (peak inversion)
        bull_steepening     : 2Y -75bp, 10Y -25bp (easing + term premium)
        bear_steepening     : 2Y -25bp, 10Y +50bp (growth re-pricing)
        butterfly_richening : wings -20bp, belly +30bp
        """
        scenarios = {
            "taper_tantrum_2013":  ScenarioEngine.parallel_shift(self.curve, +100),
            "fed_hike_2022":       ScenarioEngine.parallel_shift(self.curve, +300),
            "covid_rally_2020":    ScenarioEngine.parallel_shift(self.curve, -100),
            "inversion_2023":      ScenarioEngine.twist_shift(self.curve, +50, -25),
            "bull_steepening":     ScenarioEngine.twist_shift(self.curve, -75, -25),
            "bear_steepening":     ScenarioEngine.twist_shift(self.curve, -25, +50),
            "butterfly_richening": ScenarioEngine.butterfly_shift(self.curve, -20, +30),
        }
        rows = []
        for name, sc_curve in scenarios.items():
            total = sum(i.pv(sc_curve) for i in self.instruments.values())
            rows.append({"scenario": name, "pnl_usd": total - self.total_pv})
        return pd.DataFrame(rows).set_index("scenario")

    def summary(self) -> dict:
        """High-level risk summary."""
        ladder = self.dv01_ladder()
        return {
            "total_pv_usd":    self.total_pv,
            "total_dv01_usd":  ladder["total_dv01_usd"].sum(),
            "max_kr_dv01":     ladder["total_dv01_usd"].abs().max(),
            "max_kr_tenor":    ladder.loc[ladder["total_dv01_usd"].abs().idxmax(), "tenor_yrs"],
            "n_instruments":   len(self.instruments),
        }
