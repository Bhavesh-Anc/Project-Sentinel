"""
Tests for sofr_engine.inflation — Jarrow-Yildirim inflation model.

Covers:
  - InflationCurve construction and analytics
  - ZC inflation swap pricing
  - YoY inflation swap pricing
  - Inflation caplet / cap / floor pricing
  - Cap-floor parity
  - Curve calibration
"""
from __future__ import annotations

import math
import pytest
import numpy as np
from datetime import date

from sofr_engine.bootstrap import flat_sofr_curve
from sofr_engine.inflation import (
    InflationCurve,
    ZCInflationSwap,
    YoYInflationSwap,
    InflationCapFloor,
    ZCInflationResult,
    YoYResult,
    InflationCapResult,
    zc_inflation_pv,
    yoy_inflation_pv,
    inflation_caplet_pv,
    inflation_cap_floor_pv,
    inflation_cap_floor_parity,
    breakeven_inflation,
    calibrate_inflation_curve,
)

# ── Fixtures / helpers ────────────────────────────────────────────────────────

TODAY = date.today()
MU = 0.025   # 2.5 % flat inflation
R_NOM = 0.04  # 4 % flat nominal rate

def nom_curve(rate: float = R_NOM):
    return flat_sofr_curve(TODAY, rate)

def infl_curve(rate: float = MU):
    return InflationCurve.flat(rate)


# ═════════════════════════════════════════════════════════════════════════════
# TestInflationCurve
# ═════════════════════════════════════════════════════════════════════════════

class TestInflationCurve:

    def test_flat_constructor(self):
        """InflationCurve.flat(0.025) creates a valid single-pillar curve."""
        c = InflationCurve.flat(0.025)
        assert len(c.times) == 1
        assert len(c.rates) == 1
        assert c.rates[0] == pytest.approx(0.025)

    def test_forward_inflation_positive(self):
        """forward_inflation returns the positive rate for a positive flat curve."""
        c = infl_curve(0.03)
        assert c.forward_inflation(5.0) == pytest.approx(0.03)
        assert c.forward_inflation(1.0) == pytest.approx(0.03)

    def test_cpi_ratio_greater_than_one(self):
        """cpi_ratio(5) > 1 for a positive inflation rate."""
        c = infl_curve(0.025)
        assert c.cpi_ratio(5.0) > 1.0

    def test_cpi_ratio_at_zero(self):
        """cpi_ratio near T=0 is approximately 1."""
        c = infl_curve(0.025)
        assert c.cpi_ratio(1e-9) == pytest.approx(1.0, abs=1e-6)

    def test_cpi_ratio_monotone(self):
        """For positive inflation, cpi_ratio is monotonically increasing."""
        c = infl_curve(0.025)
        assert c.cpi_ratio(10.0) > c.cpi_ratio(5.0)

    def test_yoy_forward_positive(self):
        """yoy_forward(1, 2) > 0 for a positive inflation rate."""
        c = infl_curve(0.025)
        assert c.yoy_forward(1.0, 2.0) > 0.0

    def test_yoy_forward_flat(self):
        """For a flat curve at rate mu, yoy_forward(T1, T2) == (1+mu)^(T2-T1) - 1."""
        mu = 0.03
        c = infl_curve(mu)
        T1, T2 = 2.0, 5.0
        expected = (1 + mu) ** (T2 - T1) - 1.0
        assert c.yoy_forward(T1, T2) == pytest.approx(expected, rel=1e-8)

    def test_par_fixed_rate_approx(self):
        """par_fixed_rate(T) ~ mu for a flat curve (exact for simple compounding)."""
        mu = 0.025
        c = infl_curve(mu)
        # For a flat curve: CPI_ratio(T) = (1+mu)^T, so par_rate = mu exactly
        assert c.par_fixed_rate(10.0) == pytest.approx(mu, rel=1e-8)

    def test_par_fixed_rate_positive(self):
        """par_fixed_rate(5) > 0 for a positive inflation curve."""
        c = infl_curve(0.02)
        assert c.par_fixed_rate(5.0) > 0.0

    def test_invalid_rates_range(self):
        """ValueError if any rate exceeds 50%."""
        with pytest.raises(ValueError):
            InflationCurve(times=np.array([1.0]), rates=np.array([0.6]))

    def test_invalid_mismatched_lengths(self):
        """ValueError if times and rates have different lengths."""
        with pytest.raises(ValueError):
            InflationCurve(times=np.array([1.0, 2.0]), rates=np.array([0.025]))

    def test_calibrate_reprices(self):
        """calibrate_inflation_curve followed by par_fixed_rate recovers inputs."""
        maturities = [1.0, 5.0, 10.0]
        par_rates   = [0.020, 0.025, 0.028]
        nc = nom_curve()
        c = calibrate_inflation_curve(nc, maturities, par_rates)
        for T, K in zip(maturities, par_rates):
            assert c.par_fixed_rate(T) == pytest.approx(K, rel=1e-8)


# ═════════════════════════════════════════════════════════════════════════════
# TestZCInflationPV
# ═════════════════════════════════════════════════════════════════════════════

class TestZCInflationPV:

    def _swap(self, maturity=10.0, fixed_rate=0.020, notional=1_000_000.0, receive=True):
        return ZCInflationSwap(
            maturity=maturity,
            fixed_rate=fixed_rate,
            notional=notional,
            receive_inflation=receive,
        )

    def test_pv_positive_itm(self):
        """receive_inflation=True, K < par_rate => pv > 0."""
        nc = nom_curve()
        ic = infl_curve(0.025)
        swap = self._swap(fixed_rate=0.010)  # K << par
        result = zc_inflation_pv(nc, ic, swap)
        assert result.pv > 0.0

    def test_pv_negative_otm(self):
        """receive_inflation=True, K > par_rate => pv < 0."""
        nc = nom_curve()
        ic = infl_curve(0.025)
        swap = self._swap(fixed_rate=0.040)  # K >> par
        result = zc_inflation_pv(nc, ic, swap)
        assert result.pv < 0.0

    def test_pv_at_par(self):
        """K = par_rate => pv ~ 0."""
        nc = nom_curve()
        ic = infl_curve(0.025)
        par = ic.par_fixed_rate(10.0)
        swap = self._swap(fixed_rate=par)
        result = zc_inflation_pv(nc, ic, swap)
        assert result.pv == pytest.approx(0.0, abs=1e-6)

    def test_float_leg_positive(self):
        """float_leg_pv > 0 for positive inflation rate."""
        nc = nom_curve()
        ic = infl_curve(0.025)
        result = zc_inflation_pv(nc, ic, self._swap())
        assert result.float_leg_pv > 0.0

    def test_fixed_leg_positive(self):
        """fixed_leg_pv > 0 for K > 0."""
        nc = nom_curve()
        ic = infl_curve(0.025)
        result = zc_inflation_pv(nc, ic, self._swap(fixed_rate=0.020))
        assert result.fixed_leg_pv > 0.0

    def test_receive_vs_pay(self):
        """pv(receive) + pv(pay) == 0."""
        nc = nom_curve()
        ic = infl_curve(0.025)
        r = zc_inflation_pv(nc, ic, self._swap(receive=True))
        p = zc_inflation_pv(nc, ic, self._swap(receive=False))
        assert r.pv + p.pv == pytest.approx(0.0, abs=1e-8)

    def test_notional_scales(self):
        """pv with 2 M notional == 2 * pv with 1 M notional."""
        nc = nom_curve()
        ic = infl_curve(0.025)
        r1 = zc_inflation_pv(nc, ic, self._swap(notional=1_000_000.0))
        r2 = zc_inflation_pv(nc, ic, self._swap(notional=2_000_000.0))
        assert r2.pv == pytest.approx(2 * r1.pv, rel=1e-10)

    def test_longer_maturity_larger_pv(self):
        """10yr swap (same rate gap) has larger |pv| than 5yr swap."""
        nc = nom_curve()
        ic = infl_curve(0.025)
        r5  = zc_inflation_pv(nc, ic, self._swap(maturity=5.0,  fixed_rate=0.010))
        r10 = zc_inflation_pv(nc, ic, self._swap(maturity=10.0, fixed_rate=0.010))
        assert abs(r10.pv) > abs(r5.pv)

    def test_breakeven_bps_at_par(self):
        """breakeven_bps ~ 0 when K = par_rate."""
        nc = nom_curve()
        ic = infl_curve(0.025)
        par = ic.par_fixed_rate(10.0)
        result = zc_inflation_pv(nc, ic, self._swap(fixed_rate=par))
        assert result.breakeven_bps == pytest.approx(0.0, abs=1e-6)

    def test_result_fields(self):
        """ZCInflationResult has all required fields."""
        nc = nom_curve()
        ic = infl_curve(0.025)
        result = zc_inflation_pv(nc, ic, self._swap())
        assert hasattr(result, "pv")
        assert hasattr(result, "float_leg_pv")
        assert hasattr(result, "fixed_leg_pv")
        assert hasattr(result, "par_rate")
        assert hasattr(result, "cpi_ratio")
        assert hasattr(result, "breakeven_bps")

    def test_zero_inflation_float_leg(self):
        """rate=0 => float_leg_pv ~ 0 (cpi_ratio ~ 1 => float = N*P*(1-1) = 0)."""
        nc = nom_curve()
        ic = infl_curve(0.0)
        result = zc_inflation_pv(nc, ic, self._swap(fixed_rate=0.0))
        assert result.float_leg_pv == pytest.approx(0.0, abs=1e-6)

    def test_higher_inflation_higher_float(self):
        """Higher inflation rate => higher float_leg_pv."""
        nc = nom_curve()
        ic_lo = infl_curve(0.01)
        ic_hi = infl_curve(0.04)
        r_lo = zc_inflation_pv(nc, ic_lo, self._swap())
        r_hi = zc_inflation_pv(nc, ic_hi, self._swap())
        assert r_hi.float_leg_pv > r_lo.float_leg_pv


# ═════════════════════════════════════════════════════════════════════════════
# TestYoYInflationPV
# ═════════════════════════════════════════════════════════════════════════════

class TestYoYInflationPV:

    DATES = [1.0, 2.0, 3.0, 4.0, 5.0]  # annual payment dates

    def _swap(self, fixed_rate=MU, notional=1_000_000.0, receive=True, dates=None):
        if dates is None:
            dates = self.DATES
        return YoYInflationSwap(
            payment_dates=dates,
            fixed_rate=fixed_rate,
            notional=notional,
            receive_inflation=receive,
        )

    def test_pv_positive_itm(self):
        """K < mu, receive_inflation=True => pv > 0."""
        nc = nom_curve()
        ic = infl_curve(MU)
        result = yoy_inflation_pv(nc, ic, self._swap(fixed_rate=0.005))
        assert result.pv > 0.0

    def test_pv_negative_otm(self):
        """K > mu => pv < 0."""
        nc = nom_curve()
        ic = infl_curve(MU)
        result = yoy_inflation_pv(nc, ic, self._swap(fixed_rate=0.10))
        assert result.pv < 0.0

    def test_pv_at_par(self):
        """K = mu: for flat curve the YoY ~ mu, so pv ~ 0."""
        nc = nom_curve()
        ic = infl_curve(MU)
        # For flat curve yoy_forward(T-1, T) = (1+mu)^1 - 1 = mu exactly
        result = yoy_inflation_pv(nc, ic, self._swap(fixed_rate=MU))
        assert result.pv == pytest.approx(0.0, abs=1e-6)

    def test_receive_vs_pay(self):
        """pv(receive) + pv(pay) == 0."""
        nc = nom_curve()
        ic = infl_curve(MU)
        r = yoy_inflation_pv(nc, ic, self._swap(receive=True))
        p = yoy_inflation_pv(nc, ic, self._swap(receive=False))
        assert r.pv + p.pv == pytest.approx(0.0, abs=1e-8)

    def test_n_periods(self):
        """n_periods == len(payment_dates)."""
        nc = nom_curve()
        ic = infl_curve(MU)
        result = yoy_inflation_pv(nc, ic, self._swap())
        assert result.n_periods == len(self.DATES)

    def test_legs_positive(self):
        """float_leg_pv > 0 and fixed_leg_pv > 0 for positive rates."""
        nc = nom_curve()
        ic = infl_curve(MU)
        result = yoy_inflation_pv(nc, ic, self._swap(fixed_rate=0.01))
        assert result.float_leg_pv > 0.0
        assert result.fixed_leg_pv > 0.0

    def test_notional_scales(self):
        """pv with 2 M notional == 2 * pv with 1 M notional."""
        nc = nom_curve()
        ic = infl_curve(MU)
        r1 = yoy_inflation_pv(nc, ic, self._swap(notional=1_000_000.0))
        r2 = yoy_inflation_pv(nc, ic, self._swap(notional=2_000_000.0))
        assert r2.pv == pytest.approx(2 * r1.pv, rel=1e-10)

    def test_zero_inflation_float_leg(self):
        """rate=0 => yoy_forward = 0 => float_leg ~ 0."""
        nc = nom_curve()
        ic = infl_curve(0.0)
        result = yoy_inflation_pv(nc, ic, self._swap(fixed_rate=0.0))
        assert result.float_leg_pv == pytest.approx(0.0, abs=1e-8)


# ═════════════════════════════════════════════════════════════════════════════
# TestInflationCaplet
# ═════════════════════════════════════════════════════════════════════════════

class TestInflationCaplet:

    def _caplet(self, T_start=1.0, T_end=2.0, strike=MU, vol=0.10,
                notional=1_000_000.0, is_cap=True):
        nc = nom_curve()
        ic = infl_curve(MU)
        return inflation_caplet_pv(nc, ic, T_start, T_end, strike, vol, notional, is_cap)

    def test_caplet_itm_positive(self):
        """ITM caplet (strike << forward) > 0."""
        pv = self._caplet(strike=0.001)
        assert pv > 0.0

    def test_floorlet_positive_otm_cap_side(self):
        """OTM floorlet (strike << forward, is_cap=False) > 0 due to optionality."""
        pv = self._caplet(strike=0.001, is_cap=False)
        assert pv > 0.0

    def test_put_call_parity_single(self):
        """
        Single-period put-call parity:
        caplet - floorlet ~ F * P - K_gross * P
        where F = forward CPI ratio, K_gross = (1+strike)^alpha.
        """
        nc = nom_curve()
        ic = infl_curve(MU)
        T_start, T_end = 1.0, 2.0
        strike = MU
        vol = 0.10
        notional = 1.0
        cap_pv  = inflation_caplet_pv(nc, ic, T_start, T_end, strike, vol, notional, True)
        floor_pv = inflation_caplet_pv(nc, ic, T_start, T_end, strike, vol, notional, False)
        F = ic.cpi_ratio(T_end) / ic.cpi_ratio(T_start)
        alpha = T_end - T_start
        K_gross = (1 + strike) ** alpha
        P = nc.df(T_end)
        expected = (F - K_gross) * P
        assert (cap_pv - floor_pv) == pytest.approx(expected, rel=1e-6)

    def test_atm_cap_positive(self):
        """ATM caplet has positive value."""
        F_ratio = infl_curve(MU).cpi_ratio(2.0) / infl_curve(MU).cpi_ratio(1.0)
        alpha = 1.0
        # ATM strike: (1+K)^alpha = F_ratio => K = F_ratio^(1/alpha) - 1
        K_atm = F_ratio ** (1.0 / alpha) - 1.0
        pv = self._caplet(T_start=1.0, T_end=2.0, strike=K_atm, vol=0.10)
        assert pv > 0.0

    def test_deep_itm_cap_approx_intrinsic(self):
        """Deep ITM caplet with low vol ~ intrinsic value."""
        nc = nom_curve()
        ic = infl_curve(MU)
        # Very low strike => deeply ITM
        T_start, T_end = 1.0, 2.0
        strike = -0.05
        vol_low = 0.001
        vol_hi  = 0.20
        pv_low = inflation_caplet_pv(nc, ic, T_start, T_end, strike, vol_low, 1.0, True)
        pv_hi  = inflation_caplet_pv(nc, ic, T_start, T_end, strike, vol_hi,  1.0, True)
        # Low vol version should be close to intrinsic, high vol > low vol
        F = ic.cpi_ratio(T_end) / ic.cpi_ratio(T_start)
        alpha = T_end - T_start
        K_gross = (1 + strike) ** alpha
        P = nc.df(T_end)
        intrinsic = (F - K_gross) * P
        assert pv_low == pytest.approx(intrinsic, rel=1e-2)
        assert pv_hi >= pv_low

    def test_deep_otm_cap_near_zero(self):
        """Deep OTM caplet (very high strike, unit notional) ~ 0."""
        # Use notional=1.0 so the small Black probability is not amplified
        nc = nom_curve()
        ic = infl_curve(MU)
        # strike=0.30 >> mu=0.025: deeply OTM for a 1-year caplet
        pv = inflation_caplet_pv(nc, ic, 1.0, 2.0, 0.30, 0.10, 1.0, True)
        assert pv < 1e-3  # tiny fraction of unit notional

    def test_higher_vol_higher_cap(self):
        """Higher vol => higher ATM caplet value."""
        pv_lo = self._caplet(vol=0.05)
        pv_hi = self._caplet(vol=0.20)
        assert pv_hi > pv_lo

    def test_longer_T_start_higher_cap(self):
        """Longer T_start increases uncertainty (higher cap), all else equal."""
        nc = nom_curve()
        ic = infl_curve(MU)
        # Same one-year period but different reset dates
        pv_near = inflation_caplet_pv(nc, ic, 2.0, 3.0, MU, 0.10, 1.0, True)
        pv_far  = inflation_caplet_pv(nc, ic, 4.0, 5.0, MU, 0.10, 1.0, True)
        assert pv_far > pv_near

    def test_notional_scales(self):
        """Caplet PV scales linearly with notional."""
        pv1 = self._caplet(notional=1_000_000.0)
        pv2 = self._caplet(notional=2_000_000.0)
        assert pv2 == pytest.approx(2 * pv1, rel=1e-10)

    def test_vol_zero_intrinsic(self):
        """vol=0 gives intrinsic value (discounted)."""
        nc = nom_curve()
        ic = infl_curve(MU)
        T_start, T_end = 1.0, 2.0
        strike = MU * 0.5  # ITM
        pv = inflation_caplet_pv(nc, ic, T_start, T_end, strike, 0.0, 1.0, True)
        F = ic.cpi_ratio(T_end) / ic.cpi_ratio(T_start)
        alpha = T_end - T_start
        K_gross = (1 + strike) ** alpha
        P = nc.df(T_end)
        intrinsic = max(F - K_gross, 0.0) * P
        assert pv == pytest.approx(intrinsic, rel=1e-8)


# ═════════════════════════════════════════════════════════════════════════════
# TestInflationCapFloor
# ═════════════════════════════════════════════════════════════════════════════

class TestInflationCapFloor:

    DATES = [1.0, 2.0, 3.0, 4.0, 5.0]
    VOL   = 0.10

    def _cap(self, strike=MU, notional=1_000_000.0, is_cap=True):
        return InflationCapFloor(
            payment_dates=self.DATES,
            strike=strike,
            vol=self.VOL,
            notional=notional,
            is_cap=is_cap,
        )

    def test_cap_equals_sum_caplets(self):
        """cap.pv == sum(caplet_pvs)."""
        nc = nom_curve()
        ic = infl_curve(MU)
        result = inflation_cap_floor_pv(nc, ic, self._cap())
        assert result.pv == pytest.approx(sum(result.caplet_pvs), rel=1e-10)

    def test_floor_positive(self):
        """Floor with OTM strike (strike << forward) has positive value from optionality."""
        nc = nom_curve()
        ic = infl_curve(MU)
        # Strike much higher than inflation => floor is ITM
        floor = self._cap(strike=0.10, is_cap=False)
        result = inflation_cap_floor_pv(nc, ic, floor)
        assert result.pv > 0.0

    def test_cap_floor_parity(self):
        """inflation_cap_floor_parity returns near-zero (within 1e-6 * notional)."""
        nc = nom_curve()
        ic = infl_curve(MU)
        notional = 1_000_000.0
        diff = inflation_cap_floor_parity(nc, ic, self.DATES, MU, self.VOL, notional)
        assert abs(diff) < 1e-6 * notional

    def test_notional_scales(self):
        """Cap PV scales linearly with notional."""
        nc = nom_curve()
        ic = infl_curve(MU)
        r1 = inflation_cap_floor_pv(nc, ic, self._cap(notional=1_000_000.0))
        r2 = inflation_cap_floor_pv(nc, ic, self._cap(notional=2_000_000.0))
        assert r2.pv == pytest.approx(2 * r1.pv, rel=1e-10)

    def test_strike_below_par_cap_itm(self):
        """Lower strike => higher cap PV (deeper ITM)."""
        nc = nom_curve()
        ic = infl_curve(MU)
        r_lo = inflation_cap_floor_pv(nc, ic, self._cap(strike=0.005))
        r_hi = inflation_cap_floor_pv(nc, ic, self._cap(strike=0.04))
        assert r_lo.pv > r_hi.pv

    def test_strike_above_par_floor_itm(self):
        """Higher strike => higher floor PV (deeper ITM for floor)."""
        nc = nom_curve()
        ic = infl_curve(MU)
        r_lo = inflation_cap_floor_pv(nc, ic, self._cap(strike=0.005, is_cap=False))
        r_hi = inflation_cap_floor_pv(nc, ic, self._cap(strike=0.04,  is_cap=False))
        assert r_hi.pv > r_lo.pv

    def test_correct_number_caplets(self):
        """len(caplet_pvs) == len(payment_dates)."""
        nc = nom_curve()
        ic = infl_curve(MU)
        result = inflation_cap_floor_pv(nc, ic, self._cap())
        assert len(result.caplet_pvs) == len(self.DATES)

    def test_result_fields(self):
        """InflationCapResult has pv, caplet_pvs, is_cap."""
        nc = nom_curve()
        ic = infl_curve(MU)
        result = inflation_cap_floor_pv(nc, ic, self._cap())
        assert hasattr(result, "pv")
        assert hasattr(result, "caplet_pvs")
        assert hasattr(result, "is_cap")
        assert result.is_cap is True


# ═════════════════════════════════════════════════════════════════════════════
# TestCalibration
# ═════════════════════════════════════════════════════════════════════════════

class TestCalibration:

    MATURITIES  = [1.0, 2.0, 5.0, 10.0, 30.0]
    PAR_RATES   = [0.020, 0.022, 0.025, 0.027, 0.030]

    def test_reprices_first_maturity(self):
        """Calibrated curve reprices first maturity exactly."""
        nc = nom_curve()
        c = calibrate_inflation_curve(nc, self.MATURITIES, self.PAR_RATES)
        assert c.par_fixed_rate(self.MATURITIES[0]) == pytest.approx(self.PAR_RATES[0], rel=1e-8)

    def test_reprices_all_maturities(self):
        """Calibrated curve reprices all maturities exactly."""
        nc = nom_curve()
        c = calibrate_inflation_curve(nc, self.MATURITIES, self.PAR_RATES)
        for T, K in zip(self.MATURITIES, self.PAR_RATES):
            assert c.par_fixed_rate(T) == pytest.approx(K, rel=1e-8), \
                f"Failed at T={T}: got {c.par_fixed_rate(T):.6f}, expected {K:.6f}"

    def test_monotone_par_rates(self):
        """Monotone increasing par rates produce a valid (non-negative) forward curve."""
        nc = nom_curve()
        par_rates_mono = [0.020, 0.022, 0.024, 0.026, 0.028]
        c = calibrate_inflation_curve(nc, self.MATURITIES, par_rates_mono)
        # All forward rates should be positive (non-negative)
        assert np.all(c.rates >= 0.0)

    def test_mismatched_lengths_raises(self):
        """ValueError if maturities and zc_par_rates have different lengths."""
        nc = nom_curve()
        with pytest.raises(ValueError, match="same length"):
            calibrate_inflation_curve(nc, [1.0, 2.0], [0.025])

    def test_returns_inflation_curve(self):
        """calibrate_inflation_curve returns an InflationCurve instance."""
        nc = nom_curve()
        result = calibrate_inflation_curve(nc, self.MATURITIES, self.PAR_RATES)
        assert isinstance(result, InflationCurve)
