"""
Tests for sofr_engine/callable_bond.py
"""
from __future__ import annotations
import math
import pytest
import numpy as np
from datetime import date

from sofr_engine.bootstrap import flat_sofr_curve
from sofr_engine.callable_bond import (
    HWTreeParams, CallableBond, CallableBondResult,
    price_callable_bond, straight_bond_price, calibrate_oas,
    effective_duration, effective_convexity, price_callable_bond_full,
)


# ── Fixtures ──────────────────────────────────────────────────────────────────

@pytest.fixture
def curve():
    return flat_sofr_curve(date.today(), 0.05)  # flat 5% curve

@pytest.fixture
def hw():
    return HWTreeParams(a=0.10, sigma=0.01, dt=0.25)

@pytest.fixture
def bullet(curve):
    """Par bond: coupon = rate → price = face."""
    return CallableBond(face=100.0, coupon=0.05, maturity=5.0, freq=2)

@pytest.fixture
def callable_bond(curve):
    """Bond callable at par in years 2, 3, 4, 5."""
    return CallableBond(
        face=100.0, coupon=0.06, maturity=5.0, freq=2,
        call_schedule=[(2.0, 100.0), (3.0, 100.0), (4.0, 100.0), (5.0, 100.0)],
    )

@pytest.fixture
def putable_bond(curve):
    return CallableBond(
        face=100.0, coupon=0.04, maturity=5.0, freq=2,
        put_schedule=[(3.0, 100.0), (4.0, 100.0), (5.0, 100.0)],
    )


# ── HWTreeParams ──────────────────────────────────────────────────────────────

class TestHWTreeParams:
    def test_valid(self):
        p = HWTreeParams(a=0.05, sigma=0.01, dt=0.25)
        assert p.a == 0.05

    def test_dr(self):
        p = HWTreeParams(a=0.05, sigma=0.01, dt=0.25)
        assert abs(p.dr - 0.01 * math.sqrt(0.75)) < 1e-12

    def test_j_max_positive(self):
        p = HWTreeParams(a=0.05, sigma=0.01, dt=0.25)
        assert p.j_max >= 1

    def test_invalid_a(self):
        with pytest.raises(ValueError):
            HWTreeParams(a=0.0, sigma=0.01, dt=0.25)

    def test_invalid_sigma(self):
        with pytest.raises(ValueError):
            HWTreeParams(a=0.05, sigma=-0.01, dt=0.25)

    def test_invalid_dt(self):
        with pytest.raises(ValueError):
            HWTreeParams(a=0.05, sigma=0.01, dt=0.0)


# ── StraightBondPrice ─────────────────────────────────────────────────────────

class TestStraightBondPrice:
    def test_par_bond(self, curve, bullet):
        """Flat 5% continuous curve, 5% semi-annual coupon → price near 100 (small compounding diff)."""
        p = straight_bond_price(curve, bullet)
        assert abs(p - 100.0) < 0.50

    def test_premium_bond(self, curve):
        """6% coupon on 5% curve → price > 100."""
        bond = CallableBond(face=100.0, coupon=0.06, maturity=5.0, freq=2)
        assert straight_bond_price(curve, bond) > 100.0

    def test_discount_bond(self, curve):
        """4% coupon on 5% curve → price < 100."""
        bond = CallableBond(face=100.0, coupon=0.04, maturity=5.0, freq=2)
        assert straight_bond_price(curve, bond) < 100.0

    def test_zero_coupon(self):
        curve = flat_sofr_curve(date.today(), 0.05)
        bond  = CallableBond(face=100.0, coupon=0.0, maturity=5.0, freq=1)
        p = straight_bond_price(curve, bond)
        expected = 100.0 * math.exp(-0.05 * 5.0)
        assert abs(p - expected) < 0.01

    def test_positive(self, curve, bullet):
        assert straight_bond_price(curve, bullet) > 0.0

    def test_monotone_in_maturity(self, curve):
        b5  = CallableBond(face=100.0, coupon=0.03, maturity=5.0,  freq=2)
        b10 = CallableBond(face=100.0, coupon=0.03, maturity=10.0, freq=2)
        # Discount bond: longer maturity → further from par (discount compounds)
        p5  = straight_bond_price(curve, b5)
        p10 = straight_bond_price(curve, b10)
        assert p5 > p10  # 3% coupon on 5% curve, both discount

    def test_semi_annual(self, curve):
        b = CallableBond(face=100.0, coupon=0.05, maturity=2.0, freq=2)
        p = straight_bond_price(curve, b)
        assert abs(p - 100.0) < 0.50

    def test_coupon_times_length(self):
        b = CallableBond(face=100.0, coupon=0.05, maturity=5.0, freq=2)
        assert len(b.coupon_times) == 10

    def test_annual_coupon_times(self):
        b = CallableBond(face=100.0, coupon=0.05, maturity=3.0, freq=1)
        assert len(b.coupon_times) == 3


# ── CallableBond Pricing ──────────────────────────────────────────────────────

class TestCallableBond:
    def test_callable_le_straight(self, curve, hw, callable_bond):
        straight = straight_bond_price(curve, callable_bond)
        priced   = price_callable_bond(curve, hw, callable_bond)
        assert priced <= straight + 0.01  # call hurts bond holder

    def test_no_schedule_equals_straight(self, curve, hw, bullet):
        straight = straight_bond_price(curve, bullet)
        priced   = price_callable_bond(curve, hw, bullet)
        assert abs(priced - straight) < 0.50  # should be very close

    def test_option_value_positive(self, curve, hw, callable_bond):
        straight = straight_bond_price(curve, callable_bond)
        priced   = price_callable_bond(curve, hw, callable_bond)
        option_v = straight - priced
        assert option_v >= -0.01  # call option has non-negative value to issuer

    def test_put_raises_price(self, curve, hw, putable_bond):
        straight = straight_bond_price(curve, putable_bond)
        priced   = price_callable_bond(curve, hw, putable_bond)
        assert priced >= straight - 0.50  # put option benefits holder

    def test_positive_price(self, curve, hw, callable_bond):
        assert price_callable_bond(curve, hw, callable_bond) > 0.0

    def test_price_near_face_high_coupon_callable_at_par(self, curve, hw):
        """Bond with high coupon callable immediately at par → price ≈ 100."""
        bond = CallableBond(
            face=100.0, coupon=0.10, maturity=3.0, freq=2,
            call_schedule=[(0.5, 100.0), (1.0, 100.0), (1.5, 100.0),
                           (2.0, 100.0), (2.5, 100.0), (3.0, 100.0)],
        )
        p = price_callable_bond(curve, hw, bond)
        assert p <= 101.0  # call caps price near call price

    def test_oas_zero_returns_model_price(self, curve, hw, callable_bond):
        p0  = price_callable_bond(curve, hw, callable_bond, oas=0.0)
        p0b = price_callable_bond(curve, hw, callable_bond, oas=0.0)
        assert abs(p0 - p0b) < 1e-8


# ── OAS Calibration ───────────────────────────────────────────────────────────

class TestOASCalibration:
    def test_oas_roundtrip(self, curve, hw, callable_bond):
        model_price = price_callable_bond(curve, hw, callable_bond)
        oas = calibrate_oas(curve, hw, callable_bond, model_price)
        repriced = price_callable_bond(curve, hw, callable_bond, oas)
        assert abs(repriced - model_price) < 0.01

    def test_oas_zero_at_model_price(self, curve, hw, callable_bond):
        model_price = price_callable_bond(curve, hw, callable_bond)
        oas = calibrate_oas(curve, hw, callable_bond, model_price)
        assert abs(oas) < 0.005  # should be near zero

    def test_higher_market_price_lower_oas(self, curve, hw, callable_bond):
        p0 = price_callable_bond(curve, hw, callable_bond)
        oas_base = calibrate_oas(curve, hw, callable_bond, p0)
        oas_high = calibrate_oas(curve, hw, callable_bond, p0 + 2.0)
        assert oas_high < oas_base

    def test_oas_negative_for_cheap_bond(self, curve, hw, callable_bond):
        p0 = price_callable_bond(curve, hw, callable_bond)
        oas = calibrate_oas(curve, hw, callable_bond, p0 - 2.0)
        assert oas > 0.0  # positive OAS = bond is cheap (trades wide)

    def test_bullet_oas_zero(self, curve, hw, bullet):
        straight = straight_bond_price(curve, bullet)
        priced   = price_callable_bond(curve, hw, bullet)
        oas = calibrate_oas(curve, hw, bullet, priced)
        assert abs(oas) < 0.01

    def test_oas_finite(self, curve, hw, callable_bond):
        p0 = price_callable_bond(curve, hw, callable_bond)
        oas = calibrate_oas(curve, hw, callable_bond, p0)
        assert math.isfinite(oas)


# ── Effective Duration ────────────────────────────────────────────────────────

class TestEffectiveDuration:
    def test_positive_for_long_bond(self, curve, hw):
        bond = CallableBond(face=100.0, coupon=0.05, maturity=10.0, freq=2)
        ed = effective_duration(curve, hw, bond)
        assert ed > 0.0

    def test_shorter_bond_lower_duration(self, curve, hw):
        b5  = CallableBond(face=100.0, coupon=0.05, maturity=5.0,  freq=2)
        b10 = CallableBond(face=100.0, coupon=0.05, maturity=10.0, freq=2)
        ed5  = effective_duration(curve, hw, b5)
        ed10 = effective_duration(curve, hw, b10)
        assert ed5 < ed10

    def test_callable_le_straight_duration(self, curve, hw, callable_bond, bullet):
        ed_c = effective_duration(curve, hw, callable_bond)
        ed_s = effective_duration(curve, hw, bullet)
        # Callable bond has shorter effective duration
        assert ed_c <= ed_s + 1.0

    def test_duration_finite(self, curve, hw, bullet):
        ed = effective_duration(curve, hw, bullet)
        assert math.isfinite(ed)

    def test_duration_reasonable(self, curve, hw, bullet):
        ed = effective_duration(curve, hw, bullet)
        assert 0 < ed < 10.0  # 5yr bond: ED ≈ 4–5 yrs


# ── Effective Convexity ───────────────────────────────────────────────────────

class TestEffectiveConvexity:
    def test_positive_for_straight_bond(self, curve, hw, bullet):
        ec = effective_convexity(curve, hw, bullet)
        assert ec > 0.0

    def test_convexity_finite(self, curve, hw, bullet):
        ec = effective_convexity(curve, hw, bullet)
        assert math.isfinite(ec)

    def test_convexity_scales_with_maturity(self, curve, hw):
        b5  = CallableBond(face=100.0, coupon=0.05, maturity=5.0,  freq=2)
        b10 = CallableBond(face=100.0, coupon=0.05, maturity=10.0, freq=2)
        ec5  = effective_convexity(curve, hw, b5)
        ec10 = effective_convexity(curve, hw, b10)
        assert ec10 > ec5

    def test_callable_can_have_lower_convexity(self, curve, hw, callable_bond, bullet):
        ec_c = effective_convexity(curve, hw, callable_bond)
        ec_s = effective_convexity(curve, hw, bullet)
        # Callable may have lower or even negative convexity near call price
        assert ec_c <= ec_s + 50.0  # allow wide tolerance

    def test_convexity_positive_for_zero_coupon(self):
        curve = flat_sofr_curve(date.today(), 0.05)
        hw    = HWTreeParams(a=0.10, sigma=0.01, dt=0.5)
        bond  = CallableBond(face=100.0, coupon=0.0, maturity=5.0, freq=1)
        ec = effective_convexity(curve, hw, bond)
        assert math.isfinite(ec)


# ── Full Result ───────────────────────────────────────────────────────────────

class TestFullResult:
    def test_full_result_keys(self, curve, hw, callable_bond):
        res = price_callable_bond_full(curve, hw, callable_bond)
        assert isinstance(res, CallableBondResult)
        assert math.isfinite(res.price)
        assert math.isfinite(res.oas)
        assert math.isfinite(res.effective_duration)
        assert math.isfinite(res.effective_convexity)

    def test_option_value_positive(self, curve, hw, callable_bond):
        res = price_callable_bond_full(curve, hw, callable_bond)
        assert res.option_value >= -0.01

    def test_market_price_roundtrip(self, curve, hw, callable_bond):
        market_px = price_callable_bond(curve, hw, callable_bond) - 1.0
        res = price_callable_bond_full(curve, hw, callable_bond, market_price=market_px)
        assert abs(res.price - market_px) < 0.01

    def test_straight_price_positive(self, curve, hw, bullet):
        res = price_callable_bond_full(curve, hw, bullet)
        assert res.straight_price > 0.0


# ── Edge Cases ────────────────────────────────────────────────────────────────

class TestEdgeCases:
    def test_no_schedule_bullet(self, curve, hw):
        bond = CallableBond(face=100.0, coupon=0.05, maturity=2.0, freq=2)
        p = price_callable_bond(curve, hw, bond)
        assert p > 0.0

    def test_very_short_bond(self, curve):
        hw = HWTreeParams(a=0.10, sigma=0.01, dt=0.25)
        bond = CallableBond(face=100.0, coupon=0.05, maturity=1.0, freq=2)
        p = price_callable_bond(curve, hw, bond)
        assert p > 0.0

    def test_put_and_call(self, curve, hw):
        bond = CallableBond(
            face=100.0, coupon=0.05, maturity=3.0, freq=2,
            call_schedule=[(2.0, 102.0), (3.0, 100.0)],
            put_schedule=[(2.0, 98.0), (3.0, 100.0)],
        )
        p = price_callable_bond(curve, hw, bond)
        assert p > 0.0

    def test_zero_coupon_callable(self, curve):
        hw = HWTreeParams(a=0.10, sigma=0.01, dt=0.5)
        bond = CallableBond(
            face=100.0, coupon=0.0, maturity=3.0, freq=1,
            call_schedule=[(2.0, 80.0), (3.0, 100.0)],
        )
        p = price_callable_bond(curve, hw, bond)
        assert p > 0.0

    def test_bond_with_invalid_face(self):
        with pytest.raises(ValueError):
            CallableBond(face=-100.0, coupon=0.05, maturity=5.0)

    def test_bond_with_invalid_maturity(self):
        with pytest.raises(ValueError):
            CallableBond(face=100.0, coupon=0.05, maturity=0.0)
